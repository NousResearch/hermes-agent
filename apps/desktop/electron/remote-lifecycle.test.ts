import assert from 'node:assert/strict'
import { exec as execCallback, spawn } from 'node:child_process'
import { chmod, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { profileSshOverride } from './connection-config'
import {
  assertRemoteInstallUpdateClear,
  buildSpawnCommand,
  classifySshReuseProof,
  cleanupStale,
  connect,
  disconnect,
  expandRemotePath,
  fingerprintToken,
  isForwardBindCollision,
  isLockfileSkew,
  listRemoteHermesProfiles,
  locateHermes,
  LOCKFILE_SCHEMA_VERSION,
  lockfilePath,
  openForward,
  ownershipDirectory,
  pidIsOurDashboard,
  probeRemotePlatform,
  PROTOCOL_VERSION,
  readLockfile,
  readRemoteInstallId,
  READY_RE,
  remotePidAlive,
  scrapeReadyPort,
  spawnLogPath,
  spawnRemoteDashboard,
  spawnTokenPath,
  terminateOwnedDashboardForUpdate,
  validateRemotePath,
  writeLockfile
} from './remote-lifecycle'
import { connectDeps, fakeSsh, ownedLock, OWNERSHIP_ID, SPAWN_NONCE } from './remote-lifecycle.test-helpers'

const exec = promisify(execCallback)

test('SSH reuse proof rejects a backend whose runtime was replaced', () => {
  assert.equal(
    classifySshReuseProof(
      { ok: true, sshOwnerNonce: SPAWN_NONCE, protocolVersion: 1, runtimeIntact: false },
      SPAWN_NONCE
    ),
    'authenticated-stale'
  )
})

test('SSH reuse proof remains compatible when runtime state is absent', () => {
  assert.equal(
    classifySshReuseProof({ ok: true, sshOwnerNonce: SPAWN_NONCE, protocolVersion: 1 }, SPAWN_NONCE),
    'authenticated-ok'
  )
})

test('POSIX relaunch gate refuses live and uncertain install markers without executing Hermes', async () => {
  for (const observation of ['LIVE:4242', 'UNCERTAIN']) {
    const calls: string[] = []

    const ssh = {
      async exec(command) {
        calls.push(command)

        if (command === 'uname -s; uname -m') {
          return 'Linux\nx86_64\n'
        }

        if (command.includes('HERMES_HOME')) {
          return '/home/alice/.hermes\n'
        }

        if (command.includes('.hermes-update-in-progress')) {
          return observation
        }

        throw new Error(`unexpected command after update gate: ${command}`)
      }
    }

    await assert.rejects(
      () => connect(connectDeps(ssh)),
      (error: any) => error.kind === 'update-in-progress'
    )
    assert.equal(
      calls.some(command => /\[ -x |--version|lock\.json|serve --help|setsid/.test(command)),
      false
    )
  }
})

test('POSIX relaunch gate permits absent/dead markers and normalizes named-profile homes install-wide', async () => {
  const commands: string[] = []

  const ssh = {
    async exec(command) {
      commands.push(command)

      return 'CLEAR'
    }
  }

  await assertRemoteInstallUpdateClear(ssh, '/home/alice/.hermes/profiles/research')
  assert.match(commands[0], /home\.parent\.name/)
  assert.match(commands[0], /profiles/)
  assert.match(commands[0], /\.hermes-update-in-progress/)
})

test('POSIX relaunch gate rechecks after token upload immediately before process creation', async () => {
  const calls: string[] = []
  let markerChecks = 0

  const ssh = {
    async exec(command) {
      calls.push(command)

      if (command === 'uname -s; uname -m') {
        return 'Linux\nx86_64\n'
      }

      if (command.includes('HERMES_HOME')) {
        return '/home/alice/.hermes\n'
      }

      if (command.includes('.hermes-update-in-progress')) {
        markerChecks += 1

        return markerChecks >= 3 ? 'LIVE:4242' : 'CLEAR'
      }

      if (/\[ -x /.test(command)) {
        return 'OK'
      }

      if (command.includes('serve --help')) {
        return 'YES\n'
      }

      if (command.includes('python3 -c')) {
        return ''
      }

      if (command.includes('lock.json')) {
        return ''
      }

      return ''
    }
  }

  await assert.rejects(
    () => connect(connectDeps(ssh)),
    (error: any) => error.kind === 'update-in-progress'
  )
  assert.equal(markerChecks, 3)
  assert.equal(
    calls.some(command => /setsid|nohup/.test(command)),
    false
  )
})

test('readRemoteInstallId reads the backend identity without spawning a dashboard or minting one', async () => {
  const ssh = fakeSsh([
    [/HERMES_HOME/, '/Users/zillajr/.hermes\n'],
    [/cat .*install_id/, '0f8a1c2b3d4e5f60718293a4b5c6d7e8\n']
  ])

  assert.equal(await readRemoteInstallId(ssh), '0f8a1c2b3d4e5f60718293a4b5c6d7e8')
  // Read-only: identity is the install's to mint, never a visiting client's.
  assert.equal(
    ssh.calls.some(cmd => /serve|dashboard|>\s*\S*install_id|tee|uuidgen/.test(cmd)),
    false
  )
})

test('readRemoteInstallId reports the INSTALL root id for a profile-pinned home', async () => {
  // The whole point of the id: two ssh connections to one machine — one pinned at a profile,
  // one at the root — must report the SAME backend so their roster rows collapse.
  const pinned = fakeSsh([
    [/HERMES_HOME/, '/Users/zillajr/.hermes/profiles/dixie\n'],
    [/cat .*install_id/, '0f8a1c2b3d4e5f60718293a4b5c6d7e8\n']
  ])

  assert.equal(await readRemoteInstallId(pinned), '0f8a1c2b3d4e5f60718293a4b5c6d7e8')
  assert.equal(
    pinned.calls.some(cmd => cmd.includes('/Users/zillajr/.hermes/install_id')),
    true
  )
  assert.equal(
    pinned.calls.some(cmd => cmd.includes('/profiles/dixie/install_id')),
    false
  )
})

test('readRemoteInstallId reports no id rather than a bad one', async () => {
  for (const payload of ['', 'not-an-id\n', 'ABCDEF\n', '0f8a1c2b3d4e5f60718293a4b5c6d7e8extra\n']) {
    const ssh = fakeSsh([
      [/HERMES_HOME/, '/Users/zillajr/.hermes\n'],
      [/cat .*install_id/, payload]
    ])

    // An older backend (or one that has never been started) simply has no identity: the roster
    // falls back to a per-connection key, exactly as before.
    assert.equal(await readRemoteInstallId(ssh), undefined, payload)
  }
})

test('listRemoteHermesProfiles inventories Mini-style profile dirs without spawning a dashboard', async () => {
  const ssh = fakeSsh([
    [/HERMES_HOME/, '/Users/zillajr/.hermes\n'],
    [/ls -1/, 'bob\ndixie\ngoose\nrambo\nbob.rollback-old\n']
  ])

  assert.deepEqual(await listRemoteHermesProfiles(ssh), ['default', 'bob', 'dixie', 'goose', 'rambo'])
  assert.equal(
    ssh.calls.some(cmd => cmd.includes('serve') || cmd.includes('dashboard')),
    false
  )
})

test('listRemoteHermesProfiles rejects a hostile HERMES_HOME', async () => {
  const ssh = fakeSsh([[/HERMES_HOME/, '/tmp/x; echo pwned\n']])

  await assert.rejects(
    () => listRemoteHermesProfiles(ssh),
    (err: any) => {
      assert.equal(err.kind, 'unsafe-path')

      return true
    }
  )
  assert.equal(
    ssh.calls.some(cmd => cmd.includes('ls -1')),
    false
  )
})

test('locateHermes prefers the explicit profile path when executable', async () => {
  const ssh = fakeSsh([[/\[ -x .*\/opt\/hermes/, 'OK']])
  assert.equal(await locateHermes(ssh, '/opt/hermes'), '/opt/hermes')
})

test('locateHermes throws (no silent fallback) when an EXPLICIT path is not executable', async () => {
  // command -v WOULD find a different install, but an explicit path must not
  // silently fall back to it — that is the "connected to the wrong hermes" bug.
  const ssh = fakeSsh([
    [/command -v hermes/, '/home/u/.local/bin/hermes\n'],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK']
  ])

  await assert.rejects(
    () => locateHermes(ssh, '/bad/path/hermes'),
    (err: any) => {
      assert.equal(err.kind, 'hermes-not-found')
      assert.match(err.message, /\/bad\/path\/hermes/)

      return true
    }
  )
})

test('locateHermes falls back to the login-shell command -v probe', async () => {
  const ssh = fakeSsh([
    [/command -v hermes/, '/home/u/.local/bin/hermes\n'],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK']
  ])

  assert.equal(await locateHermes(ssh, ''), '/home/u/.local/bin/hermes')
})

test('locateHermes preserves an installer wrapper instead of resolving its interpreter', async () => {
  // install.sh venv mode writes: exec "$HERMES_BIN" "$HERMES_ENTRYPOINT" "$@",
  // where $HERMES_BIN is the venv python. The old canonicalization returned
  // that interpreter, so `<python> --version` printed "Python x.y.z" and
  // `<python> serve --help` failed outright (#74411). The wrapper itself is
  // executable and forwards args correctly — return it untouched.
  const ssh = fakeSsh([
    [/command -v hermes/, '/home/u/.local/bin/hermes\n'],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK'],
    // If the removed python3 wrapper-parser were ever reintroduced, this rule
    // would reward it with an interpreter path and the assertions below fail.
    [/python3 -c/, '/home/u/.hermes/hermes-agent/venv/bin/python\n']
  ])

  assert.equal(await locateHermes(ssh, ''), '/home/u/.local/bin/hermes')
  assert.ok(
    !ssh.calls.some(cmd => cmd.includes('python3 -c')),
    'locateHermes must not shell out to a python3 parser to rewrite the launcher'
  )
})

test('locateHermes returns an explicit remoteHermesPath unchanged', async () => {
  // The override half of #74411: an explicit remoteHermesPath pointing at a
  // wrapper was also canonicalized to its interpreter, so overriding to
  // ~/.local/bin/hermes changed nothing for affected users.
  const ssh = fakeSsh([
    [/\[ -x .*\.local\/bin\/hermes/, 'OK'],
    [/python3 -c/, '/home/u/.hermes/hermes-agent/venv/bin/python\n']
  ])

  assert.equal(await locateHermes(ssh, '~/.local/bin/hermes'), '~/.local/bin/hermes')
  assert.ok(!ssh.calls.some(cmd => cmd.includes('python3 -c')), 'an explicit remoteHermesPath must never be rewritten')
})

test('locateHermes falls back to ~/.local/bin/hermes when the login-shell probe misses', async () => {
  // ~/.local/bin is the non-root installer's command location (scripts/install.sh).
  const ssh = fakeSsh([
    [/command -v hermes/, ''],
    [/\[ -x .*\.local\/bin\/hermes/, 'OK']
  ])

  assert.equal(await locateHermes(ssh, ''), '~/.local/bin/hermes')
})

test('locateHermes tries the conventional venv path last', async () => {
  const ssh = fakeSsh([[/\[ -x .*venv\/bin\/hermes/, 'OK']])
  assert.equal(await locateHermes(ssh, ''), '~/.hermes/hermes-agent/venv/bin/hermes')
})

test('locateHermes throws a hermes-not-found error with an install hint', async () => {
  const ssh = fakeSsh([]) // nothing is executable
  await assert.rejects(
    () => locateHermes(ssh, ''),
    (err: any) => {
      assert.equal(err.kind, 'hermes-not-found')
      assert.match(err.message, /install/i)

      return true
    }
  )
})

test('locateHermes uses a login shell for the command -v probe', async () => {
  const ssh = fakeSsh([
    [/command -v hermes/, '/x/hermes'],
    [/\[ -x/, 'OK']
  ])

  await locateHermes(ssh, '')
  assert.ok(
    ssh.calls.some(c => /bash -lc/.test(c)),
    'must probe in a login shell (PATH pitfall)'
  )
})

test('probeRemotePlatform accepts Linux and macOS', async () => {
  assert.deepEqual(await probeRemotePlatform(fakeSsh([[/uname/, 'Linux\nx86_64']])), {
    os: 'Linux',
    arch: 'x86_64'
  })
  assert.deepEqual(await probeRemotePlatform(fakeSsh([[/uname/, 'Darwin\narm64']])), {
    os: 'Darwin',
    arch: 'arm64'
  })
})

test('probeRemotePlatform rejects unsupported remote platforms', async () => {
  await assert.rejects(
    () => probeRemotePlatform(fakeSsh([[/uname/, 'MINGW64_NT\nx86_64']])),
    (err: any) => {
      assert.equal(err.kind, 'unsupported-platform')

      return true
    }
  )
})

test('ownership paths are isolated by ownership ID and spawn nonce', () => {
  assert.equal(ownershipDirectory(OWNERSHIP_ID), `~/.hermes/desktop-ssh/${OWNERSHIP_ID}`)
  assert.equal(lockfilePath(OWNERSHIP_ID), `~/.hermes/desktop-ssh/${OWNERSHIP_ID}/backend.lock.json`)
  assert.equal(spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), `~/.hermes/desktop-ssh/${OWNERSHIP_ID}/${SPAWN_NONCE}.log`)
})

test('readLockfile returns null ONLY for a missing/empty lockfile', async () => {
  assert.equal(await readLockfile(fakeSsh([[/cat/, '']]), OWNERSHIP_ID), null)
  const good = ownedLock({ pid: 1, port: 2 })
  assert.deepEqual(await readLockfile(fakeSsh([[/cat/, JSON.stringify(good)]]), OWNERSHIP_ID), good)
})

// #95532 fail-closed guard: a lockfile that EXISTS but doesn't match what this
// build writes is SKEW (foreign fork build, corruption, or a future schema) —
// it must be distinguishable from "no lockfile" so no reap/overwrite path can
// treat foreign live state as reapable.
test('readLockfile classifies existing-but-foreign lockfiles as skew, never null', async () => {
  // (a) foreign-schema lockfile (fork build wrote a different shape)
  const foreign = await readLockfile(fakeSsh([[/cat/, 'not json at all']]), OWNERSHIP_ID)
  assert.equal(isLockfileSkew(foreign), true)

  // (b) truncated lockfile (partial write / corruption)
  const truncated = await readLockfile(fakeSsh([[/cat/, JSON.stringify(ownedLock()).slice(0, 40)]]), OWNERSHIP_ID)

  assert.equal(isLockfileSkew(truncated), true)

  // (c) future schemaVersion (newer build owns this remote)
  const future = await readLockfile(
    fakeSsh([[/cat/, JSON.stringify(ownedLock({ schemaVersion: LOCKFILE_SCHEMA_VERSION + 1 }))]]),
    OWNERSHIP_ID
  )

  assert.equal(isLockfileSkew(future), true)
  // unknown schema number entirely
  const unknown = await readLockfile(fakeSsh([[/cat/, JSON.stringify({ schemaVersion: 999 })]]), OWNERSHIP_ID)
  assert.equal(isLockfileSkew(unknown), true)

  // missing ownershipId / foreign ownership
  const foreignOwner = await readLockfile(
    fakeSsh([[/cat/, JSON.stringify(ownedLock({ ownershipId: undefined }))]]),
    OWNERSHIP_ID
  )

  assert.equal(isLockfileSkew(foreignOwner), true)

  // every skew carries a diagnosable reason and is never a valid lock
  for (const skew of [foreign, truncated, future, unknown, foreignOwner]) {
    assert.equal(typeof (skew as any).reason, 'string')
    assert.notEqual(skew, null)
  }

  // a valid lock and a missing lockfile are NOT skew
  assert.equal(isLockfileSkew(await readLockfile(fakeSsh([[/cat/, '']]), OWNERSHIP_ID)), false)
  assert.equal(isLockfileSkew(await readLockfile(fakeSsh([[/cat/, JSON.stringify(ownedLock())]]), OWNERSHIP_ID)), false)
})

// #95532: on skew the reap pass must FAIL CLOSED — no kill, no lockfile
// removal/overwrite, no fresh spawn on top of foreign live state.
test('connect() fails closed on lockfile schema/ownership skew: skips reap, touches nothing', async () => {
  const skewShapes: Array<[string, string]> = [
    ['foreign-schema lockfile', '{"pid":333,"owner":"some-fork-desktop","version":"9.9.9"}'],
    ['truncated lockfile', JSON.stringify(ownedLock()).slice(0, 40)],
    ['future schemaVersion', JSON.stringify(ownedLock({ schemaVersion: LOCKFILE_SCHEMA_VERSION + 1 }))]
  ]

  for (const [label, raw] of skewShapes) {
    const ssh = fakeSsh([
      [/uname/, 'Linux\nx86_64'],
      [/\[ -x/, 'OK'],
      [/cat .*lock\.json/, raw],
      [/kill -0/, 'ALIVE'],
      [/print\("OWNED"/, 'OWNED\n']
    ])

    await assert.rejects(
      () => connect(connectDeps(ssh, { reuseToken: 'stored-token' })),
      (error: any) => error.kind === 'remote-lockfile-skew',
      `${label}: connect must refuse with remote-lockfile-skew`
    )
    assert.ok(
      // Any signal, a literal pid: the probe watchdog's `kill -9 $__htp`
      // targets its own child, not a lockfile pid.
      !ssh.calls.some(c => /(^|[^-\d])kill(?: -\w+)? \d/.test(c) && !/kill -0/.test(c)),
      `${label}: must not kill any pid`
    )
    assert.ok(!ssh.calls.some(c => /rm -f/.test(c)), `${label}: must not remove any remote file`)
    assert.ok(!ssh.calls.some(c => /setsid|nohup/.test(c)), `${label}: must not spawn on top of foreign state`)
    assert.ok(!ssh.calls.some(c => /printf '%s' '.*schemaVersion/.test(c)), `${label}: must not overwrite the lockfile`)
  }
})

test('disconnect() fails closed on lockfile skew: never reaps, never drops the foreign lockfile', async () => {
  const ssh = fakeSsh([
    [/cat .*lock\.json/, JSON.stringify(ownedLock({ schemaVersion: LOCKFILE_SCHEMA_VERSION + 1 }))],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  await disconnect(ssh, OWNERSHIP_ID)
  assert.ok(!ssh.calls.some(c => /(^|[^-\d])kill -?9? ?\d/.test(c) && !/kill -0/.test(c)), 'must not kill any pid')
  assert.ok(!ssh.calls.some(c => /rm -f/.test(c)), 'must not remove the foreign lockfile or logs')
})

test('cleanupStale is inert when handed a skew sentinel (defense in depth)', async () => {
  const ssh = fakeSsh([[/print\("OWNED"/, 'OWNED\n']])
  await cleanupStale(ssh, OWNERSHIP_ID, { skew: true, reason: 'schema-version' } as any)
  assert.equal(ssh.calls.length, 0, 'skew must short-circuit before any remote command')
})

test('writeLockfile mkdir -ps and stamps the schema version', async () => {
  const ssh = fakeSsh([])
  await writeLockfile(ssh, OWNERSHIP_ID, ownedLock({ pid: 7, port: 9 }))
  const cmd = ssh.calls.join('\n')
  assert.match(cmd, /mkdir -p/)
  assert.match(cmd, new RegExp(`"schemaVersion":${LOCKFILE_SCHEMA_VERSION}`))
})

test('remotePidAlive maps kill -0 ALIVE/DEAD', async () => {
  assert.equal(await remotePidAlive(fakeSsh([[/kill -0/, 'ALIVE']]), 123), true)
  assert.equal(await remotePidAlive(fakeSsh([[/kill -0/, 'DEAD']]), 123), false)
  assert.equal(await remotePidAlive(fakeSsh([]), null), false)
})

test('metadata and process proof transport failures remain indeterminate', async () => {
  const failure = new Error('connection reset')
  await assert.rejects(
    () => readLockfile(fakeSsh([[/cat/, failure]]), OWNERSHIP_ID),
    (error: any) => error.kind === 'transient-transport-error'
  )
  await assert.rejects(
    () => remotePidAlive(fakeSsh([[/kill -0/, failure]]), 123),
    (error: any) => error.kind === 'transient-transport-error'
  )
  await assert.rejects(
    () => pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, failure]]), 5, SPAWN_NONCE, '/x/hermes'),
    (error: any) => error.kind === 'transient-transport-error'
  )
})

test('pidIsOurDashboard requires the exact serve ownership nonce', async () => {
  const ours = `/x/hermes serve --isolated --ssh-owner-nonce ${SPAWN_NONCE}`
  assert.equal(await pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, 'OWNED\n']]), 5, SPAWN_NONCE, '/x/hermes'), true)
  assert.equal(
    await pidIsOurDashboard(
      fakeSsh([[/print\("OWNED"/, command => (command.includes('fedcba9876543210') ? 'FOREIGN\n' : 'OWNED\n')]]),
      5,
      'fedcba9876543210',
      '/x/hermes'
    ),
    false
  )
  assert.equal(await pidIsOurDashboard(fakeSsh([[/print\("OWNED"/, 'FOREIGN\n']]), 5, SPAWN_NONCE, '/x/hermes'), false)
})

test('pidIsOurDashboard accepts the venv entrypoint an installer wrapper execs into', async () => {
  let ownershipProbe = ''

  const ssh = fakeSsh([
    [
      /python3 -c/,
      (command: string) => {
        ownershipProbe = command

        return 'OWNED\n'
      }
    ]
  ])

  assert.equal(
    await pidIsOurDashboard(ssh, 5, SPAWN_NONCE, '~/.local/bin/hermes', '/Users/cd9c/.hermes', OWNERSHIP_ID, 'ops'),
    true
  )
  assert.match(ownershipProbe, /hermes-agent.*venv.*bin.*hermes/)
  assert.match(ownershipProbe, /desktop-ssh.*0123456789abcdef\.token/)
  assert.match(ownershipProbe, /expected_profile=.*ops/)
})

test.skipIf(process.platform === 'win32')(
  'pidIsOurDashboard recognizes an installer wrapper after it execs python + entrypoint',
  async () => {
    const temp = await mkdtemp(path.join(os.tmpdir(), 'hermes wrapper ownership '))
    const installDir = path.join(temp, 'install dir')
    const venvBin = path.join(installDir, 'venv', 'bin')
    const pythonLink = path.join(venvBin, 'python')
    const entrypoint = path.join(installDir, 'hermes')
    const launcher = path.join(temp, 'hermes launcher')
    const python = (await exec('command -v python3')).stdout.trim()
    const tokenPath = path.join(os.homedir(), spawnTokenPath(OWNERSHIP_ID, SPAWN_NONCE).replace(/^~\//, ''))

    await mkdir(venvBin, { recursive: true })
    await symlink(python, pythonLink)
    await writeFile(entrypoint, 'import time\ntime.sleep(30)\n', 'utf8')
    await writeFile(launcher, `#!/bin/bash\nexec "${pythonLink}" "${entrypoint}" "$@"\n`, 'utf8')
    await chmod(launcher, 0o755)

    const backendFlags = [
      '--host',
      '127.0.0.1',
      '--port',
      '0',
      '--ssh-session-token-file',
      tokenPath,
      '--ssh-owner-nonce',
      SPAWN_NONCE
    ]

    const children: ReturnType<typeof spawn>[] = []

    const spawnInstaller = (args: string[]) => {
      const process = spawn(launcher, args, { stdio: 'ignore' })

      children.push(process)

      return process
    }

    const child = spawnInstaller(['--profile', 'ops', 'serve', '--isolated', ...backendFlags])

    const ssh = {
      exec: async (command: string) => (await exec(command, { shell: '/bin/bash' })).stdout
    }

    const waitForEntrypoint = async (process: ReturnType<typeof spawn>) => {
      for (let attempt = 0; attempt < 40; attempt += 1) {
        const command = (await exec(`ps -ww -o command= -p ${process.pid}`)).stdout

        if (command.includes(entrypoint)) {
          return true
        }

        await new Promise(resolve => setTimeout(resolve, 25))
      }

      return false
    }

    try {
      assert.equal(await waitForEntrypoint(child), true, 'wrapper must exec into the fake installer entrypoint')
      assert.equal(
        await pidIsOurDashboard(ssh, child.pid, SPAWN_NONCE, launcher, '/unrelated/hermes-home', OWNERSHIP_ID, 'ops'),
        true
      )
      assert.equal(
        await pidIsOurDashboard(
          ssh,
          child.pid,
          SPAWN_NONCE,
          launcher,
          '/unrelated/hermes-home',
          'fedcba9876543210fedcba9876543210',
          'ops'
        ),
        false
      )
      assert.equal(
        await pidIsOurDashboard(
          ssh,
          child.pid,
          SPAWN_NONCE,
          launcher,
          '/unrelated/hermes-home',
          OWNERSHIP_ID,
          'wrong-profile'
        ),
        false
      )

      const misplacedIsolated = spawnInstaller(['--profile', 'ops', '--isolated', 'serve', ...backendFlags])

      assert.equal(await waitForEntrypoint(misplacedIsolated), true)
      assert.equal(
        await pidIsOurDashboard(
          ssh,
          misplacedIsolated.pid,
          SPAWN_NONCE,
          launcher,
          '/unrelated/hermes-home',
          OWNERSHIP_ID,
          'ops'
        ),
        false,
        '--isolated before serve must remain foreign'
      )

      const conflictingProfile = spawnInstaller([
        '--profile',
        'ops',
        'serve',
        '--isolated',
        ...backendFlags,
        '--profile',
        'foreign'
      ])

      assert.equal(await waitForEntrypoint(conflictingProfile), true)
      assert.equal(
        await pidIsOurDashboard(
          ssh,
          conflictingProfile.pid,
          SPAWN_NONCE,
          launcher,
          '/unrelated/hermes-home',
          OWNERSHIP_ID,
          'ops'
        ),
        false,
        'a duplicate conflicting profile must remain foreign'
      )
    } finally {
      for (const process of children) {
        process.kill('SIGTERM')
      }

      await rm(temp, { force: true, recursive: true })
    }
  }
)

test('disconnect reaps the backend recorded for this desktop ownership', async () => {
  const lock = ownedLock()

  const ssh = fakeSsh([
    [/cat .*backend\.lock\.json/, JSON.stringify(lock)],
    [/kill -0 333/, 'ALIVE\n'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  await disconnect(ssh, OWNERSHIP_ID)

  assert.ok(ssh.calls.some(command => /kill 333\b/.test(command)))
  assert.ok(ssh.calls.some(command => /rm -f .*backend\.lock\.json/.test(command)))
})

test('disconnect is a no-op when this desktop has no lockfile', async () => {
  const ssh = fakeSsh([[/cat .*backend\.lock\.json/, '']])

  await disconnect(ssh, OWNERSHIP_ID)

  assert.ok(!ssh.calls.some(command => /\bkill\b/.test(command)))
})

test('cleanupStale kills ONLY a provably-ours pid, always drops the lockfile', async () => {
  const notOurs = fakeSsh([[/print\("OWNED"/, 'FOREIGN\n']])
  await cleanupStale(notOurs, OWNERSHIP_ID, {
    pid: 5,
    spawnNonce: SPAWN_NONCE,
    hermesPath: '/x/hermes',
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
  })
  assert.ok(
    notOurs.calls.some(c => /print\("OWNED"/.test(c)),
    'must perform an ownership preflight'
  )
  assert.ok(notOurs.calls.some(c => /rm -f/.test(c)))

  const ours = fakeSsh([
    [/print\("OWNED"/, 'OWNED\n'],
    [cmd => /printf TERMINATED/.test(cmd), 'TERMINATED\n']
  ])

  await cleanupStale(ours, OWNERSHIP_ID, {
    pid: 9,
    spawnNonce: SPAWN_NONCE,
    hermesPath: '/x/hermes',
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
  })
  assert.ok(ours.calls.some(c => /kill 9\b/.test(c)))
  assert.ok(ours.calls.some(c => /rm -f/.test(c)))
})

test('buildSpawnCommand is headless serve, detached, token not in argv', () => {
  const cmd = buildSpawnCommand('/x/hermes', 'work', { logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE) })
  assert.match(cmd, /serve --isolated/)
  assert.match(cmd, /--host 127\.0\.0\.1 --port 0/)
  assert.doesNotMatch(cmd, /--skip-build|--no-open/)
  assert.doesNotMatch(cmd, /\bdashboard\b/)
  assert.match(cmd, /--profile/)
  assert.match(cmd, /work/)
  assert.match(cmd, /setsid/)
  assert.match(cmd, /<\/dev\/null/)
  assert.match(cmd, /echo \$!/)
  assert.ok(!cmd.includes('tok_secret_value'), 'token must not appear in spawn command')
  assert.ok(!cmd.includes('HERMES_DASHBOARD_SESSION_TOKEN'), 'token env var must not appear')
})

test.skipIf(process.platform === 'win32')('detached backend does not inherit the update mutex descriptor', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'hermes-update-mutex-'))
  const hermesPath = path.join(directory, 'hermes')
  const reportPath = path.join(directory, 'descriptor-report')
  const logPath = path.join(directory, 'spawn.log')

  try {
    await writeFile(
      hermesPath,
      `#!/bin/sh
: > ${reportPath}
for fd in /proc/$$/fd/*; do
  target=$(readlink "$fd" 2>/dev/null || true)
  case "$target" in
    *hermes-update-in-progress.mutex) printf '%s\\n' "$target" >> ${reportPath} ;;
  esac
done
`,
      { mode: 0o700 }
    )

    const command = buildSpawnCommand(hermesPath, '', {
      hermesHome: path.join(directory, 'home'),
      logPath
    })

    await exec(command, { shell: '/bin/bash' })

    for (let attempt = 0; attempt < 40; attempt += 1) {
      try {
        const report = await readFile(reportPath, 'utf8')
        assert.equal(report, '', 'the backend process must not retain the update mutex descriptor')

        return
      } catch (error: any) {
        if (error?.code !== 'ENOENT') {
          throw error
        }

        await new Promise(resolve => setTimeout(resolve, 25))
      }
    }

    assert.fail('the detached backend did not write its descriptor report')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('spawnRemoteDashboard returns exact ownership artifacts', async () => {
  const ssh = fakeSsh([
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid|nohup/, '4242\n']
  ])

  const { pid, spawnNonce, logPath } = await spawnRemoteDashboard(ssh, {
    hermesPath: '/x/hermes',
    profile: '',
    token: 'tk',
    ownershipId: OWNERSHIP_ID
  })

  assert.equal(pid, 4242)
  assert.match(spawnNonce, /^[0-9a-f]{16}$/)
  assert.equal(logPath, spawnLogPath(OWNERSHIP_ID, spawnNonce))
})

test('READY_RE accepts both serve and dashboard sentinels', () => {
  assert.equal(READY_RE.exec('HERMES_BACKEND_READY port=4321')?.[1], '4321')
  assert.equal(READY_RE.exec('HERMES_DASHBOARD_READY port=8765')?.[1], '8765')
  // The remote log is `>> log 2>&1`, so a stderr chunk without a newline can be spliced onto the sentinel.
  assert.equal(READY_RE.exec('INFO  Started server process [4711]HERMES_BACKEND_READY port=65238')?.[1], '65238')
})

test('spawnRemoteDashboard rejects when no pid is returned', async () => {
  const ssh = fakeSsh([
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid|nohup/, 'not-a-pid']
  ])

  await assert.rejects(
    () => spawnRemoteDashboard(ssh, { hermesPath: '/x/hermes', profile: '', token: 't', ownershipId: OWNERSHIP_ID }),
    (err: any) => {
      assert.equal(err.kind, 'spawn-failed')

      return true
    }
  )
})

test('scrapeReadyPort reads only the named spawn log', async () => {
  const logPath = spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE)
  const ssh = fakeSsh([[/cat/, 'some noise\nHERMES_DASHBOARD_READY port=51234\n']])
  const port = await scrapeReadyPort(ssh, logPath, { timeoutMs: 1000 })
  assert.equal(port, 51234)
  assert.ok(ssh.calls.every(call => !call.includes('desktop-ssh.log')))
})

test('scrapeReadyPort times out and reports a dead spawn', async () => {
  // never emits a READY line
  const ssh = fakeSsh([[/cat .*\.log/, 'still starting...']])
  await assert.rejects(
    () => scrapeReadyPort(ssh, spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), { timeoutMs: 60 }),
    (err: any) => {
      assert.equal(err.kind, 'ready-timeout')

      return true
    }
  )
  // dead process before announcement → spawn-failed
  await assert.rejects(
    () =>
      scrapeReadyPort(fakeSsh([[/cat/, '']]), spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE), {
        timeoutMs: 1000,
        isAlive: async () => false
      }),
    (err: any) => {
      assert.equal(err.kind, 'spawn-failed')

      return true
    }
  )
})

test('connect() spawns fresh when there is no lockfile, adopts the served token', async () => {
  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''], // no lockfile
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''], // token file write
    [/printf '%s\\n'/, ''],
    [/setsid/, '777\n'],
    [/kill -0 777/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=51999\n']
  ])

  const result = await connect(
    connectDeps(ssh, {
      adoptServedToken: async () => 'the-served-token',
      platform: { os: 'Linux', arch: 'x86_64' }
    })
  )

  assert.equal(ssh.calls.filter(command => command === 'uname -s; uname -m').length, 0)
  assert.equal(result.reused, false)
  assert.equal(result.remotePort, 51999)
  assert.equal(result.localPort, 50001)
  assert.equal(result.pid, 777)
  assert.equal(result.token, 'the-served-token')
  assert.equal(result.baseUrl, 'http://127.0.0.1:50001')
  assert.equal(result.tokenFingerprint, fingerprintToken('the-served-token'))
})

test('managed SSH maps a local scope to a different non-default remote profile', async () => {
  const localScope = 'work'

  const sshConfig = profileSshOverride(
    {
      profiles: {
        [localScope]: {
          mode: 'ssh',
          host: 'remote-box',
          remoteProfile: 'writer_2'
        }
      }
    },
    localScope
  )

  assert.equal(sshConfig?.remoteProfile, 'writer_2')

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid/, '778\n'],
    [/kill -0 778/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_BACKEND_READY port=52000\n']
  ])

  await connect(
    connectDeps(ssh, {
      profile: sshConfig?.remoteProfile,
      adoptServedToken: async () => 'mapped-profile-token'
    })
  )

  const spawn = ssh.calls.find(command => /setsid|nohup/.test(command)) || ''
  assert.match(spawn, /--profile\b/)
  assert.ok(spawn.includes('writer_2'))
  assert.match(spawn, /serve\s+--isolated/)
  assert.match(spawn, /\.hermes\/desktop-ssh\/[0-9a-f]{32}\/[0-9a-f]{16}\.token/)
  assert.ok(!spawn.includes(' work'), 'the local Desktop scope must not become the remote profile')
})

test('connect() reuses a healthy dashboard when fingerprint + probe pass', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  const result = await connect(connectDeps(ssh, { reuseToken, adoptServedToken: async (_b, t) => t }))
  assert.equal(result.reused, true)
  assert.equal(result.pid, 333)
  assert.equal(result.remotePort, 40000)
  // never spawned
  assert.ok(!ssh.calls.some(c => /setsid/.test(c)), 'reuse path must not spawn a new dashboard')
})

test('connect() respawns when the requested remote profile differs from the lockfile profile', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ profile: 'desktop-work', tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0 333/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n'],
    [cmd => /pidfd_open/.test(cmd), 'TERMINATED\n'],
    [/kill 333/, ''],
    [/--version/, 'Hermes Agent v0.18.2\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '890\n'],
    [/kill -0 890/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=52050\n']
  ])

  const result = await connect(
    connectDeps(ssh, { profile: 'default', reuseToken, adoptServedToken: async () => 'fresh' })
  )

  assert.equal(result.reused, false)
  assert.ok(
    ssh.calls.some(c => /setsid/.test(c)),
    'profile mismatch must spawn a fresh dashboard'
  )
})

test('connect() respawns when the lockfile hermesPath differs from the resolved path', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ hermesPath: '/old/stale/hermes', tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/--version/, 'Hermes Agent v0.18.2\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '890\n'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=52050\n']
  ])

  const result = await connect(
    connectDeps(ssh, { reuseToken, remoteHermesPath: '/new/hermes', adoptServedToken: async () => 'fresh' })
  )

  assert.equal(result.reused, false, 'must respawn, not reuse the old-path dashboard')
  assert.ok(
    ssh.calls.some(c => /setsid/.test(c)),
    'a fresh dashboard must be spawned'
  )
})

test('connect() respawns when the lockfile protocolVersion is incompatible', async () => {
  const reuseToken = 'stored-token'

  const lock = ownedLock({
    protocolVersion: PROTOCOL_VERSION + 99,
    tokenFingerprint: fingerprintToken(reuseToken)
  })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0 333/, 'ALIVE'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '901\n'],
    [/kill -0 901/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=44100\n']
  ])

  const result = await connect(connectDeps(ssh, { reuseToken, adoptServedToken: async () => 'fresh' }))
  assert.equal(result.reused, false, 'incompatible protocol must force a fresh spawn, not a reattach')
  assert.equal(result.pid, 901)
})

test('connect() fresh spawn writes hermesHome + protocolVersion into the lockfile', async () => {
  const writes: string[] = []

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, ''], // no lockfile
    [/HERMES_HOME/, '/home/alice/.hermes\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/printf '%s\\n'/, ''],
    [/setsid/, '700\n'],
    [/kill -0 700/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=45500\n'],
    [
      /printf '%s' '/,
      c => {
        writes.push(c)

        return ''
      }
    ]
  ])

  await connect(connectDeps(ssh, { adoptServedToken: async () => 'fresh' }))
  const lockWrite = writes.find(c => c.includes('schemaVersion')) || ''
  assert.match(lockWrite, new RegExp(`"protocolVersion":${PROTOCOL_VERSION}`))
  assert.match(lockWrite, /"hermesHome":"\/home\/alice\/\.hermes"/)
})

test('connect() respawns when the lockfile pid is dead (killed dashboard)', async () => {
  const lock = ownedLock({ tokenFingerprint: fingerprintToken('t') })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0 333/, 'DEAD'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '888\n'],
    [/kill -0 888/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=42000\n']
  ])

  const result = await connect(connectDeps(ssh, { reuseToken: 't', adoptServedToken: async () => 'fresh' }))
  assert.equal(result.reused, false)
  assert.equal(result.pid, 888)
  assert.equal(result.remotePort, 42000)
  assert.ok(
    !ssh.calls.some(command => command.includes('pid=333') && command.includes('print("OWNED"')),
    'a dead pid has no process identity to verify'
  )
})

test('managed update drain preserves a live foreign POSIX owner and its lock bytes', async () => {
  const lock = ownedLock()
  const rawLock = JSON.stringify(lock)

  const ssh = fakeSsh([
    [/cat .*lock\.json/, rawLock],
    [/kill -0 333/, 'ALIVE'],
    [/value="linux:"/, 'linux:123456\n'],
    [/print\("OWNED"/, 'FOREIGN\n']
  ])

  await assert.rejects(terminateOwnedDashboardForUpdate(ssh, lock), /ownership is unproven/)
  assert.equal(
    ssh.calls.some(command => /kill 333 &&/.test(command)),
    false,
    'foreign process is never signalled'
  )
  assert.equal(
    ssh.calls.some(command => /rm -f .*backend\.lock\.json/.test(command)),
    false,
    'foreign ownership bytes remain untouched'
  )
})

test('managed update drain rechecks the POSIX ownership record before signalling', async () => {
  const lock = ownedLock()
  const replacement = ownedLock({ pid: 334, spawnNonce: 'fedcba9876543210' })
  let reads = 0

  const ssh = fakeSsh([
    [
      /cat .*lock\.json/,
      () => {
        reads += 1

        return JSON.stringify(reads === 1 ? lock : replacement)
      }
    ],
    [/kill -0 333/, 'ALIVE'],
    [/value="linux:"/, 'linux:123456\n'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  await assert.rejects(terminateOwnedDashboardForUpdate(ssh, lock), /changed during process verification/)
  assert.equal(
    ssh.calls.some(command => /kill 333 &&/.test(command)),
    false
  )
})

test('managed update drain never falls back to a bare kill when the identity-bound signal is refused', async () => {
  const lock = ownedLock()
  const rawLock = JSON.stringify(lock)

  const ssh = fakeSsh([
    [/cat .*lock\.json/, rawLock],
    [/kill -0 333/, 'ALIVE'],
    [/value="linux:"/, 'linux:123456\n'],
    [/print\("OWNED"/, 'OWNED\n'],
    [/pidfd_open/, 'REFUSED\n']
  ])

  await assert.rejects(terminateOwnedDashboardForUpdate(ssh, lock), /identity changed at the signal boundary/)
  assert.equal(
    ssh.calls.some(command => /^kill 333\b/.test(command.trim())),
    false,
    'the final signal must stay inside the identity-checking helper'
  )
})

test('connect() respawns when the dashboard is wedged (alive pid, probe fails)', async () => {
  const reuseToken = 'stored'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'FOREIGN\n'],
    [/grep -q ssh-session-token-file/, 'YES\n'],
    [/python3 -c/, ''],
    [/setsid/, '999\n'],
    [/kill -0 999/, 'ALIVE'],
    [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=43000\n']
  ])

  const result = await connect(
    connectDeps(ssh, {
      reuseToken,
      probeReuseProof: async () => 'authenticated-stale',
      adoptServedToken: async () => 'fresh'
    })
  )

  assert.equal(result.reused, false)
  assert.equal(result.pid, 999)
  assert.equal(result.remotePort, 43000)
})

test('connect() aborts on an unsupported remote platform before doing anything else', async () => {
  const ssh = fakeSsh([[/uname/, 'SunOS\nsun4v']])
  await assert.rejects(
    () => connect(connectDeps(ssh)),
    (err: any) => {
      assert.equal(err.kind, 'unsupported-platform')

      return true
    }
  )
  assert.ok(!ssh.calls.some(c => /setsid/.test(c)))
})

test('openForward retries bind collisions only', async () => {
  const ports = [41001, 41002]
  const calls: number[] = []

  const localPort = await openForward(
    {
      pickLocalPort: async () => ports.shift(),
      forward: async port => {
        calls.push(port)

        if (calls.length === 1) {
          throw new Error('bind: Address already in use')
        }
      }
    },
    9119
  )

  assert.equal(localPort, 41002)
  assert.deepEqual(calls, [41001, 41002])
  assert.equal(isForwardBindCollision(new Error('Permission denied')), false)
})

test('connect() preserves an owned backend when a reuse transport throws', async () => {
  const reuseToken = 'stored-token'
  const lock = ownedLock({ tokenFingerprint: fingerprintToken(reuseToken) })

  const ssh = fakeSsh([
    [/uname/, 'Linux\nx86_64'],
    [/\[ -x/, 'OK'],
    [/cat .*lock\.json/, JSON.stringify(lock)],
    [/kill -0/, 'ALIVE'],
    [/print\("OWNED"/, 'OWNED\n']
  ])

  await assert.rejects(
    () =>
      connect(
        connectDeps(ssh, {
          reuseToken,
          forward: async () => {
            throw new Error('network reset')
          }
        })
      ),
    /network reset/
  )
  assert.ok(!ssh.calls.some(cmd => /kill 333\b/.test(cmd)))
})

test('validateRemotePath accepts absolute POSIX paths', () => {
  assert.doesNotThrow(() => validateRemotePath('/usr/bin/hermes'))
  assert.doesNotThrow(() => validateRemotePath('/home/user/.hermes/hermes-agent/venv/bin/hermes'))
})

test('validateRemotePath accepts ~/ prefix paths', () => {
  assert.doesNotThrow(() => validateRemotePath('~/bin/hermes'))
  assert.doesNotThrow(() => validateRemotePath('~/.hermes/logs/desktop-ssh.log'))
  assert.doesNotThrow(() => validateRemotePath('~'))
})

test('validateRemotePath accepts paths with spaces and quotes', () => {
  assert.doesNotThrow(() => validateRemotePath('/home/user/my project/hermes'))
  assert.doesNotThrow(() => validateRemotePath("~/path with 'quotes'/file"))
  assert.doesNotThrow(() => validateRemotePath('/path with "double quotes"/file'))
})

test('validateRemotePath rejects relative paths', () => {
  assert.throws(() => validateRemotePath('hermes'), /absolute|relative/i)
  assert.throws(() => validateRemotePath('./bin/hermes'), /absolute|relative/i)
  assert.throws(() => validateRemotePath('../etc/passwd'), /absolute|relative/i)
})

test('validateRemotePath rejects NUL and newline', () => {
  assert.throws(() => validateRemotePath('/usr/bin/hermes\x00'), /unsafe/i)
  assert.throws(() => validateRemotePath('/usr/bin/hermes\n'), /unsafe/i)
  assert.throws(() => validateRemotePath('/usr/bin/hermes\r'), /unsafe/i)
})

test('validateRemotePath preserves shell metacharacters as path data', () => {
  for (const p of ['/usr/$(whoami)/hermes', '/usr/`id`/hermes', '/usr/a;b|c&d<e>f']) {
    assert.doesNotThrow(() => validateRemotePath(p))
    assert.match(expandRemotePath(p), /^'/)
  }
})

test('expandRemotePath expands ~/ to "$HOME"/', () => {
  const result = expandRemotePath('~/.hermes/logs/desktop-ssh.log')
  assert.match(result, /\$HOME/)
  assert.ok(!result.includes('eval'), 'must not use eval')
  assert.ok(!result.includes('echo'), 'must not use echo for expansion')
})

test('expandRemotePath returns quoted absolute paths unchanged', () => {
  const result = expandRemotePath('/usr/local/bin/hermes')
  assert.ok(result.includes('/usr/local/bin/hermes'))
  assert.ok(!result.includes('eval'))
})

test('expandRemotePath preserves spaces as data', () => {
  const result = expandRemotePath('/home/user/my project/hermes')
  assert.ok(result.includes('my project'), 'spaces must be preserved, not split')
})
