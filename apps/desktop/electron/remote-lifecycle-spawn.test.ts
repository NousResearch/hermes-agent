import assert from 'node:assert/strict'
import { type ChildProcess, exec as execCallback, type ExecOptionsWithStringEncoding, spawn } from 'node:child_process'
import { mkdir, mkdtemp, readdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { expect, test } from 'vitest'

import {
  assertRemoteInstallUpdateClear,
  buildSpawnCommand,
  cleanupStale,
  connectReservationPath,
  expandRemotePath,
  fingerprintToken,
  LOCKFILE_SCHEMA_VERSION,
  lockfilePath,
  pidIsOurDashboard,
  PROTOCOL_VERSION,
  readLockfile,
  spawnLogPath,
  spawnTokenPath
} from './remote-lifecycle'
import { shellExec, shellSshDouble } from './remote-ssh-exec.test-helpers'
import type { SshConnection } from './ssh-connection'

const exec: (command: string, options?: ExecOptionsWithStringEncoding) => Promise<{ stdout: string; stderr: string }> =
  promisify(execCallback)

const ownershipId: string = '0123456789abcdef0123456789abcdef'
const spawnNonce: string = '0123456789abcdef'
interface SpawnVariant {
  launcher: 'setsid' | 'nohup'
  owned: boolean
}

const variants: SpawnVariant[] = [
  ...(process.platform === 'linux'
    ? [
        { launcher: 'setsid' as const, owned: false },
        { launcher: 'setsid' as const, owned: true }
      ]
    : []),
  { launcher: 'nohup', owned: false },
  { launcher: 'nohup', owned: true }
]

interface ChildReport {
  pid: number
  sessionId: number
  hupIgnored: boolean
  mutexFds: number[]
}

interface SpawnLock {
  schemaVersion: number
  ownershipId: string
  spawnNonce: string
  pid: number
  port: number
  profile: string
  hermesPath: string
  hermesHome: string
  logPath: string
  tokenFingerprint: string
  protocolVersion: number
  startedAt: string
}

interface SpawnFixture {
  root: string
  shell: string
  env: NodeJS.ProcessEnv
  command: string
  lock: SpawnLock
  marker: string
  run: (command: string) => Promise<{ stdout: string; stderr: string }>
  ssh: Pick<SshConnection, 'exec'>
  localPath: (remotePath: string) => string
  dispose: () => Promise<void>
}

async function spawnFixture(launcher: 'setsid' | 'nohup', owned: boolean): Promise<SpawnFixture> {
  const root: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-spawn-'))
  const bin: string = path.join(root, 'bin')
  const shell: string = (await exec('command -v sh', { shell: 'sh' })).stdout.trim()
  const localPath = (remotePath: string): string => remotePath.replace(/^~/, root)
  const env: NodeJS.ProcessEnv = { HOME: root, HERMES_HOME: path.join(root, '.hermes'), PATH: bin, LANG: 'C.UTF-8' }

  const run = async (command: string): Promise<{ stdout: string; stderr: string }> => shellExec(command, { shell, env })

  const ssh: Pick<SshConnection, 'exec'> = shellSshDouble({ shell, env })

  const hermesPath: string = path.join(root, 'fake hermes')
  const hermesHome: string = path.join(root, '.hermes')
  const marker: string = path.join(hermesHome, '.hermes-update-in-progress')

  const lock: SpawnLock = {
    schemaVersion: LOCKFILE_SCHEMA_VERSION,
    ownershipId,
    spawnNonce,
    pid: 0,
    port: 0,
    profile: 'ops__PID__',
    hermesPath,
    hermesHome,
    logPath: spawnLogPath(ownershipId, spawnNonce),
    tokenFingerprint: fingerprintToken('fixture-token'),
    protocolVersion: PROTOCOL_VERSION,
    startedAt: new Date().toISOString()
  }

  await mkdir(bin)
  await mkdir(hermesHome)
  await mkdir(path.join(root, 'reports'))

  // Restrict capability discovery, not the reported host OS. The fallback runs
  // real nohup with no setsid executable anywhere on the child's PATH.
  for (const tool of [
    'sh',
    'python3',
    'mkdir',
    'dirname',
    'rm',
    'cat',
    'sed',
    'head',
    'sleep',
    'mv',
    'ln',
    'env',
    'ps',
    launcher
  ]) {
    const executable: string = (await exec(`command -v ${tool}`, { shell })).stdout.trim()
    await symlink(executable, path.join(bin, tool))
  }

  await writeFile(
    hermesPath,
    `#!${path.join(bin, 'python3')}
import json,os,signal,time
from pathlib import Path
pid=os.getpid()
mutex=Path(os.environ['HERMES_HOME'])/'.hermes-update-in-progress.lock'
identity=mutex.stat()
fds=[]
for fd in range(3,256):
 try:
  candidate=os.fstat(fd)
  if (candidate.st_dev,candidate.st_ino)==(identity.st_dev,identity.st_ino):fds.append(fd)
 except OSError:pass
report={'pid':pid,'sessionId':os.getsid(0),'hupIgnored':signal.getsignal(signal.SIGHUP)==signal.SIG_IGN,'mutexFds':fds}
p=Path(os.environ['HOME'])/'reports'/str(pid)
p.with_suffix('.tmp').write_text(json.dumps(report),encoding='utf-8')
p.with_suffix('.tmp').replace(p)
time.sleep(30)
`,
    { encoding: 'utf8', mode: 0o700 }
  )

  const command: string = buildSpawnCommand(hermesPath, lock.profile, {
    hermesHome,
    logPath: lock.logPath,
    spawnNonce,
    tokenFilePath: spawnTokenPath(ownershipId, spawnNonce),
    ownershipId: owned ? ownershipId : undefined,
    reservationNonce: spawnNonce,
    lockMetadata: owned ? lock : undefined
  })

  const dispose = async (): Promise<void> => {
    for (const name of await readdir(path.join(root, 'reports'))) {
      if (!/^[1-9][0-9]*$/.test(name)) {
        continue
      }

      const report: ChildReport = JSON.parse(await readFile(path.join(root, 'reports', name), 'utf8'))

      try {
        if (await pidIsOurDashboard(ssh, report.pid, spawnNonce, hermesPath, hermesHome, ownershipId, lock.profile)) {
          process.kill(report.pid, 'SIGKILL')
        }
      } catch (error) {
        if (!(error instanceof Error && 'code' in error && error.code === 'ESRCH')) {
          throw error
        }
      }
    }

    await rm(root, { recursive: true, force: true })
  }

  return { root, shell, env, command, lock, marker, run, ssh, localPath, dispose }
}

async function reports(fixture: SpawnFixture): Promise<ChildReport[]> {
  const names: string[] = (await readdir(path.join(fixture.root, 'reports'))).filter((name: string): boolean =>
    /^[1-9][0-9]*$/.test(name)
  )

  return Promise.all(
    names.map(async (name: string): Promise<ChildReport> =>
      JSON.parse(await readFile(path.join(fixture.root, 'reports', name), 'utf8'))
    )
  )
}

test.skipIf(process.platform === 'win32').each(variants)(
  '$launcher spawn (ownership=$owned) returns only the detached child PID and preserves ownership',
  async ({ launcher, owned }: SpawnVariant): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(launcher, owned)

    const foreign: ChildProcess = spawn('python3', ['-c', 'import time;time.sleep(30)'], {
      env: fixture.env,
      stdio: 'ignore'
    })

    assert.ok(foreign.pid)

    try {
      const parentSession: number = Number(
        (await fixture.run('python3 -c "import os;print(os.getsid(0))"')).stdout.trim()
      )

      const results: { stdout: string; stderr: string }[] = await Promise.all(
        Array.from({ length: owned ? 2 : 1 }, (): Promise<{ stdout: string; stderr: string }> =>
          fixture.run(fixture.command)
        )
      )

      const result: { stdout: string; stderr: string } | undefined = results.find(
        (output: { stdout: string }): boolean => output.stdout !== 'EXISTING'
      )

      assert.ok(result)

      if (owned) {
        assert.equal(results.filter((output: { stdout: string }): boolean => output.stdout === 'EXISTING').length, 1)
      }

      await expect.poll(async (): Promise<number> => (await reports(fixture)).length, { timeout: 5_000 }).toBe(1)
      const [report]: ChildReport[] = await reports(fixture)
      assert.match(result.stdout, /^[1-9][0-9]*\n$/)
      assert.equal(result.stdout.trim(), String(report.pid))
      assert.equal(result.stderr, '')
      assert.deepEqual(report.mutexFds, [])
      assert.ok(process.kill(report.pid, 0), 'the reported child survives the spawning shell exiting')

      if (launcher === 'setsid') {
        assert.notEqual(report.sessionId, parentSession)
      } else {
        assert.equal(report.sessionId, parentSession)
        assert.equal(report.hupIgnored, true)
        process.kill(report.pid, 'SIGHUP')
      }

      // The child remains alive while another process acquires the exact mutex.
      await fixture.run(
        `python3 -c 'import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)' ${expandRemotePath(`${fixture.marker}.lock`)}`
      )
      // The published ownership record carries the profile the spawn argv
      // actually pinned — normalized like the CLI would (#88842) — never the
      // raw roster/bridge label the fixture passed in. Derive the working lock
      // from the published record exactly the way a reconnecting client does.
      const lock: SpawnLock = { ...fixture.lock, pid: report.pid, profile: 'ops__pid__' }

      if (owned) {
        assert.deepEqual(JSON.parse(await readFile(fixture.localPath(lockfilePath(ownershipId)), 'utf8')), lock)
        assert.deepEqual(await readLockfile(fixture.ssh, ownershipId), lock)
        assert.equal((await fixture.run(fixture.command)).stdout, 'EXISTING')
        assert.equal((await reports(fixture)).length, 1)
        assert.ok(
          !(await readdir(path.dirname(fixture.localPath(lockfilePath(ownershipId))))).some((name: string): boolean =>
            name.endsWith('.tmp')
          )
        )
      }

      await assert.rejects(readFile(fixture.localPath(connectReservationPath(ownershipId))), { code: 'ENOENT' })
      await cleanupStale(fixture.ssh, ownershipId, { ...lock, pid: foreign.pid, logPath: '' })
      assert.ok(process.kill(foreign.pid, 0), 'a foreign process is never killed by cleanup')
      await cleanupStale(fixture.ssh, ownershipId, lock)
      assert.ok(process.kill(foreign.pid, 0), 'owned cleanup leaves the unrelated child alive')
      assert.throws((): boolean => process.kill(report.pid, 0), { code: 'ESRCH' })
      await assert.rejects(readFile(fixture.localPath(lockfilePath(ownershipId))), { code: 'ENOENT' })
    } finally {
      foreign.kill('SIGKILL')
      await fixture.dispose()
    }
  },
  20_000
)

test.skipIf(process.platform === 'win32').each(variants)(
  '$launcher spawn (ownership=$owned) refuses live or malformed update markers without reserving an owner',
  async ({ launcher, owned }: SpawnVariant): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(launcher, owned)

    try {
      for (const marker of [`${process.pid}\n${Math.floor(Date.now() / 1000)}\n`, 'not-a-pid\n']) {
        await writeFile(fixture.marker, marker, 'utf8')
        await assert.rejects(fixture.run(fixture.command), { code: 75, stdout: '' })
        assert.deepEqual(await reports(fixture), [])
        await assert.rejects(readFile(fixture.localPath(connectReservationPath(ownershipId))), { code: 'ENOENT' })
        await assert.rejects(readFile(fixture.localPath(lockfilePath(ownershipId))), { code: 'ENOENT' })
      }
    } finally {
      await fixture.dispose()
    }
  },
  20_000
)

/** A python3 child that prints one line (after its setup) and stays alive. */
async function pythonChild(fixture: SpawnFixture, script: string, ...args: string[]): Promise<[ChildProcess, string]> {
  const child: ChildProcess = spawn('python3', ['-c', script, ...args], {
    env: fixture.env,
    stdio: ['ignore', 'pipe', 'inherit']
  })

  const line: string = await new Promise((resolve, reject) => {
    child.stdout!.once('data', (chunk: Buffer) => resolve(chunk.toString().trim()))
    child.once('exit', code => reject(new Error(`python child exited ${code}`)))
  })

  return [child, line]
}

async function deadPid(): Promise<number> {
  const child: ChildProcess = spawn('true')
  await new Promise(resolve => child.once('exit', resolve))

  return child.pid!
}

test.skipIf(process.platform === 'win32')(
  'v2 claims are judged by pid + creation time: a dead claim is cleared, a live delegate refuses',
  async (): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(variants[0].launcher, false)
    const [delegate, ct] = await pythonChild(fixture, 'import time;print(time.time(),flush=True);time.sleep(30)')
    const dead: number = await deadPid()
    const now: number = Math.floor(Date.now() / 1000)

    try {
      await writeFile(fixture.marker, `${dead}\n${now}\nct:${ct}\ndelegate:${dead} ct:${ct}\nrun:desk-1\n`)
      await assertRemoteInstallUpdateClear(fixture.ssh, fixture.lock.hermesHome)
      await assert.rejects(readFile(fixture.marker), { code: 'ENOENT' })

      const live: string = `${dead}\n${now}\nct:${ct}\ndelegate:${delegate.pid} ct:${ct}\n`
      await writeFile(fixture.marker, live)
      await assert.rejects(
        assertRemoteInstallUpdateClear(fixture.ssh, fixture.lock.hermesHome),
        (error: any) => error.kind === 'update-in-progress' && error.message.includes(`process ${delegate.pid}`)
      )
      await assert.rejects(fixture.run(fixture.command), { code: 75, stdout: '' })
      assert.equal(await readFile(fixture.marker, 'utf8'), live)

      // The live pid at another incarnation (creation time 60 s off) is a dead claim.
      await writeFile(fixture.marker, `${delegate.pid}\n${now}\nct:${Number(ct) - 60}\n`)
      assert.match((await fixture.run(fixture.command)).stdout, /^[1-9][0-9]*\n$/)
      await assert.rejects(readFile(fixture.marker), { code: 'ENOENT' })
    } finally {
      delegate.kill('SIGKILL')
      await fixture.dispose()
    }
  },
  20_000
)

test.skipIf(process.platform === 'win32')(
  "spawn waits on the updaters' <marker>.lock, so a claim made under it is never raced",
  async (): Promise<void> => {
    const fixture: SpawnFixture = await spawnFixture(variants[0].launcher, false)

    // An updater: hold <marker>.lock (update_lock.py / marker.sh flock), then
    // write its v2 claim and release, staying alive as the live owner.
    const [updater] = await pythonChild(
      fixture,
      `import fcntl,os,sys,time
t=time.time()
fd=os.open(sys.argv[1]+'.lock',os.O_RDWR|os.O_CREAT)
fcntl.flock(fd,fcntl.LOCK_EX)
print('held',flush=True)
time.sleep(0.5)
open(sys.argv[1],'w').write('%d\\n%d\\nct:%.3f\\n'%(os.getpid(),int(t),t))
time.sleep(0.5)
fcntl.flock(fd,fcntl.LOCK_UN)
time.sleep(30)`,
      fixture.marker
    )

    try {
      await assert.rejects(fixture.run(fixture.command), { code: 75, stdout: '' })
      assert.deepEqual(await reports(fixture), [])
      assert.match(await readFile(fixture.marker, 'utf8'), new RegExp(`^${updater.pid}\\n`))
    } finally {
      updater.kill('SIGKILL')
      await fixture.dispose()
    }
  },
  20_000
)

interface ReservationRace {
  root: string
  home: string
  /** The reservation block `buildSpawnCommand` composes, byte for byte. */
  payload: string
  reservation: string
  owner: string
}

/**
 * Capture the reservation payload the app really composes: the argv the remote
 * shell hands to python3 is recorded and the payload is argv[3] of that call.
 * Every race below runs those bytes verbatim instead of a copy of them.
 */
async function reservationRace(): Promise<ReservationRace> {
  const root: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-reservation-race-'))
  const home: string = path.join(root, 'home')
  const argvShim: string = path.join(root, 'argv-shim')
  const argvFile: string = path.join(root, 'argv.bin')

  await mkdir(home)
  await mkdir(argvShim)

  await writeFile(path.join(argvShim, 'python3'), `#!/bin/sh\nprintf '%s\\0' "$@" > '${argvFile}'\n`, {
    mode: 0o755
  })

  const command: string = buildSpawnCommand('/x/hermes', 'work', {
    hermesHome: '~/.hermes',
    lockMetadata: { ownershipId, spawnNonce },
    logPath: spawnLogPath(ownershipId, spawnNonce),
    ownershipId,
    reservationNonce: spawnNonce,
    spawnNonce,
    tokenFilePath: spawnTokenPath(ownershipId, spawnNonce)
  })

  await exec(command, { env: { ...process.env, HOME: home, PATH: `${argvShim}:${process.env.PATH}` }, shell: 'sh' })

  const payload: string = (await readFile(argvFile, 'utf8')).split('\0')[3]
  const reservation: string = home + connectReservationPath(ownershipId).replace(/^~/, '')

  return { root, home, payload, reservation, owner: path.join(reservation, 'owner') }
}

/** The reservation block alone: everything the payload does before it spawns. */
function reservationBlock(payload: string): string {
  const reserve: number = payload.indexOf('i=0;')
  const spawned: number = payload.indexOf('mkdir -p "$(dirname', reserve)

  assert.ok(reserve > 0 && spawned > reserve, 'the reservation block moved; teach this test its new shape')

  return payload.slice(0, spawned)
}

/**
 * A caller that runs the shipped block, then behaves like the holder: it proves
 * the record it took is its own, holds long enough for a second holder to be
 * observable, and re-checks the record before it releases. `notmine`/`lost` are
 * a live holder whose record was taken away — the defect under test.
 */
async function holderRacer(root: string, payload: string): Promise<string> {
  const race: string = path.join(root, 'race.sh')

  await writeFile(
    race,
    `${reservationBlock(payload)}` +
      `printf '%s\\n' "$$" >> "$MARK_DIR/entries"; ` +
      `case "$(cat "$owner_file" 2>/dev/null || true)" in "$$:"*) ;; *) : > "$MARK_DIR/notmine";; esac; ` +
      `if [ -e "$MARK_DIR/held" ]; then : > "$MARK_DIR/double"; fi; ` +
      `: > "$MARK_DIR/held"; sleep 0.15; ` +
      `case "$(cat "$owner_file" 2>/dev/null || true)" in "$$:"*) ;; *) : > "$MARK_DIR/lost";; esac; ` +
      `rm -f "$MARK_DIR/held"; `,
    { mode: 0o755 }
  )

  return race
}

/** Run one round of `callers` real callers and fail on any overlap/steal marker. */
async function raceRound(race: string, home: string, mark: string, callers: number, label: string): Promise<string[]> {
  await rm(mark, { force: true, recursive: true })
  await mkdir(mark)

  await Promise.all(
    Array.from({ length: callers }, (): Promise<{ stdout: string }> =>
      exec(race, { env: { ...process.env, HOME: home, MARK_DIR: mark }, shell: 'sh' })
    )
  )

  for (const name of ['double', 'notmine', 'lost']) {
    await assert.rejects(
      readFile(path.join(mark, name)),
      { code: 'ENOENT' },
      `${label}: ${
        name === 'double'
          ? 'two callers held the ownership reservation at once'
          : name === 'notmine'
            ? 'a caller that believed it held the reservation found a foreign owner record'
            : 'a live holder lost its owner record while it was holding'
      }`
    )
  }

  return (await readFile(path.join(mark, 'entries'), 'utf8')).trim().split('\n')
}

// The reservation must be decided by an atomic create, never by the exit status
// of `mkdir`. Ubuntu 26.04 ships uutils coreutils 0.10.0, whose `mkdir` can
// report success twice for the same path under concurrency; with mkdir as the
// mutex the loser never took the "already reserved" path, two backends spawned
// for one ownership id, and a reaper could delete a reservation whose owner
// record was not written yet. A `mkdir` that claims a create it lost must not
// be able to make two holders exist.
test.skipIf(process.platform === 'win32')(
  'the ownership reservation admits one holder when a mkdir reports a create it lost',
  async (): Promise<void> => {
    const { root, home, payload }: ReservationRace = await reservationRace()

    try {
      const round: string = path.join(root, 'round')
      const raceShim: string = path.join(root, 'race-shim')
      const race: string = path.join(root, 'race.sh')

      await mkdir(raceShim)

      await writeFile(
        path.join(raceShim, 'mkdir'),
        '#!/bin/sh\n' +
          // `-p` is idempotent container creation, not an ownership claim; every
          // bare `mkdir <path>` reports success whether or not it created it.
          'for a in "$@"; do case "$a" in -p|--parents) command -p mkdir "$@" ;; esac; done\n' +
          'command -p mkdir "$@" 2>/dev/null\n' +
          'exit 0\n',
        { mode: 0o755 }
      )

      // Racer protocol: whoever holds the reservation sleeps; a second holder
      // inside that window is the double-win the mkdir mutex allowed, and the
      // one the reaper's reservation delete raced against.
      await writeFile(
        race,
        `${reservationBlock(payload)}` +
          `if [ -e "$MARK_DIR/held" ]; then : > "$MARK_DIR/double"; fi; ` +
          `: > "$MARK_DIR/held"; sleep 0.3; rm -f "$MARK_DIR/held"; `,
        { mode: 0o755 }
      )

      const env: NodeJS.ProcessEnv = { ...process.env, HOME: home, PATH: `${raceShim}:${process.env.PATH}` }

      for (let index = 0; index < 8; index++) {
        await rm(round, { force: true, recursive: true })
        await mkdir(round)

        await Promise.all(
          [1, 2].map((): Promise<{ stdout: string }> => exec(race, { env: { ...env, MARK_DIR: round }, shell: 'sh' }))
        )

        await assert.rejects(
          readFile(path.join(round, 'double')),
          { code: 'ENOENT' },
          `round ${index + 1}: two spawns held the ownership reservation at once`
        )
      }
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
  60_000
)

// Reclaim must have the atomicity of acquire. A dead owner record used to be
// removed with `rm -rf "$reservation"` by every waiter that judged the same pid
// dead; a delete could land after another caller created its record, so a live
// holder lost it and a second backend spawned. Measured on Ubuntu 26.04 with
// uutils 0.10.0 at K=8: 14 of 20 rounds admitted two holders and 34-35 of 50
// rounds took a live holder's record away.
test.skipIf(process.platform === 'win32')(
  'reclaiming a dead owner record admits one holder and never takes a live record away',
  async (): Promise<void> => {
    const { root, home, payload, reservation, owner }: ReservationRace = await reservationRace()
    const race: string = await holderRacer(root, payload)

    try {
      for (let index = 0; index < 6; index++) {
        const dead: number = await deadPid()

        assert.throws(
          (): boolean => process.kill(dead, 0),
          { code: 'ESRCH' },
          `round ${index + 1}: the seeded owner pid ${dead} is alive`
        )

        await mkdir(reservation, { recursive: true })
        await writeFile(owner, `${dead}:${spawnNonce}`)

        const entries: string[] = await raceRound(race, home, path.join(root, `dead-${index}`), 8, `round ${index + 1}`)

        assert.equal(entries.length, 8, `round ${index + 1}: every caller should have taken the reservation in turn`)
        assert.equal(new Set(entries).size, 8, `round ${index + 1}: a caller took the reservation twice`)
      }
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
  120_000
)

// The takeover is the crash window the reclaim now owns: a caller that dies
// between "record moved" and "record judged" must leave the record moved — never
// unlinked — and must not wedge the next callers.
test.skipIf(process.platform === 'win32')(
  'a reclaim killed mid-takeover leaves the record moved, not unlinked, and the next callers admit one holder',
  async (): Promise<void> => {
    const { root, home, payload, reservation, owner }: ReservationRace = await reservationRace()
    const shim: string = path.join(root, 'kill-shim')
    const race: string = path.join(root, 'race.sh')
    const dead: number = await deadPid()

    await mkdir(shim)
    // The takeover is the first `mv` the block runs: do the real rename and then
    // kill the caller, which is exactly the crash between the two halves.
    await writeFile(path.join(shim, 'mv'), '#!/bin/sh\ncommand -p mv "$@"\nkill -9 "$PPID"\n', { mode: 0o755 })
    await writeFile(race, reservationBlock(payload), { mode: 0o755 })

    try {
      await mkdir(reservation, { recursive: true })
      await writeFile(owner, `${dead}:${spawnNonce}`)

      const crash: { code: number | null; signal: string | null } = await new Promise(resolve => {
        const child: ChildProcess = spawn('sh', [race], {
          env: { ...process.env, HOME: home, PATH: `${shim}:${process.env.PATH}` },
          stdio: 'ignore'
        })

        child.once('close', (code, signal) => resolve({ code, signal }))
      })

      assert.equal(crash.signal, 'SIGKILL', 'the racer should be dead after the takeover moved the record')

      await assert.rejects(
        readFile(owner),
        { code: 'ENOENT' },
        'the dead record must have been moved off the shared path'
      )

      const owning: string = path.dirname(reservation)

      const moved: string[] = (await readdir(owning)).filter((name: string): boolean =>
        name.startsWith('.connect.lock.reclaim.')
      )

      assert.equal(moved.length, 1, 'the takeover must leave exactly one moved copy of the record')
      assert.equal(
        await readFile(path.join(owning, moved[0]), 'utf8'),
        `${dead}:${spawnNonce}`,
        'the moved copy must still be the record the takeover judged'
      )

      const next: string = await holderRacer(root, payload)
      const entries: string[] = await raceRound(next, home, path.join(root, 'next'), 4, 'after the crash')

      assert.equal(entries.length, 4, 'the crashed takeover must not wedge the next callers')
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
  120_000
)
