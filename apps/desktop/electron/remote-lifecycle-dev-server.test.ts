import assert from 'node:assert/strict'
import { exec as execCallback, execFile as execFileCallback } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import {
  buildSpawnCommand,
  connect,
  expandRemotePath,
  fingerprintToken,
  LOCKFILE_SCHEMA_VERSION,
  PROTOCOL_VERSION,
  spawnLogPath
} from './remote-lifecycle'

const exec = promisify(execCallback)
const execFile = promisify(execFileCallback)

const OWNERSHIP_ID = '0123456789abcdef0123456789abcdef'
const SPAWN_NONCE = '0123456789abcdef'
const REUSE_TOKEN = 'stored-token'

test.skipIf(process.platform === 'win32')(
  'a dev build hands its renderer origin to the remote serve (real sh parse)',
  async (): Promise<void> => {
    // The backend trusts a dev renderer's http origin only when it inherits HERMES_DESKTOP_DEV_SERVER,
    // as a locally spawned backend does. A packaged build (no dev server) must not set it at all.
    const shell: string = (await exec('command -v bash', { shell: 'bash' })).stdout.trim()
    const directory: string = await mkdtemp(path.join(os.tmpdir(), 'hermes-dev-server-env-'))
    const hermesPath: string = path.join(directory, 'hermes')
    const reportPath: string = path.join(directory, 'dev-server-report')

    try {
      // The serve is detached: it reports through a FIFO, whose read blocks until it has written.
      await execFile('mkfifo', [reportPath])
      await writeFile(
        hermesPath,
        `#!${shell}\nprintf '%s' "\${HERMES_DESKTOP_DEV_SERVER-unset}" > ${expandRemotePath(reportPath)}\n`,
        { encoding: 'utf8', mode: 0o700 }
      )

      for (const [devServer, expected] of [
        ["http://127.0.0.1:5175/'$(x)", "http://127.0.0.1:5175/'$(x)"],
        ['', 'unset']
      ]) {
        const command: string = buildSpawnCommand(hermesPath, '', {
          devServer,
          hermesHome: path.join(directory, 'home'),
          logPath: path.join(directory, 'spawn.log')
        })

        // A runner launched from a dev shell may export the variable itself; the spawn must not inherit it.
        const env: NodeJS.ProcessEnv = { ...process.env, HOME: directory, HERMES_HOME: directory }
        delete env.HERMES_DESKTOP_DEV_SERVER
        await exec(command, { shell, env })

        assert.equal(await readFile(reportPath, 'utf8'), expected)
      }
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  }
)

// A live backend this desktop owns, recorded as spawned for `devServer` (absent = an older or packaged record).
function ownedLock(devServer?: string): Record<string, unknown> {
  return {
    schemaVersion: LOCKFILE_SCHEMA_VERSION,
    protocolVersion: PROTOCOL_VERSION,
    ownershipId: OWNERSHIP_ID,
    spawnNonce: SPAWN_NONCE,
    pid: 333,
    port: 40000,
    profile: '',
    hermesPath: '~/.local/bin/hermes',
    hermesHome: '~/.hermes',
    logPath: spawnLogPath(OWNERSHIP_ID, SPAWN_NONCE),
    tokenFingerprint: fingerprintToken(REUSE_TOKEN),
    startedAt: '2026-07-14T00:00:00.000Z',
    creationTime: 'linux:123456',
    ...(devServer === undefined ? {} : { devServer })
  }
}

// A remote whose owned backend passes every reuse check, and which can replace it with pid 890.
function ownedBackendSsh(lock: Record<string, unknown>): { calls: string[]; exec: (cmd: string) => Promise<string> } {
  const calls: string[] = []

  return {
    calls,
    async exec(cmd: string): Promise<string> {
      calls.push(cmd)
      // Spawn payloads run under the relaunch gate's marker mutex; answer them as the spawn, not as a probe.
      const spawning = /setsid|nohup/.test(cmd)

      if (spawning) {
        return '890\n'
      }

      if (cmd.includes('.hermes-update-in-progress')) {
        return 'CLEAR'
      }

      if (cmd.includes('printf TERMINATED') || (cmd.includes('os.kill(pid') && !cmd.includes('pidfd_open'))) {
        return 'TERMINATED'
      }

      const rules: [RegExp, string][] = [
        [/uname/, 'Linux\nx86_64'],
        [/\[ -x/, 'OK'],
        [/cat .*lock\.json/, JSON.stringify(lock)],
        [/kill -0/, 'ALIVE'],
        [/print\("OWNED"/, 'OWNED\n'],
        [/pidfd_open/, 'TERMINATED\n'],
        [/grep -q ssh-session-token-file/, 'YES\n'],
        [/cat .*\.log/, 'HERMES_DASHBOARD_READY port=52050\n']
      ]

      return rules.find(([matcher]) => matcher.test(cmd))?.[1] ?? ''
    }
  }
}

for (const [label, recorded, current, reused] of [
  ['the same dev renderer reuses its backend', 'http://127.0.0.1:5174', 'http://127.0.0.1:5174', true],
  ['a dev renderer on another port replaces the backend', 'http://127.0.0.1:5174', 'http://127.0.0.1:5175', false],
  ['a dev build replaces a backend a packaged build spawned', undefined, 'http://127.0.0.1:5175', false],
  ['a packaged build replaces a backend a dev build spawned', 'http://127.0.0.1:5174', '', false],
  ['a packaged build reuses a packaged backend', undefined, '', true]
] as const) {
  test(`SSH reuse: ${label}`, async (): Promise<void> => {
    // The backend trusts the renderer's http origin only from the HERMES_DESKTOP_DEV_SERVER it was
    // spawned with, so reusing one spawned for another renderer would refuse every renderer socket.
    const ssh = ownedBackendSsh(ownedLock(recorded))

    const result = await connect({
      ssh,
      ownershipId: OWNERSHIP_ID,
      profile: '',
      reuseToken: REUSE_TOKEN,
      devServer: current,
      platform: { os: 'Linux', arch: 'x86_64' },
      forward: async () => {},
      cancelForward: async () => {},
      pickLocalPort: async () => 50001,
      waitForHermes: async () => {},
      probeReuseProof: async () => 'authenticated-ok',
      adoptServedToken: async (_baseUrl: string, token: string) => token,
      readyTimeoutMs: 2000
    })

    const spawn = ssh.calls.find(cmd => /setsid|nohup/.test(cmd))

    assert.equal(result.reused, reused)

    if (reused) {
      assert.equal(result.pid, 333)
      assert.equal(spawn, undefined, 'a compatible backend must not be respawned')

      return
    }

    assert.equal(result.pid, 890)
    assert.ok(
      ssh.calls.some(cmd => /kill 333\b/.test(cmd)),
      'the owned backend must be reaped, not left running'
    )
    assert.ok(spawn?.includes(JSON.stringify({ devServer: current }).slice(1, -1)), 'the new record names its renderer')
  })
}
