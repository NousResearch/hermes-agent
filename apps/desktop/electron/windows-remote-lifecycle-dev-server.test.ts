import assert from 'node:assert/strict'
import crypto from 'node:crypto'

import { test } from 'vitest'

import { connectWindowsRemote, decodePowerShellCommand } from './windows-remote-lifecycle'

const OWNERSHIP_ID = '0123456789abcdef0123456789abcdef'
const SPAWN_NONCE = '0123456789abcdef'
const REUSE_TOKEN = 'stored-token'
const HERMES_PATH = 'C:\\h\\hermes.exe'
const HERMES_HOME = 'C:\\h'

function fingerprint(token: string): string {
  return crypto.createHash('sha256').update(token).digest('hex').slice(0, 32)
}

// A live backend this desktop owns, recorded as spawned for `devServer` (absent = an older or packaged record).
function ownedLock(devServer?: string): Record<string, unknown> {
  return {
    schemaVersion: 2,
    protocolVersion: 1,
    ownershipId: OWNERSHIP_ID,
    spawnNonce: SPAWN_NONCE,
    pid: 333,
    creationTimeNs: '1784219690452757504',
    port: 40000,
    profile: '',
    hermesPath: HERMES_PATH,
    hermesHome: HERMES_HOME,
    tokenFingerprint: fingerprint(REUSE_TOKEN),
    startedAt: '2026-07-14T00:00:00.000Z',
    ...(devServer === undefined ? {} : { devServer })
  }
}

interface Exec {
  script: string
  stdinData: string
}

// A Windows remote whose owned backend (`lock`, or none) passes every ownership check, and which
// can replace it with pid 890. Every exec is a PowerShell -EncodedCommand; answer by its operation.
function ownedWindowsSsh(lock: null | Record<string, unknown>) {
  const execs: Exec[] = []

  return {
    execs,
    async exec(command: string, options: { stdinData?: string } = {}): Promise<string> {
      const script = decodePowerShellCommand(command)
      execs.push({ script, stdinData: String(options.stdinData ?? '') })
      const op = (name: string) => script.includes(`'hermes_cli.windows_ssh_runtime' '${name}'`)

      if (op('spawn')) {
        return JSON.stringify({ pid: 890, creationTimeNs: '1784219690452757600' })
      }

      if (script.includes('Get-Command hermes')) {
        return JSON.stringify({ os: 'Windows', arch: 'AMD64', hermesHome: HERMES_HOME, hermesPath: HERMES_PATH, python: 'C:\\h\\python.exe' })
      }

      if (script.includes('.hermes-update-in-progress')) {
        return 'CLEAR'
      }

      if (op('inspect')) {
        return JSON.stringify({ path: HERMES_PATH, version: 'Hermes Agent v1', runtimeBootstrapOwnership: 1, supported: true })
      }

      if (op('read-lock')) {
        return JSON.stringify(lock)
      }

      if (op('process-state')) {
        return JSON.stringify({ alive: true, owned: true, indeterminate: false })
      }

      if (op('read-log')) {
        return JSON.stringify({ content: 'HERMES_DASHBOARD_READY port=52050\n' })
      }

      return JSON.stringify({ ok: true })
    }
  }
}

function connectAs(ssh: ReturnType<typeof ownedWindowsSsh>, devServer: string) {
  return connectWindowsRemote({
    ssh,
    ownershipId: OWNERSHIP_ID,
    profile: '',
    reuseToken: REUSE_TOKEN,
    devServer,
    pickLocalPort: async () => 50001,
    forward: async () => {},
    cancelForward: async () => {},
    waitForHermes: async () => {},
    probeReuseProof: async () => 'authenticated-ok',
    readyTimeoutMs: 2000
  })
}

const spawnOf = (ssh: ReturnType<typeof ownedWindowsSsh>) =>
  ssh.execs.find(({ script }) => script.includes("'hermes_cli.windows_ssh_runtime' 'spawn'"))?.script

const writtenLocks = (ssh: ReturnType<typeof ownedWindowsSsh>) =>
  ssh.execs
    .filter(({ script }) => script.includes("'hermes_cli.windows_ssh_runtime' 'write-lock'") && !script.includes("'spawn'"))
    .map(({ stdinData }) => JSON.parse(stdinData))

for (const [label, devServer, expected] of [
  ['a dev build hands its renderer origin to the Windows serve', 'http://127.0.0.1:5175', 'http://127.0.0.1:5175'],
  ['a packaged build hands the Windows serve no renderer origin', '', '']
] as const) {
  test(`Windows spawn: ${label}`, async (): Promise<void> => {
    // The backend trusts a dev renderer's http origin only when it inherits HERMES_DESKTOP_DEV_SERVER;
    // windows_ssh_runtime.spawn_backend sets it from the payload's devServer, and only when non-empty.
    const ssh = ownedWindowsSsh(null)

    const result = await connectAs(ssh, devServer)
    const spawn = spawnOf(ssh)

    assert.equal(result.reused, false)
    assert.ok(spawn, 'a backend must be spawned')
    assert.ok(spawn.includes(`"devServer":"${expected}"`), 'the spawn payload names its renderer')
    assert.ok(spawn.includes(`devServer='${expected}'`), 'the record published under the spawn mutex names it')
    assert.deepEqual(
      writtenLocks(ssh).map(lock => lock.devServer),
      [expected, expected],
      'every record written for the new backend names its renderer'
    )
  })
}

for (const [label, recorded, current, reused] of [
  ['the same dev renderer reuses its backend', 'http://127.0.0.1:5174', 'http://127.0.0.1:5174', true],
  ['a dev renderer on another port replaces the backend', 'http://127.0.0.1:5174', 'http://127.0.0.1:5175', false],
  ['a dev build replaces a backend a packaged build spawned', undefined, 'http://127.0.0.1:5175', false],
  ['a packaged build replaces a backend a dev build spawned', 'http://127.0.0.1:5174', '', false],
  ['a packaged build reuses a packaged backend', undefined, '', true]
] as const) {
  test(`Windows SSH reuse: ${label}`, async (): Promise<void> => {
    // Reusing a backend spawned for another renderer would refuse every renderer socket.
    const ssh = ownedWindowsSsh(ownedLock(recorded))

    const result = await connectAs(ssh, current)
    const spawn = spawnOf(ssh)

    assert.equal(result.reused, reused)

    if (reused) {
      assert.equal(result.pid, 333)
      assert.equal(spawn, undefined, 'a compatible backend must not be respawned')

      return
    }

    assert.equal(result.pid, 890)
    assert.ok(
      ssh.execs.some(({ script }) => script.includes("'terminate' '333'")),
      'the owned backend must be reaped, not left running'
    )
    assert.ok(spawn?.includes(`"devServer":"${current}"`), 'the new backend is spawned for this renderer')
  })
}
