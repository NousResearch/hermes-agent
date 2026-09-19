import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import { test } from 'vitest'

import { getSshBinary, probeSshBinary, resolveSshBinary, sshBinaryCandidates } from './ssh-binary'

type FakeBehavior = {
  code?: number
  output?: string
  spawnError?: boolean
}

// A spawnFn that scripts each candidate's `ssh -V` outcome. Emissions are
// queued so the probe can attach its listeners first.
function scriptedSpawn(behaviors: Record<string, FakeBehavior>) {
  const calls: string[] = []

  const spawnFn = (command: string) => {
    calls.push(command)

    const behavior = behaviors[command] ?? { spawnError: true }
    const child: any = new EventEmitter()

    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()
    child.kill = () => {}

    queueMicrotask(() => {
      if (behavior.spawnError) {
        child.emit('error', new Error(`spawn ${command} ENOENT`))

        return
      }

      if (behavior.output) {
        child.stderr.emit('data', Buffer.from(behavior.output))
      }

      child.emit('close', behavior.code ?? 0)
    })

    return child
  }

  return { calls, spawnFn }
}

const nothingExists = () => false

const existsOnly =
  (...files: string[]) =>
  (p: string) =>
    files.includes(p)

const WIN_ENV = {
  SystemRoot: 'C:\\Windows',
  LOCALAPPDATA: 'C:\\Users\\test\\AppData\\Local',
  ProgramFiles: 'C:\\Program Files',
  'ProgramFiles(x86)': 'C:\\Program Files (x86)'
}

const SYSTEM32_SSH = 'C:\\Windows\\System32\\OpenSSH\\ssh.exe'
const GIT_SSH = 'C:\\Program Files\\Git\\usr\\bin\\ssh.exe'
const PORTABLE_GIT_SSH = 'C:\\Users\\test\\AppData\\Local\\hermes\\git\\usr\\bin\\ssh.exe'
const HEALTHY = { code: 0, output: 'OpenSSH_for_Windows_9.5p1, LibreSSL 3.8.2' }

test('win32 candidates: env override first, deduped against the inbox path', () => {
  const candidates = sshBinaryCandidates({ ...WIN_ENV, HERMES_SSH_PATH: SYSTEM32_SSH }, nothingExists)

  assert.deepEqual(candidates, [SYSTEM32_SSH, 'ssh'])
})

test('win32 candidates: no override and no Git install leaves inbox + bare ssh', () => {
  const candidates = sshBinaryCandidates({ SystemRoot: 'D:\\Win' }, nothingExists)

  assert.deepEqual(candidates, ['D:\\Win\\System32\\OpenSSH\\ssh.exe', 'ssh'])
})

test('win32 candidates: Git for Windows ssh comes from the shared Git discovery', () => {
  // System-wide install (ProgramFiles\Git\bin\bash.exe) -> sibling usr/bin ssh.
  const candidates = sshBinaryCandidates(WIN_ENV, existsOnly('C:\\Program Files\\Git\\bin\\bash.exe'))

  assert.deepEqual(candidates, [SYSTEM32_SSH, 'ssh', GIT_SSH])
})

test('win32 candidates: Hermes PortableGit and HERMES_GIT_BASH_PATH are covered', () => {
  const portable = sshBinaryCandidates(
    WIN_ENV,
    existsOnly('C:\\Users\\test\\AppData\\Local\\hermes\\git\\usr\\bin\\bash.exe')
  )

  assert.deepEqual(portable, [SYSTEM32_SSH, 'ssh', PORTABLE_GIT_SSH])

  const viaEnv = sshBinaryCandidates(
    { ...WIN_ENV, HERMES_GIT_BASH_PATH: 'D:\\tools\\git\\bin\\bash.exe' },
    existsOnly('D:\\tools\\git\\bin\\bash.exe')
  )

  assert.deepEqual(viaEnv, [SYSTEM32_SSH, 'ssh', 'D:\\tools\\git\\usr\\bin\\ssh.exe'])
})

test('win32 candidates: a PATH-resolved ssh identical to the inbox binary is deduped', () => {
  const env = { ...WIN_ENV, Path: 'C:\\Windows\\System32\\OpenSSH;C:\\tools' }
  const candidates = sshBinaryCandidates(env, existsOnly(SYSTEM32_SSH))

  assert.deepEqual(candidates, [SYSTEM32_SSH])
})

test('non-Windows platforms keep the bare ssh and never probe', async () => {
  const spawnFn = () => {
    throw new Error('must not be called')
  }

  assert.equal(await resolveSshBinary({ platform: 'darwin', env: {}, spawnFn, fileExists: nothingExists }), 'ssh')
  assert.equal(await resolveSshBinary({ platform: 'linux', env: {}, spawnFn, fileExists: nothingExists }), 'ssh')
})

test('a healthy HERMES_SSH_PATH override wins', async () => {
  const { calls, spawnFn } = scriptedSpawn({ 'D:\\tools\\ssh.exe': HEALTHY })
  const binary = await resolveSshBinary({
    platform: 'win32',
    env: { ...WIN_ENV, HERMES_SSH_PATH: 'D:\\tools\\ssh.exe' },
    spawnFn,
    fileExists: nothingExists
  })

  assert.equal(binary, 'D:\\tools\\ssh.exe')
  assert.deepEqual(calls, ['D:\\tools\\ssh.exe'])
})

test('a broken HERMES_SSH_PATH override is skipped, not fatal', async () => {
  // The #103288 failure mode on the override, with a healthy inbox binary.
  const { calls, spawnFn } = scriptedSpawn({
    'D:\\tools\\ssh.exe': { code: 255 },
    [SYSTEM32_SSH]: HEALTHY
  })
  const warnings: string[] = []
  const binary = await resolveSshBinary({
    platform: 'win32',
    env: { ...WIN_ENV, HERMES_SSH_PATH: 'D:\\tools\\ssh.exe' },
    spawnFn,
    fileExists: nothingExists,
    log: line => warnings.push(line)
  })

  assert.equal(binary, SYSTEM32_SSH)
  assert.deepEqual(calls, ['D:\\tools\\ssh.exe', SYSTEM32_SSH])
  assert.ok(warnings.some(line => line.includes('D:\\tools\\ssh.exe')))
})

test('a broken inbox OpenSSH falls back to the Git for Windows bundle', async () => {
  // Exactly the reporter's machine: System32 ssh dies instantly with no
  // output, bare ssh is not on the app PATH, Git's MSYS OpenSSH works.
  const { spawnFn } = scriptedSpawn({
    [SYSTEM32_SSH]: { code: 255 },
    [GIT_SSH]: { code: 0, output: 'OpenSSH_10.3p1, OpenSSL 3.5.2 5 Aug 2025' }
  })
  const binary = await resolveSshBinary({
    platform: 'win32',
    env: WIN_ENV,
    spawnFn,
    fileExists: existsOnly('C:\\Program Files\\Git\\bin\\bash.exe')
  })

  assert.equal(binary, GIT_SSH)
})

test('when nothing is healthy, fall back to bare ssh for the original error', async () => {
  const { spawnFn } = scriptedSpawn({})
  const binary = await resolveSshBinary({ platform: 'win32', env: WIN_ENV, spawnFn, fileExists: nothingExists })

  assert.equal(binary, 'ssh')
})

test('a chain of hanging binaries cannot starve the final Git fallback probe', async () => {
  // Three hanging candidates (override, inbox, PATH) would eat a naive total
  // budget before the Git fallback is ever tried; the resolver reserves a
  // full probe for the final candidate instead.
  const calls: string[] = []
  const spawnFn = (command: string) => {
    calls.push(command)

    const child: any = new EventEmitter()

    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()

    if (command === GIT_SSH) {
      queueMicrotask(() => {
        child.stderr.emit('data', Buffer.from(HEALTHY.output))
        child.emit('close', 0)
      })
    } else {
      child.kill = () => queueMicrotask(() => child.emit('close', null))
    }

    return child
  }

  const binary = await resolveSshBinary({
    platform: 'win32',
    env: { ...WIN_ENV, HERMES_SSH_PATH: 'D:\\bad\\ssh.exe', Path: 'C:\\tools' },
    spawnFn,
    fileExists: existsOnly('C:\\tools\\ssh.exe', 'C:\\Program Files\\Git\\bin\\bash.exe'),
    probeTimeoutMs: 50,
    budgetMs: 130
  })

  assert.equal(binary, GIT_SSH)
  assert.ok(calls.includes(GIT_SSH), 'final fallback candidate must always be probed')
})

test('probe rejects a zero-exit binary that is not OpenSSH', async () => {
  const { spawnFn } = scriptedSpawn({ 'ssh': { code: 0, output: 'usage: ssh ...' } })

  assert.equal(await probeSshBinary('ssh', spawnFn), false)
})

test('probe rejects a binary that cannot be spawned', async () => {
  const { spawnFn } = scriptedSpawn({})

  assert.equal(await probeSshBinary('ssh', spawnFn), false)
})

test('probe accepts a healthy binary reporting on stdout or stderr', async () => {
  const { spawnFn } = scriptedSpawn({ 'ssh': HEALTHY })

  assert.equal(await probeSshBinary('ssh', spawnFn), true)
})

test('probe kills a hanging binary and settles on its close', async () => {
  const child: any = new EventEmitter()

  child.stdout = new EventEmitter()
  child.stderr = new EventEmitter()

  let killed = false

  child.kill = () => {
    killed = true
    queueMicrotask(() => child.emit('close', null))
  }

  assert.equal(await probeSshBinary('ssh', () => child, 30), false)
  assert.ok(killed)
})

test('probe still settles within a grace period when the kill never reaps the child', async () => {
  const child: any = new EventEmitter()

  child.stdout = new EventEmitter()
  child.stderr = new EventEmitter()
  child.kill = () => {} // close never arrives

  const started = Date.now()

  assert.equal(await probeSshBinary('ssh', () => child, 30), false)
  assert.ok(Date.now() - started < 5_000, 'grace-settled probe must not hang the resolver')
})

test('getSshBinary memoizes the resolution for the process', () => {
  assert.equal(getSshBinary(), getSshBinary())
})
