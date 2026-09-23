import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopLocalRuntime } from './desktop-local-runtime'

function fixture(overrides: Record<string, unknown> = {}) {
  const events: string[] = []
  let failure: Error | null = null
  let repairRequested = false

  const deps = {
    hermesHome: 'C:/test/hermes',
    activeRoot: 'C:/test/hermes/hermes-agent',
    venvRoot: 'C:/test/hermes/hermes-agent/venv',
    sourceRepoRoot: 'C:/test/source',
    installStamp: null,
    isWindows: process.platform === 'win32',
    isPackaged: false,
    isWsl: false,
    findPythonForRoot: async (root: string) => `${root}/.venv/python.exe`,
    venvRootForPython: (_python: string, root: string) => path.join(root, '.venv'),
    getVenvPython: (root: string) => path.join(root, 'python.exe'),
    fileExists: () => false,
    findSystemPython: async () => null,
    isHermesSourceRoot: (root: string) => root === 'C:/test/source',
    activeRuntimeState: async () => ({ shouldUseActiveRuntime: false, hasValidMarker: false }),
    findOnPath: () => null,
    isWindowsBinaryPathInWsl: () => false,
    looksLikeDesktopAppBinary: () => false,
    unwrapWindowsVenvHermesCommand: async () => null,
    isCommandScript: () => false,
    rememberLog: (message: string) => events.push(message),
    localBackendLifecycle: {
      start: async (run: () => Promise<unknown>) => {
        events.push('lifecycle.start')

        return run()
      },
      assertCanStart: () => events.push('lifecycle.assert')
    },
    firstRunBoot: {
      advanceBootProgress: async () => {
        events.push('progress')
      },
      broadcastBootstrapEvent: () => {},
      updateBootProgress: () => {}
    },
    handOffWindowsBootstrapRecovery: async () => false,
    writeBootstrapMarker: () => {},
    resolveGitBinary: () => 'git',
    findGitBash: () => 'bash',
    state: {
      get bootstrapRepairRequested() {
        return repairRequested
      },
      set bootstrapRepairRequested(value: boolean) {
        repairRequested = value
      },
      set bootstrapRepairAttempt(_value: number) {},
      set bootstrapAbortController(_value: AbortController | null) {},
      set bootstrapFailure(value: Error | null) {
        failure = value
      }
    },
    ...overrides
  }

  return {
    runtime: createDesktopLocalRuntime(deps as any),
    events,
    getFailure: () => failure,
    setRepairRequested: (value: boolean) => {
      repairRequested = value
    }
  }
}

test('development source wins the local resolver and an external backend stays under lifecycle admission', async () => {
  const { runtime, events } = fixture()
  const backend = await runtime.resolveHermesBackend(['serve'])

  assert.equal(backend.root, 'C:/test/source')
  assert.equal(backend.bootstrap, false)
  assert.equal(backend.command, 'C:/test/source/.venv/python.exe')

  const resolved = await runtime.ensureRuntime(backend, () => events.push('owned'))

  assert.equal(resolved, backend)
  assert.deepEqual(events, ['lifecycle.start', 'lifecycle.assert', 'owned', 'progress'])
})

test('bootstrap recovery handoff latches failure and refuses to launch a backend', async () => {
  const { runtime, getFailure, setRepairRequested } = fixture({
    isPackaged: true,
    isHermesSourceRoot: (root: string) => root === 'C:/test/hermes/hermes-agent',
    activeRuntimeState: async () => ({ shouldUseActiveRuntime: true, hasValidMarker: true }),
    handOffWindowsBootstrapRecovery: async () => true
  })

  assert.equal((await runtime.resolveHermesBackend(['serve'])).root, 'C:/test/hermes/hermes-agent')
  setRepairRequested(true)
  const repairBackend = await runtime.resolveHermesBackend(['serve'])

  assert.equal(repairBackend.kind, 'bootstrap-needed')

  await assert.rejects(
    runtime.ensureRuntime(repairBackend, () => {}),
    /handed off to Hermes Setup/
  )

  assert.equal((getFailure() as Error & { bootstrapHandedOff?: boolean })?.bootstrapHandedOff, true)
})
