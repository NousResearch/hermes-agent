/**
 * Tests for electron/installs-ipc.ts.
 *
 * The IPC layer is exercised with a fake ipcMain and a fake runner, so these
 * tests need no electron and no child process.
 */

import assert from 'node:assert/strict'

import { afterEach, beforeEach, test, vi } from 'vitest'

import {
  createInstallsNotice,
  createInstallsRunner,
  type InstallsBackend,
  type InstallsChild,
  type InstallsNotice,
  type InstallsRunOutcome,
  registerInstallsIpc
} from './installs-ipc'

const noNotice = { take: (): null => null }

const SAMPLE_LIST = {
  current: 'abc',
  installs: [
    {
      id: 'abc',
      root: '/home/u/hermes-agent',
      steward: 'git',
      version: '1.2.3',
      sources: ['path'],
      current: true,
      package_full_name: null,
      removable: false,
      action: 'refuse',
      refusal: 'not removed: this install is running'
    }
  ],
  launchers: [],
  notice: { count: 0, dismissed: false, partial: false }
}

type Handler = (event: unknown, payload?: unknown) => Promise<unknown>

function fakeIpcMain(): { handlers: Map<string, Handler>; handle: (channel: string, handler: Handler) => void } {
  const handlers = new Map<string, Handler>()

  return {
    handlers,
    handle: (channel, handler) => {
      handlers.set(channel, handler)
    }
  }
}

const call = (ipc: ReturnType<typeof fakeIpcMain>, channel: string, payload?: unknown): Promise<unknown> =>
  ipc.handlers.get(channel)!(null, payload)

test('registerInstallsIpc list parses the CLI JSON', async () => {
  const ipc = fakeIpcMain()
  const runs: string[][] = []

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: args => {
      runs.push(args)

      return Promise.resolve({ code: 0, stdout: `noise\n${JSON.stringify(SAMPLE_LIST, null, 2)}\n`, stderr: '' })
    },
    notice: noNotice,
    logDebug: () => undefined
  })

  const list = (await call(ipc, 'hermes:installs:list')) as typeof SAMPLE_LIST | null

  assert.deepEqual(runs, [['installs', 'list', '--json']])
  assert.equal(list?.current, 'abc')
  assert.equal(list?.installs[0]?.removable, false)
})

test('registerInstallsIpc list returns null on a non-zero exit and bad JSON', async () => {
  const ipc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: () => Promise.resolve({ code: 1, stdout: '', stderr: 'boom' }),
    notice: noNotice,
    logDebug: () => undefined
  })

  assert.equal(await call(ipc, 'hermes:installs:list'), null)

  const badIpc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: badIpc,
    runInstalls: () => Promise.resolve({ code: 0, stdout: 'not json', stderr: '' }),
    notice: noNotice,
    logDebug: () => undefined
  })

  assert.equal(await call(badIpc, 'hermes:installs:list'), null)
})

test('registerInstallsIpc remove validates the id before building argv', async () => {
  const ipc = fakeIpcMain()
  const runs: string[][] = []

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: args => {
      runs.push(args)

      return Promise.resolve({ code: 0, stdout: 'Removed.', stderr: '' })
    },
    notice: noNotice,
    logDebug: () => undefined
  })

  const bad = (await call(ipc, 'hermes:installs:remove', { id: 'rm -rf /' })) as { ok: boolean; error?: string }

  assert.equal(bad.ok, false)
  assert.equal(bad.error, 'invalid-id')
  assert.deepEqual(runs, [])

  const good = (await call(ipc, 'hermes:installs:remove', { id: 'abc123' })) as { ok: boolean }

  assert.equal(good.ok, true)
  assert.deepEqual(runs, [['installs', 'remove', 'abc123', '--yes']])
})

test('registerInstallsIpc remove surfaces the CLI refusal on exit 1', async () => {
  const ipc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: () => Promise.resolve({ code: 1, stdout: '', stderr: 'not removed: this install is running' }),
    notice: noNotice,
    logDebug: () => undefined
  })

  const result = (await call(ipc, 'hermes:installs:remove', { id: 'abc' })) as { ok: boolean; message?: string }

  assert.equal(result.ok, false)
  assert.equal(result.message, 'not removed: this install is running')
})

test('registerInstallsIpc dismiss runs the dismiss argv', async () => {
  const ipc = fakeIpcMain()
  const runs: string[][] = []

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: args => {
      runs.push(args)

      return Promise.resolve({ code: 0, stdout: 'ok', stderr: '' })
    },
    notice: noNotice,
    logDebug: () => undefined
  })

  const result = (await call(ipc, 'hermes:installs:dismiss')) as { ok: boolean }

  assert.equal(result.ok, true)
  assert.deepEqual(runs, [['installs', 'dismiss']])
})

// --- boot notice ---

const NOTICED = {
  current: 'abc',
  installs: [],
  launchers: [],
  notice: { count: 2, dismissed: false, partial: false }
}

let notice: InstallsNotice
let signals: number
let debugs: string[]
let runnerOutcome: InstallsRunOutcome
let runnerShouldReject: boolean

const runner = (): Promise<InstallsRunOutcome> =>
  runnerShouldReject ? Promise.reject(new Error('spawn failed')) : Promise.resolve(runnerOutcome)

const check = (): void =>
  notice.check({
    runInstalls: runner,
    signalNotice: () => {
      signals += 1
    },
    logDebug: message => debugs.push(message)
  })

const wait = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms))

beforeEach(() => {
  notice = createInstallsNotice()
  signals = 0
  debugs = []
  runnerOutcome = { code: 0, stdout: JSON.stringify(NOTICED), stderr: '' }
  runnerShouldReject = false
})

afterEach(() => {
  runnerOutcome = { code: null, stdout: '', stderr: '' }
  runnerShouldReject = false
})

test('boot notice waits for the renderer: found with no window listening, taken once on mount', async () => {
  check()
  await wait(10)

  // Nothing listened when it was found (the renderer had not mounted). The pull still gets it.
  assert.equal(signals, 1)
  assert.deepEqual(notice.take(), { count: 2, partial: false })
  assert.equal(notice.take(), null)
  assert.deepEqual(debugs, [])
})

test('boot notice stays silent when dismissed, empty, failed, or malformed', async () => {
  for (const outcome of [
    { code: 0, stdout: JSON.stringify({ ...NOTICED, notice: { count: 2, dismissed: true, partial: false } }), stderr: '' },
    { code: 0, stdout: JSON.stringify({ ...NOTICED, notice: { count: 0, dismissed: false, partial: false } }), stderr: '' },
    { code: 1, stdout: '', stderr: 'boom' },
    { code: 0, stdout: 'half printed {', stderr: '' }
  ]) {
    runnerOutcome = outcome
    check()
    await wait(10)
  }

  assert.equal(signals, 0)
  assert.equal(notice.take(), null)
})

test('boot notice is found once per app launch, and a launch with none keeps checking', async () => {
  runnerOutcome = { code: 0, stdout: JSON.stringify({ ...NOTICED, notice: { count: 0, dismissed: false, partial: false } }), stderr: '' }
  check()
  await wait(10)
  assert.equal(signals, 0)

  runnerOutcome = { code: 0, stdout: JSON.stringify(NOTICED), stderr: '' }
  check()
  await wait(10)
  check()
  await wait(10)

  assert.equal(signals, 1)
})

test('boot notice failures are logged, never thrown', async () => {
  runnerShouldReject = true
  check()
  await wait(10)

  assert.equal(signals, 0)
  assert.equal(notice.take(), null)
  assert.equal(debugs.length, 1)
  assert.match(debugs[0]!, /spawn failed/)
})

test('registerInstallsIpc take-notice hands the renderer the pending notice', async () => {
  const ipc = fakeIpcMain()

  registerInstallsIpc({
    ipcMain: ipc,
    runInstalls: runner,
    notice: { take: () => ({ count: 3, partial: false }) },
    logDebug: () => undefined
  })

  assert.deepEqual(await call(ipc, 'hermes:installs:take-notice'), { count: 3, partial: false })
})

// --- the runner: which runtime answers ---

const BUNDLED: InstallsBackend = {
  command: '/payload/bin/hermes',
  args: ['installs', 'list', '--json'],
  env: { HERMES_RUNTIME_DIR: '/payload/tools' },
  root: '/payload/repo',
  shell: false
}

interface FakeChild extends InstallsChild {
  emit: (event: 'exit' | 'error', value: Error | null | number) => void
  killed: boolean
  push: (chunk: string) => void
}

function fakeChild(): FakeChild {
  const listeners: Record<string, ((value: never) => void)[]> = {}
  const onData: ((chunk: string) => void)[] = []

  const child: FakeChild = {
    killed: false,
    stdout: { on: (_event, listener) => void onData.push(listener as (chunk: string) => void) },
    stderr: { on: () => undefined },
    on: ((event: string, listener: (value: never) => void) => {
      ;(listeners[event] ??= []).push(listener)
    }) as FakeChild['on'],
    kill: () => {
      child.killed = true
    },
    emit: (event, value) => listeners[event]?.forEach(listener => listener(value as never)),
    push: chunk => onData.forEach(listener => listener(chunk))
  }

  return child
}

test('the runner runs the backend the app runs, not a fixed venv python', async () => {
  const child = fakeChild()
  const spawned: { args: string[]; command: string; cwd?: string; env: NodeJS.ProcessEnv }[] = []

  const run = createInstallsRunner({
    resolveBackend: async () => BUNDLED,
    spawn: (command, args, options) => {
      spawned.push({ command, args, ...options })

      return child
    },
    hermesHome: '/home/u/.hermes'
  })

  const outcome = run(['installs', 'list', '--json'])
  await wait(0)
  child.push('{"ok":')
  child.push('true}')
  child.emit('exit', 0)

  assert.deepEqual(await outcome, { code: 0, stdout: '{"ok":true}', stderr: '' })
  assert.equal(spawned.length, 1)
  assert.equal(spawned[0]!.command, '/payload/bin/hermes')
  assert.deepEqual(spawned[0]!.args, ['installs', 'list', '--json'])
  assert.equal(spawned[0]!.cwd, '/payload/repo')
  assert.equal(spawned[0]!.env.HERMES_RUNTIME_DIR, '/payload/tools')
  assert.equal(spawned[0]!.env.HERMES_HOME, '/home/u/.hermes')
})

test('the runner reports no runtime instead of spawning when none is installed yet', async () => {
  const run = createInstallsRunner({
    resolveBackend: async () => ({ ...BUNDLED, command: null, label: 'Hermes Agent not installed yet' }),
    spawn: () => {
      throw new Error('must not spawn')
    },
    hermesHome: '/home/u/.hermes'
  })

  assert.deepEqual(await run(['installs', 'list', '--json']), {
    code: null,
    stdout: '',
    stderr: 'Hermes Agent not installed yet'
  })
})

test('the runner kills a command that outlives its timeout, and a remove waits longer than a list', async () => {
  vi.useFakeTimers()

  try {
    const children: FakeChild[] = []

    const run = createInstallsRunner({
      resolveBackend: async () => BUNDLED,
      spawn: () => {
        const child = fakeChild()
        children.push(child)

        return child
      },
      hermesHome: '/home/u/.hermes'
    })

    const list = run(['installs', 'list', '--json'])
    const remove = run(['installs', 'remove', 'abc', '--yes'])
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(31_000)

    assert.equal((await list).stderr, 'timeout')
    assert.deepEqual(children.map(child => child.killed), [true, false])

    await vi.advanceTimersByTimeAsync(300_000)

    assert.equal((await remove).stderr, 'timeout')
    assert.equal(children[1]!.killed, true)
  } finally {
    vi.useRealTimers()
  }
})
