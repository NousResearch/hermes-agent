import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

const electron = vi.hoisted(() => {
  type Listener = (...args: unknown[]) => void
  const windows: FakeBrowserWindow[] = []

  class FakeWebContents {
    private destroyed = false
    private readonly listeners = new Map<string, Listener[]>()
    private title = ''
    private url = 'about:blank'
    readonly navigationHistory = {
      canGoBack: () => false,
      canGoForward: () => false,
      goBack: () => undefined,
      goForward: () => undefined
    }
    readonly debugger = {
      isAttached: () => false,
      attach: () => undefined,
      sendCommand: async () => ({})
    }

    on(event: string, listener: Listener): this {
      const current = this.listeners.get(event) ?? []

      current.push(listener)
      this.listeners.set(event, current)

      return this
    }

    private emit(event: string, ...args: unknown[]): void {
      for (const listener of this.listeners.get(event) ?? []) {
        listener(...args)
      }
    }

    setWindowOpenHandler(): void {}
    setFrameRate(): void {}
    focus(): void {}
    reload(): void {}
    stop(): void {}

    async loadURL(url: string): Promise<void> {
      this.url = url
      this.emit('did-start-loading')
      this.emit('did-navigate')
      this.emit('did-stop-loading')
    }

    getURL(): string {
      return this.url
    }

    getTitle(): string {
      return this.title
    }

    setTitle(title: string): void {
      this.title = title
      this.emit('page-title-updated')
    }

    isDestroyed(): boolean {
      return this.destroyed
    }

    close(): void {
      if (this.destroyed) {
        return
      }

      this.destroyed = true
      this.emit('destroyed')
    }

    async executeJavaScript(source: string): Promise<unknown> {
      if (source.includes('byRef.get')) {
        return { success: true, x: 50, y: 50 }
      }
      if (source.includes('__hermesWorkstationRefs')) {
        return {
          url: this.url,
          title: this.title,
          text: this.url.includes('challenge') ? 'confirm you are human' : '',
          totalTextChars: 0,
          truncated: false,
          elements: []
        }
      }
      if (source.includes('selectorUsed')) {
        return {
          count: 1,
          selector_used: 'article',
          items: [{ index: 1, title: 'Item 1', url: 'https://example.test/1', price: '$10' }]
        }
      }

      return {}
    }

    async capturePage(): Promise<{ toPNG: () => Uint8Array }> {
      return { toPNG: () => new Uint8Array() }
    }
  }

  class FakeWebContentsView {
    readonly webContents = new FakeWebContents()
    setBackgroundColor(): void {}
    setBounds(): void {}
  }

  class FakeBrowserWindow {
    destroyed = false
    readonly contentView = {
      children: [] as FakeWebContentsView[],
      addChildView: (view: FakeWebContentsView) => {
        if (!this.contentView.children.includes(view)) {
          this.contentView.children.push(view)
        }
      },
      removeChildView: (view: FakeWebContentsView) => {
        this.contentView.children = this.contentView.children.filter(candidate => candidate !== view)
      }
    }
    readonly webContents = { send: () => undefined }

    constructor() {
      windows.push(this)
    }

    static getAllWindows(): FakeBrowserWindow[] {
      return windows.filter(window => !window.destroyed)
    }

    static fromWebContents(): FakeBrowserWindow | null {
      return null
    }

    isDestroyed(): boolean {
      return this.destroyed
    }

    getContentBounds(): { x: number; y: number; width: number; height: number } {
      return { x: 0, y: 0, width: 1440, height: 900 }
    }
  }

  const browserSession = {
    getCacheSize: async () => 0,
    clearCache: async () => undefined
  }

  return {
    windows,
    FakeBrowserWindow,
    FakeWebContents,
    FakeWebContentsView,
    app: {
      getPath: () => 'C:/tmp/hermes-runtime-resilience-app',
      whenReady: () => new Promise<never>(() => undefined),
      on: () => undefined
    },
    ipcMain: { handle: () => undefined },
    session: { fromPath: () => browserSession }
  }
})

vi.mock('electron', () => ({
  app: electron.app,
  BrowserWindow: electron.FakeBrowserWindow,
  ipcMain: electron.ipcMain,
  session: electron.session,
  WebContentsView: electron.FakeWebContentsView
}))

import { WorkstationBrowserRuntime, workstationBrowserSessionStatePath } from './workstation-browser-runtime'
import { BrowserSessionStateFilePersistence } from './workstation-browser-session-state'

const tempRoots: string[] = []

afterEach(() => {
  delete process.env.HERMES_WORKSTATION_HOME
  electron.windows.splice(0)

  for (const root of tempRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function runtimeHome(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-runtime-resilience-'))

  tempRoots.push(root)
  process.env.HERMES_WORKSTATION_HOME = root

  return root
}

function failNextRenamePersistence(home: string): {
  persistence: BrowserSessionStateFilePersistence
  failNextRename(): void
} {
  let shouldFail = false

  const io = {
    ...fs,
    renameSync: (...args: Parameters<typeof fs.renameSync>) => {
      if (shouldFail) {
        shouldFail = false
        throw new Error('simulated BrowserTask destroy persistence failure')
      }

      fs.renameSync(...args)
    }
  }

  return {
    persistence: new BrowserSessionStateFilePersistence(
      workstationBrowserSessionStatePath(),
      path.join(home, 'Runtime', 'browser-tasks.json'),
      io
    ),
    failNextRename(): void {
      shouldFail = true
    }
  }
}

function persistedTaskIds(): string[] {
  const composite = JSON.parse(fs.readFileSync(workstationBrowserSessionStatePath(), 'utf-8')) as {
    browserTasks: { tasks: Array<{ taskId: string }> }
  }

  return composite.browserTasks.tasks.map(task => task.taskId)
}

function persistedTask(
  taskId: string
): { taskId: string; sessionHost: string | null; kanbanCardId?: string | null; runId?: string | null } | undefined {
  const composite = JSON.parse(fs.readFileSync(workstationBrowserSessionStatePath(), 'utf-8')) as {
    browserTasks: {
      tasks: Array<{ taskId: string; sessionHost: string | null; kanbanCardId?: string | null; runId?: string | null }>
    }
  }

  return composite.browserTasks.tasks.find(task => task.taskId === taskId)
}

function executeControlRequest(
  runtime: WorkstationBrowserRuntime,
  request: Record<string, unknown>
): Promise<Record<string, unknown>> {
  return (
    runtime as unknown as {
      executeControlRequest(value: Record<string, unknown>): Promise<Record<string, unknown>>
    }
  ).executeControlRequest(request)
}

async function assertFreshRecreation(
  runtime: WorkstationBrowserRuntime,
  taskId: string,
  originalTabId: string
): Promise<void> {
  await runtime.navigate('https://ordinary.example.test/after-destroy-failure')
  assert.deepEqual(persistedTaskIds(), [])

  runtime.createTask({ taskId })
  const recreated = runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId)

  assert.equal(recreated.length, 1)
  assert.notEqual(recreated[0].id, originalTabId)
  assert.equal(recreated[0].url, 'about:blank')
}

test('failed BrowserTask destroy persistence completes in-memory cleanup before the same task id is recreated', async () => {
  const home = runtimeHome()
  const fault = failNextRenamePersistence(home)
  const runtime = new WorkstationBrowserRuntime(fault.persistence)
  const taskId = 'task-destroy-failure'

  runtime.createTask({ taskId })
  const originalTab = runtime.state().tabs.find(tab => tab.ownerTaskId === taskId)

  assert.ok(originalTab)

  const originalContents = runtime.getWebContents(originalTab.id) as unknown as InstanceType<
    typeof electron.FakeWebContents
  >

  assert.ok(originalContents)
  await originalContents.loadURL('https://task.example.test/workspace')
  assert.deepEqual(persistedTaskIds(), [taskId])

  fault.failNextRename()
  assert.throws(() => runtime.destroyTask(taskId), /simulated BrowserTask destroy persistence failure/)

  // The explicit destroy already completed its process-local semantic mutation:
  // task metadata is gone and the prior page is dead, even though disk remains
  // at the previous complete snapshot until another composite save succeeds.
  assert.deepEqual(runtime.listTasks(), [])
  assert.equal(originalContents.isDestroyed(), true)
  assert.deepEqual(persistedTaskIds(), [taskId])

  // Reusing the same task id after convergence is a new BrowserTask. It must not
  // inherit the destroyed page's pending URL/identity metadata.
  await assertFreshRecreation(runtime, taskId, originalTab.id)
  await runtime.destroy()
})

test('resource projection is derived from the live BrowserTask state without creating another store', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()

  runtime.createTask({ taskId: 'resource-task', sessionHost: 'resource-session', runId: 'resource-run' })
  const snapshot = runtime.resources()
  const task = snapshot.resources.find(resource => resource.resource_type === 'browser_task')

  assert.equal(snapshot.schema_version, 1)
  assert.equal(task?.resource_id, 'browser-task:resource-task')
  assert.equal(task?.session_id, 'resource-session')
  assert.equal(task?.state.lineage && (task.state.lineage as { run_id?: string }).run_id, 'resource-run')
  assert.equal(snapshot.resources.filter(resource => resource.task_id === 'resource-task').length, 2)
  await runtime.destroy()
})

test('event projection reads the bounded canonical journal for one task', async () => {
  const home = runtimeHome()
  const previousHermesHome = process.env.HERMES_HOME
  process.env.HERMES_HOME = home
  const runtime = new WorkstationBrowserRuntime()

  try {
    const taskId = 'event-task'
    runtime.createTask({ taskId, sessionHost: 'event-session' })
    const journalPath = path.join(home, 'workstation', 'journals', `${taskId}.jsonl`)
    fs.mkdirSync(path.dirname(journalPath), { recursive: true })
    fs.writeFileSync(
      journalPath,
      [
        JSON.stringify({
          event_id: 'event-1',
          kind: 'task_started',
          task_id: taskId,
          session_id: 'event-session',
          message: 'started',
          timestamp: '2026-09-11T12:00:00.000Z'
        }),
        JSON.stringify({
          event_id: 'event-2',
          kind: 'progress',
          task_id: taskId,
          session_id: 'event-session',
          message: 'progress',
          timestamp: '2026-09-11T12:00:01.000Z'
        })
      ].join('\n'),
      'utf-8'
    )

    const snapshot = runtime.events(taskId, 1)

    assert.equal(snapshot.schema_version, 1)
    assert.equal(snapshot.task_id, taskId)
    assert.deepEqual(
      snapshot.events.map(event => event.event_id),
      ['event-2']
    )
    assert.equal(snapshot.events[0]?.session_id, 'event-session')
    assert.equal(snapshot.events[0]?.elapsed_seconds, 1)
  } finally {
    await runtime.destroy()
    if (previousHermesHome === undefined) {
      delete process.env.HERMES_HOME
    } else {
      process.env.HERMES_HOME = previousHermesHome
    }
  }
})

test('failed destroy also clears a recovery hint when the task page was already gone', async () => {
  const home = runtimeHome()
  const fault = failNextRenamePersistence(home)
  const runtime = new WorkstationBrowserRuntime(fault.persistence)
  const taskId = 'task-pending-destroy-failure'

  runtime.createTask({ taskId })
  const originalTab = runtime.state().tabs.find(tab => tab.ownerTaskId === taskId)

  assert.ok(originalTab)

  const originalContents = runtime.getWebContents(originalTab.id) as unknown as InstanceType<
    typeof electron.FakeWebContents
  >

  assert.ok(originalContents)
  await originalContents.loadURL('https://task.example.test/stale-recovery-url')

  // Simulate an already-gone renderer/page. BrowserTask remains logical while
  // BrowserSessionState keeps only a stale recovery hint for that page.
  originalContents.close()
  assert.equal(
    runtime.state().tabs.some(tab => tab.ownerTaskId === taskId),
    false
  )
  assert.equal(
    runtime.listTasks().some(task => task.taskId === taskId),
    true
  )
  assert.deepEqual(persistedTaskIds(), [taskId])

  fault.failNextRename()
  assert.throws(() => runtime.destroyTask(taskId), /simulated BrowserTask destroy persistence failure/)

  assert.deepEqual(runtime.listTasks(), [])
  assert.deepEqual(persistedTaskIds(), [taskId])

  await assertFreshRecreation(runtime, taskId, originalTab.id)
  await runtime.destroy()
})

test('controller-created BrowserTask persists its Hermes session identity and rejects cross-session retargeting', async () => {
  runtimeHome()
  const taskId = 'task-session-linked'
  const first = new WorkstationBrowserRuntime()

  await executeControlRequest(first, {
    action: 'browser_navigate',
    task_id: taskId,
    session_id: 'hermes-session-a',
    arguments: { url: 'https://example.test/session-a' }
  })

  assert.equal(first.listTasks().find(task => task.taskId === taskId)?.sessionHost, 'hermes-session-a')
  assert.equal(persistedTask(taskId)?.sessionHost, 'hermes-session-a')
  await first.destroy()

  const second = new WorkstationBrowserRuntime()
  second.ensure()
  assert.equal(second.listTasks().find(task => task.taskId === taskId)?.sessionHost, 'hermes-session-a')
  assert.equal(
    second.state().tabs.some(tab => tab.ownerTaskId === taskId),
    false
  )

  await assert.rejects(
    () =>
      executeControlRequest(second, {
        action: 'browser_navigate',
        task_id: taskId,
        session_id: 'hermes-session-b',
        arguments: { url: 'https://example.test/session-b' }
      }),
    /session identity mismatch/
  )
  assert.equal(second.listTasks().find(task => task.taskId === taskId)?.sessionHost, 'hermes-session-a')
  assert.equal(
    second.state().tabs.some(tab => tab.ownerTaskId === taskId),
    false
  )

  await executeControlRequest(second, {
    action: 'browser_navigate',
    task_id: taskId,
    session_id: 'hermes-session-a',
    arguments: { url: 'https://example.test/session-a-restored' }
  })
  assert.equal(second.state().tabs.filter(tab => tab.ownerTaskId === taskId).length, 1)
  assert.equal(persistedTask(taskId)?.sessionHost, 'hermes-session-a')
  await second.destroy()
})

test('controller binds an unbound existing BrowserTask once and rejects invalid session identities before mutation', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const taskId = 'task-manual-session-bind'

  runtime.createTask({ taskId })
  assert.equal(runtime.listTasks().find(task => task.taskId === taskId)?.sessionHost, null)

  await executeControlRequest(runtime, {
    action: 'browser_snapshot',
    task_id: taskId,
    session_id: 'hermes-session-a',
    arguments: {}
  })
  assert.equal(runtime.listTasks().find(task => task.taskId === taskId)?.sessionHost, 'hermes-session-a')
  assert.equal(persistedTask(taskId)?.sessionHost, 'hermes-session-a')

  await assert.rejects(
    () =>
      executeControlRequest(runtime, {
        action: 'browser_snapshot',
        task_id: taskId,
        session_id: 'hermes-session-b',
        arguments: {}
      }),
    /session identity mismatch/
  )
  assert.equal(runtime.listTasks().find(task => task.taskId === taskId)?.sessionHost, 'hermes-session-a')

  const tasksBeforeInvalid = runtime.listTasks().map(task => task.taskId)
  await assert.rejects(
    () =>
      executeControlRequest(runtime, {
        action: 'browser_navigate',
        task_id: 'task-invalid-session',
        session_id: 'bad\nsession',
        arguments: { url: 'https://example.test/invalid-session' }
      }),
    /invalid session identity/
  )
  assert.deepEqual(
    runtime.listTasks().map(task => task.taskId),
    tasksBeforeInvalid
  )
  assert.equal(
    runtime.state().tabs.some(tab => tab.ownerTaskId === 'task-invalid-session'),
    false
  )
  await runtime.destroy()
})

test('failed session identity persistence remains fail-closed and converges on the next composite write', async () => {
  const home = runtimeHome()
  const fault = failNextRenamePersistence(home)
  const runtime = new WorkstationBrowserRuntime(fault.persistence)
  const taskId = 'task-session-write-failure'

  runtime.createTask({ taskId })
  assert.equal(runtime.listTasks().find(task => task.taskId === taskId)?.sessionHost, null)
  assert.equal(persistedTask(taskId)?.sessionHost, null)

  fault.failNextRename()
  await assert.rejects(
    () =>
      executeControlRequest(runtime, {
        action: 'browser_snapshot',
        task_id: taskId,
        session_id: 'hermes-session-a',
        arguments: {}
      }),
    /simulated BrowserTask destroy persistence failure/
  )

  // The lifecycle mutation is already the process-local intent even though the
  // last complete durable snapshot still carries the old unbound value.
  assert.equal(runtime.listTasks().find(task => task.taskId === taskId)?.sessionHost, 'hermes-session-a')
  assert.equal(persistedTask(taskId)?.sessionHost, null)

  await assert.rejects(
    () =>
      executeControlRequest(runtime, {
        action: 'browser_snapshot',
        task_id: taskId,
        session_id: 'hermes-session-b',
        arguments: {}
      }),
    /session identity mismatch/
  )
  assert.equal(runtime.listTasks().find(task => task.taskId === taskId)?.sessionHost, 'hermes-session-a')

  await runtime.navigate('https://ordinary.example.test/session-bind-convergence')
  assert.equal(persistedTask(taskId)?.sessionHost, 'hermes-session-a')
  await runtime.destroy()
})

test('controller binds kanbanCardId and runId and rejects mismatch fail-closed across restarts', async () => {
  runtimeHome()
  const taskId = 'task-kanban-identity-linked'
  const first = new WorkstationBrowserRuntime()

  await executeControlRequest(first, {
    action: 'browser_navigate',
    task_id: taskId,
    session_id: 'hermes-session-k',
    kanban_card_id: 'card-alpha',
    run_id: 'run-101',
    arguments: { url: 'https://example.test/kanban-work' }
  })

  const taskFirst = first.listTasks().find(task => task.taskId === taskId)
  assert.equal(taskFirst?.sessionHost, 'hermes-session-k')
  assert.equal(taskFirst?.kanbanCardId, 'card-alpha')
  assert.equal(taskFirst?.runId, 'run-101')
  assert.equal(persistedTask(taskId)?.kanbanCardId, 'card-alpha')
  assert.equal(persistedTask(taskId)?.runId, 'run-101')
  await first.destroy()

  const second = new WorkstationBrowserRuntime()
  second.ensure()
  const taskSecond = second.listTasks().find(task => task.taskId === taskId)
  assert.equal(taskSecond?.sessionHost, 'hermes-session-k')
  assert.equal(taskSecond?.kanbanCardId, 'card-alpha')
  assert.equal(taskSecond?.runId, 'run-101')

  await assert.rejects(
    () =>
      executeControlRequest(second, {
        action: 'browser_snapshot',
        task_id: taskId,
        session_id: 'hermes-session-k',
        kanban_card_id: 'card-different',
        run_id: 'run-101',
        arguments: {}
      }),
    /kanban card mismatch/
  )

  await assert.rejects(
    () =>
      executeControlRequest(second, {
        action: 'browser_snapshot',
        task_id: taskId,
        session_id: 'hermes-session-k',
        kanban_card_id: 'card-alpha',
        run_id: 'run-different',
        arguments: {}
      }),
    /run mismatch/
  )

  // Idempotent with same identities succeeds and resumes task work
  const nav = await executeControlRequest(second, {
    action: 'browser_navigate',
    task_id: taskId,
    session_id: 'hermes-session-k',
    kanban_card_id: 'card-alpha',
    run_id: 'run-101',
    arguments: { url: 'https://example.test/kanban-work-resumed' }
  })

  assert.ok(nav)
  assert.equal(second.state().tabs.filter(tab => tab.ownerTaskId === taskId).length, 1)
  assert.equal(persistedTask(taskId)?.kanbanCardId, 'card-alpha')
  assert.equal(persistedTask(taskId)?.runId, 'run-101')
  await second.destroy()
})

test('browser_extract_items extracts structured cards via controller request', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const taskId = 'task-extract-test'

  await executeControlRequest(runtime, {
    action: 'browser_navigate',
    task_id: taskId,
    arguments: { url: 'https://example.test/store' }
  })

  const extracted = await executeControlRequest(runtime, {
    action: 'browser_extract_items',
    task_id: taskId,
    arguments: { selector: 'article', limit: 10 }
  })

  assert.equal(extracted.success, true)
  assert.equal(extracted.count, 1)
  assert.equal(extracted.selector_used, 'article')
  assert.equal((extracted.items as Array<Record<string, unknown>>)[0].title, 'Item 1')
  await runtime.destroy()
})

test('browser_snapshot detects verification wall and flags human handoff', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const taskId = 'task-wall-test'

  const nav = await executeControlRequest(runtime, {
    action: 'browser_navigate',
    task_id: taskId,
    arguments: { url: 'https://example.test/gz/account-verification' }
  })

  assert.equal(nav.success, true)
  assert.equal(nav.wall_detected, true)
  assert.match(String(nav.snapshot), /HUMAN_HANDOFF_REQUIRED/)
  assert.match(String(nav.wall_reason), /Verification URL pattern/)
  await runtime.destroy()
})

test('browser_type handles clear and append options without errors', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const taskId = 'task-type-test'

  await executeControlRequest(runtime, {
    action: 'browser_navigate',
    task_id: taskId,
    arguments: { url: 'https://example.test/form' }
  })

  const typed = await executeControlRequest(runtime, {
    action: 'browser_type',
    task_id: taskId,
    arguments: { ref: '@e1', text: 'cleared text', clear: true, append: false }
  })
  assert.equal(typed.success, true)

  const appended = await executeControlRequest(runtime, {
    action: 'browser_type',
    task_id: taskId,
    arguments: { ref: '@e1', text: ' more text', clear: false, append: true }
  })
  assert.equal(appended.success, true)
  await runtime.destroy()
})

