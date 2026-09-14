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
    frameRate = 60
    focused = false
    readonly navigationHistory = {
      canGoBack: () => false,
      canGoForward: () => false,
      goBack: () => undefined,
      goForward: () => undefined
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
    setFrameRate(rate: number): void {
      this.frameRate = rate
    }

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

    focus(): void {
      this.focused = true
    }

    reload(): void {}
    stop(): void {}

    async executeJavaScript(source: string): Promise<unknown> {
      if (source.includes('__hermesWorkstationRefs')) {
        return {
          url: this.url,
          title: this.title,
          text: '',
          totalTextChars: 0,
          truncated: false,
          elements: []
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
    bounds = { x: 0, y: 0, width: 0, height: 0 }
    setBackgroundColor(): void {}
    setBounds(bounds: { x: number; y: number; width: number; height: number }): void {
      this.bounds = bounds
    }
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
      return { x: 0, y: 0, width: 1200, height: 800 }
    }

    destroy(): void {
      this.destroyed = true
    }
  }

  return {
    BrowserWindow: FakeBrowserWindow,
    WebContentsView: FakeWebContentsView,
    app: {
      getPath: () => os.tmpdir(),
      isReady: () => true,
      whenReady: async () => undefined,
      on: () => undefined
    },
    ipcMain: {
      handle: () => undefined
    },
    session: {
      fromPath: () => ({
        getCacheSize: async () => 4096,
        clearCache: async () => undefined,
        clearStorageData: async () => undefined,
        setPermissionRequestHandler: () => undefined,
        on: () => undefined,
        webRequest: {
          onBeforeSendHeaders: () => undefined,
          onHeadersReceived: () => undefined
        }
      })
    }
  }
})

vi.mock('electron', () => electron)

import { WorkstationBrowserRuntime } from './workstation-browser-runtime'
import { BrowserSessionStateFilePersistence } from './workstation-browser-session-state'

const createdDirs: string[] = []

function createPersistence(name: string): BrowserSessionStateFilePersistence {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-recovery-test-${name}-`))
  createdDirs.push(dir)

  return new BrowserSessionStateFilePersistence(
    path.join(dir, 'browser-session.json'),
    path.join(dir, 'browser-tasks.json')
  )
}

function executeControlRequest(runtime: WorkstationBrowserRuntime, request: Record<string, unknown>): Promise<unknown> {
  const method = Reflect.get(runtime as object, 'executeControlRequest') as (req: unknown) => Promise<unknown>

  return method.call(runtime, request)
}

afterEach(() => {
  for (const dir of createdDirs.splice(0)) {
    try {
      fs.rmSync(dir, { recursive: true, force: true })
    } catch {
      // Best effort cleanup.
    }
  }
})

test('V1 #13 Golden Recovery Scenario: interruption -> pause -> reconnect/rebind -> verification -> resume', async () => {
  const persistence = createPersistence('golden')
  const taskId = 'task-golden-mvp'
  const sessionId = 'session-golden-user-1'
  const kanbanCardId = 'card-golden-alpha'
  const runId = 'run-golden-999'

  // Step 1: Initial runtime boots and starts task with identities
  const first = new WorkstationBrowserRuntime(persistence)
  first.ensure()

  const nav1 = await executeControlRequest(first, {
    action: 'browser_navigate',
    task_id: taskId,
    session_id: sessionId,
    kanban_card_id: kanbanCardId,
    run_id: runId,
    arguments: { url: 'https://example.test/phase-1' }
  })

  assert.ok(nav1)

  // Verify initial task state and persistent bindings
  const task1 = first.listTasks().find(t => t.taskId === taskId)
  assert.equal(task1?.sessionHost, sessionId)
  assert.equal(task1?.kanbanCardId, kanbanCardId)
  assert.equal(task1?.runId, runId)
  assert.equal(first.state().tabs.filter(t => t.ownerTaskId === taskId).length, 1)

  // Step 2 & 3: Interruption happens -> pause
  await first.pause()
  assert.equal(first.state().paused, true)

  // Mutating requests during pause are rejected
  await assert.rejects(
    () =>
      executeControlRequest(first, {
        action: 'browser_navigate',
        task_id: taskId,
        arguments: { url: 'https://example.test/illegal' }
      }),
    /paused/
  )

  // Simulate process shutdown/interruption without erasing persistent metadata
  await first.destroy()

  // Step 4: Reconnect/Rebind on fresh runtime instance with same persistence
  const second = new WorkstationBrowserRuntime(persistence)
  second.ensure()

  // Verify task was recovered with exact persistent identities
  const recoveredTask = second.listTasks().find(t => t.taskId === taskId)
  assert.ok(recoveredTask)
  assert.equal(recoveredTask.taskId, taskId)
  assert.equal(recoveredTask.sessionHost, sessionId)
  assert.equal(recoveredTask.kanbanCardId, kanbanCardId)
  assert.equal(recoveredTask.runId, runId)

  // Step 5: Fail-closed identity verification: mismatch rejection
  await assert.rejects(
    () =>
      executeControlRequest(second, {
        action: 'browser_snapshot',
        task_id: taskId,
        session_id: sessionId,
        kanban_card_id: 'card-imposter',
        run_id: runId,
        arguments: {}
      }),
    /kanban card mismatch/
  )

  await assert.rejects(
    () =>
      executeControlRequest(second, {
        action: 'browser_snapshot',
        task_id: taskId,
        session_id: sessionId,
        kanban_card_id: kanbanCardId,
        run_id: 'run-imposter',
        arguments: {}
      }),
    /run mismatch/
  )

  // Step 6 & 7: Resume with authentic identities and continue work
  await second.resume()
  assert.equal(second.state().paused, false)

  const nav2 = await executeControlRequest(second, {
    action: 'browser_navigate',
    task_id: taskId,
    session_id: sessionId,
    kanban_card_id: kanbanCardId,
    run_id: runId,
    arguments: { url: 'https://example.test/phase-2-resumed' }
  })

  assert.ok(nav2)

  // Step 8: Assert exactly 1 owned page and persistent metadata intact
  assert.equal(second.state().tabs.filter(t => t.ownerTaskId === taskId).length, 1)
  const finalTask = second.listTasks().find(t => t.taskId === taskId)
  assert.equal(finalTask?.kanbanCardId, kanbanCardId)
  assert.equal(finalTask?.runId, runId)
  assert.equal(finalTask?.sessionHost, sessionId)

  await second.destroy()
})
