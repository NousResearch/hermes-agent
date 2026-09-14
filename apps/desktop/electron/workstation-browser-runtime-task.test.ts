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
    private windowOpenHandler: ((details: { url: string }) => { action: string }) | null = null
    frameRate = 60
    focused = false
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

    setWindowOpenHandler(handler: (details: { url: string }) => { action: string }): void {
      this.windowOpenHandler = handler
    }

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

    simulateRenderProcessGone(): void {
      this.emit('render-process-gone', {}, { reason: 'crashed' })
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
      return { x: 0, y: 0, width: 1440, height: 900 }
    }
  }

  const extensions: Record<string, { id: string; name: string; version: string }> = {}
  const browserSession = {
    getCacheSize: async () => 0,
    clearCache: async () => undefined,
    loadExtension: async (extensionPath: string) => {
      const id = path.basename(extensionPath)
      const extension = { id, name: 'Fixture extension', version: '1.0.0' }
      extensions[id] = extension
      return extension
    },
    getAllExtensions: () => extensions,
    removeExtension: (id: string) => { delete extensions[id] }
  }

  return {
    windows,
    FakeBrowserWindow,
    FakeWebContents,
    FakeWebContentsView,
    app: {
      getPath: () => 'C:/tmp/hermes-runtime-task-app',
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

import {
  getStandardChromeUserAgent,
  WorkstationBrowserRuntime,
  workstationBrowserSessionStatePath
} from './workstation-browser-runtime'
import { BrowserSessionStateFilePersistence } from './workstation-browser-session-state'

const tempRoots: string[] = []
afterEach(() => {
  delete process.env.HERMES_WORKSTATION_HOME
  delete process.env.HERMES_HOME
  electron.windows.splice(0)

  for (const root of tempRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function runtimeHome(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-runtime-task-'))
  tempRoots.push(root)
  process.env.HERMES_WORKSTATION_HOME = root

  return root
}

function hostWindow(): InstanceType<typeof electron.FakeBrowserWindow> {
  return new electron.FakeBrowserWindow()
}

function interruptibleSessionPersistence(home: string): {
  persistence: BrowserSessionStateFilePersistence
  failAfterSuccessfulReplacements(count: number): void
} {
  let successfulReplacementsBeforeFailure: number | null = null

  const io = {
    ...fs,
    renameSync: (...args: Parameters<typeof fs.renameSync>) => {
      if (successfulReplacementsBeforeFailure === 0) {
        successfulReplacementsBeforeFailure = null
        throw new Error('simulated BrowserSessionState projection interruption')
      }

      if (successfulReplacementsBeforeFailure !== null) {
        successfulReplacementsBeforeFailure -= 1
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
    failAfterSuccessfulReplacements(count: number): void {
      successfulReplacementsBeforeFailure = count
    }
  }
}

function persistedComposite(): {
  activeTabId: string | null
  tabs: Array<{ id: string; browserTaskId: string | null; safeUrl: string | null; safeTitle: string | null }>
  browserTasks: { tasks: Array<{ taskId: string; status: string; recoveryState: string | null }> }
} {
  return JSON.parse(fs.readFileSync(workstationBrowserSessionStatePath(), 'utf-8'))
}

test('ordinary tabs restore in order with the active logical tab and only safe URL/title metadata', async () => {
  const home = runtimeHome()
  const first = new WorkstationBrowserRuntime()
  const initial = first.ensure().tabs[0]
  assert.ok(initial)
  const initialContents = first.getWebContents(initial.id) as unknown as InstanceType<typeof electron.FakeWebContents>
  assert.ok(initialContents)
  await initialContents.loadURL('https://example.test/account?access_token=page-secret#fragment-secret')
  initialContents.setTitle('Access token = title-secret')

  const withSecond = first.createTab('https://docs.example.test/guide?q=ordinary-search', true)
  const secondTab = withSecond.tabs.at(-1)
  assert.ok(secondTab)
  const secondContents = first.getWebContents(secondTab.id) as unknown as InstanceType<typeof electron.FakeWebContents>
  assert.ok(secondContents)
  secondContents.setTitle('Example Docs — Guide')
  assert.equal(first.state().tabs.find(tab => tab.id === secondTab.id)?.title, 'Example Docs — Guide')
  assert.equal(first.state().activeTabId, secondTab.id)

  await first.destroy()

  const persisted = fs.readFileSync(path.join(home, 'Runtime', 'browser-session.json'), 'utf-8')
  assert.equal(persisted.includes('page-secret'), false)
  assert.equal(persisted.includes('fragment-secret'), false)
  assert.equal(persisted.includes('title-secret'), false)
  assert.equal(persisted.includes('ordinary-search'), false)
  assert.equal(persisted.includes('WebContents'), false)

  const second = new WorkstationBrowserRuntime()
  const restored = second.ensure()
  assert.deepEqual(
    restored.tabs.map(tab => tab.id),
    [initial.id, secondTab.id]
  )
  assert.deepEqual(
    restored.tabs.map(tab => tab.url),
    ['https://example.test/account', 'https://docs.example.test/guide']
  )
  assert.equal(restored.tabs[0].title, 'New Tab')
  assert.equal(restored.tabs[1].title, 'New Tab')
  assert.equal(restored.activeTabId, secondTab.id)

  await second.destroy()
})

test('BrowserTask metadata coexists with ordinary tabs and restores one task page lazily', async () => {
  runtimeHome()
  const first = new WorkstationBrowserRuntime()
  const ordinary = first.ensure().tabs[0]
  first.createTask({ taskId: 'task-session', sessionHost: 'hermes-session-1' })
  const taskTab = first.state().tabs.find(tab => tab.ownerTaskId === 'task-session')
  assert.ok(taskTab)
  const taskContents = first.getWebContents(taskTab.id) as unknown as InstanceType<typeof electron.FakeWebContents>
  assert.ok(taskContents)
  await taskContents.loadURL('https://task.example.test/work?session_id=page-secret')
  taskContents.setTitle('Task Workspace')
  assert.equal(first.state().tabs.find(tab => tab.id === taskTab.id)?.title, 'Task Workspace')
  const tail = first.createTab('https://tail.example.test/ordinary', false).tabs.at(-1)
  assert.ok(tail)
  await first.destroy()

  const second = new WorkstationBrowserRuntime()
  const beforeShow = second.ensure()
  assert.deepEqual(
    beforeShow.tabs.map(tab => tab.id),
    [ordinary.id, tail.id]
  )
  assert.equal(second.listTasks()[0]?.taskId, 'task-session')
  assert.equal(second.listTasks()[0]?.sessionHost, 'hermes-session-1')
  assert.equal(
    beforeShow.tabs.some(tab => tab.ownerTaskId === 'task-session'),
    false
  )

  const shown = second.showTask('task-session', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  assert.equal(shown.recoveryState, 'recreated')
  const recovered = second.state().tabs.filter(tab => tab.ownerTaskId === 'task-session')
  assert.equal(recovered.length, 1)
  assert.equal(recovered[0].id, taskTab.id)
  assert.equal(recovered[0].url, 'https://task.example.test/work')
  assert.equal(recovered[0].title, 'New Tab')
  assert.deepEqual(
    second.state().tabs.map(tab => tab.id),
    [ordinary.id, taskTab.id, tail.id]
  )

  second.showTask('task-session', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  assert.equal(second.state().tabs.filter(tab => tab.ownerTaskId === 'task-session').length, 1)
  await second.destroy()
})

test('a lazy BrowserTask remains the logical active tab until its page is recreated', async () => {
  runtimeHome()
  const first = new WorkstationBrowserRuntime()
  first.ensure()
  first.createTask({ taskId: 'task-active' })
  const taskTab = first.state().tabs.find(tab => tab.ownerTaskId === 'task-active')
  assert.ok(taskTab)
  first.activateTab(taskTab.id)
  await first.destroy()

  const second = new WorkstationBrowserRuntime()
  const beforeShow = second.ensure()
  assert.equal(
    beforeShow.tabs.some(tab => tab.ownerTaskId === 'task-active'),
    false
  )
  assert.equal(JSON.parse(fs.readFileSync(workstationBrowserSessionStatePath(), 'utf-8')).activeTabId, taskTab.id)

  second.showTask('task-active', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  const recovered = second.state().tabs.filter(tab => tab.ownerTaskId === 'task-active')
  assert.equal(recovered.length, 1)
  assert.equal(recovered[0].id, taskTab.id)
  assert.equal(second.state().activeTabId, taskTab.id)
  await second.destroy()
})

test('an unexpectedly destroyed ordinary entry remains a logical stale tab and reconciles on restart', async () => {
  runtimeHome()
  const first = new WorkstationBrowserRuntime()
  const tab = first.ensure().tabs[0]
  const contents = first.getWebContents(tab.id) as unknown as InstanceType<typeof electron.FakeWebContents>
  assert.ok(contents)
  await contents.loadURL('https://stale.example.test/recover')
  contents.setTitle('Recoverable Page')
  assert.equal(first.state().tabs.find(candidate => candidate.id === tab.id)?.title, 'Recoverable Page')
  contents.close()
  assert.equal(first.state().tabs.length, 0)
  await first.destroy()

  const second = new WorkstationBrowserRuntime()
  const restored = second.ensure()
  assert.equal(restored.tabs.length, 1)
  assert.equal(restored.tabs[0].id, tab.id)
  assert.equal(restored.tabs[0].url, 'https://stale.example.test/recover')
  assert.equal(restored.tabs[0].title, 'New Tab')
  await second.destroy()
})

test('an unknown BrowserSessionState version starts safely without importing stale legacy state', async () => {
  const home = runtimeHome()
  fs.mkdirSync(path.dirname(workstationBrowserSessionStatePath()), { recursive: true })
  fs.writeFileSync(workstationBrowserSessionStatePath(), JSON.stringify({ version: 999, tabs: [] }))
  fs.writeFileSync(
    path.join(home, 'Runtime', 'browser-tasks.json'),
    JSON.stringify({ version: 1, browserTaskCounter: 1, tasks: [{ taskId: 'stale-task' }] })
  )

  const runtime = new WorkstationBrowserRuntime()
  const state = runtime.ensure()
  assert.equal(state.tabs.length, 1)
  assert.equal(state.tabs[0].url, 'about:blank')
  assert.deepEqual(runtime.listTasks(), [])
  await runtime.destroy()
  assert.equal(JSON.parse(fs.readFileSync(workstationBrowserSessionStatePath(), 'utf-8')).version, 999)
})

test('C1 create interruption recovers new BrowserTask metadata with the previous sanitized tab projection', async () => {
  const home = runtimeHome()
  const fault = interruptibleSessionPersistence(home)
  const interrupted = new WorkstationBrowserRuntime(fault.persistence)
  const ordinary = interrupted.ensure().tabs[0]

  const ordinaryContents = interrupted.getWebContents(ordinary.id) as unknown as InstanceType<
    typeof electron.FakeWebContents
  >

  await ordinaryContents.loadURL('https://ordinary.example.test/customers/482913?access_token=c1-query-secret')
  ordinaryContents.setTitle('Recovery code 482913')
  const before = persistedComposite()
  assert.equal(before.tabs[0]?.safeUrl, 'https://ordinary.example.test/customers/482913')
  assert.equal(before.tabs[0]?.safeTitle, null)

  fault.failAfterSuccessfulReplacements(1)
  interrupted.createTask({ taskId: 'task-c1' })

  // This is an expected composite intermediate: BrowserTask persistence has
  // advanced, while the sanitized tabs/active projection is still previous.
  const intermediate = persistedComposite()
  assert.deepEqual(intermediate.tabs, before.tabs)
  assert.deepEqual(
    intermediate.browserTasks.tasks.map(task => task.taskId),
    ['task-c1']
  )
  assert.equal(
    intermediate.tabs.some(tab => tab.browserTaskId === 'task-c1'),
    false
  )
  const serializedIntermediate = JSON.stringify(intermediate)
  assert.equal(serializedIntermediate.includes('c1-query-secret'), false)
  assert.equal(serializedIntermediate.includes('Recovery code 482913'), false)

  const recovered = new WorkstationBrowserRuntime()
  const beforeShow = recovered.ensure()
  assert.equal(recovered.listTasks().filter(task => task.taskId === 'task-c1').length, 1)
  assert.equal(recovered.listTasks()[0]?.status, 'parked')
  assert.equal(
    beforeShow.tabs.some(tab => tab.ownerTaskId === 'task-c1'),
    false
  )

  recovered.showTask('task-c1', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  const owned = recovered.state().tabs.filter(tab => tab.ownerTaskId === 'task-c1')
  assert.equal(owned.length, 1)
  assert.equal(recovered.state().activeTabId, owned[0].id)

  const canonical = persistedComposite()
  assert.deepEqual(
    canonical.browserTasks.tasks.map(task => task.taskId),
    ['task-c1']
  )
  assert.equal(canonical.tabs.filter(tab => tab.browserTaskId === 'task-c1').length, 1)
  assert.deepEqual(
    canonical.tabs.map(tab => tab.id),
    recovered.state().tabs.map(tab => tab.id)
  )
  assert.equal(canonical.activeTabId, owned[0].id)
})

test('C2 show interruption recovers one lazy task from new lifecycle metadata plus the previous projection', async () => {
  const home = runtimeHome()
  const seed = new WorkstationBrowserRuntime()
  seed.createTask({ taskId: 'task-c2' })
  const seededTab = seed.state().tabs.find(tab => tab.ownerTaskId === 'task-c2')
  assert.ok(seededTab)
  const seededContents = seed.getWebContents(seededTab.id) as unknown as InstanceType<typeof electron.FakeWebContents>
  await seededContents.loadURL('https://task.example.test/customers/482913?code=c2-oauth-secret')
  seededContents.setTitle('Magic login code 482913')
  await seed.destroy()

  const fault = interruptibleSessionPersistence(home)
  const interrupted = new WorkstationBrowserRuntime(fault.persistence)
  const lazyState = interrupted.ensure()
  assert.equal(
    lazyState.tabs.some(tab => tab.ownerTaskId === 'task-c2'),
    false
  )
  const previousProjection = persistedComposite()
  const pendingTaskTab = previousProjection.tabs.find(tab => tab.browserTaskId === 'task-c2')
  assert.ok(pendingTaskTab)
  assert.equal(pendingTaskTab.safeUrl, 'https://task.example.test/customers/482913')
  assert.equal(pendingTaskTab.safeTitle, null)

  fault.failAfterSuccessfulReplacements(2)
  interrupted.showTask('task-c2', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })

  // Lifecycle recovery/status writes are newer, but tabs/active are still the
  // previous sanitized lazy projection until the final runtime replacement.
  const intermediate = persistedComposite()
  assert.deepEqual(intermediate.tabs, previousProjection.tabs)
  assert.equal(intermediate.activeTabId, previousProjection.activeTabId)
  assert.equal(intermediate.browserTasks.tasks[0]?.taskId, 'task-c2')
  assert.equal(intermediate.browserTasks.tasks[0]?.status, 'visible')
  assert.equal(intermediate.browserTasks.tasks[0]?.recoveryState, 'recreated')
  const serializedIntermediate = JSON.stringify(intermediate)
  assert.equal(serializedIntermediate.includes('c2-oauth-secret'), false)
  assert.equal(serializedIntermediate.includes('Magic login code 482913'), false)

  const recovered = new WorkstationBrowserRuntime()
  const beforeShow = recovered.ensure()
  assert.equal(recovered.listTasks().filter(task => task.taskId === 'task-c2').length, 1)
  assert.equal(recovered.listTasks()[0]?.status, 'parked')
  assert.equal(
    beforeShow.tabs.some(tab => tab.ownerTaskId === 'task-c2'),
    false
  )

  recovered.showTask('task-c2', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  recovered.showTask('task-c2', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  const owned = recovered.state().tabs.filter(tab => tab.ownerTaskId === 'task-c2')
  assert.equal(owned.length, 1)
  assert.equal(owned[0].id, pendingTaskTab.id)
  assert.equal(owned[0].url, 'https://task.example.test/customers/482913')
  assert.equal(owned[0].title, 'New Tab')
  assert.equal(recovered.state().activeTabId, pendingTaskTab.id)
  assert.equal(recovered.state().tabs[0]?.id, pendingTaskTab.id)

  const canonical = persistedComposite()
  assert.equal(canonical.browserTasks.tasks.filter(task => task.taskId === 'task-c2').length, 1)
  assert.equal(canonical.tabs.filter(tab => tab.browserTaskId === 'task-c2').length, 1)
  assert.deepEqual(
    canonical.tabs.map(tab => tab.id),
    recovered.state().tabs.map(tab => tab.id)
  )
  assert.equal(canonical.activeTabId, pendingTaskTab.id)
  assert.equal(JSON.stringify(canonical).includes('Magic login code 482913'), false)
})

test('C3 destroy interruption drops the removed task and orphan projection deterministically', async () => {
  const home = runtimeHome()
  const fault = interruptibleSessionPersistence(home)
  const interrupted = new WorkstationBrowserRuntime(fault.persistence)
  const ordinary = interrupted.ensure().tabs[0]
  interrupted.createTask({ taskId: 'task-c3' })
  const taskTab = interrupted.state().tabs.find(tab => tab.ownerTaskId === 'task-c3')
  assert.ok(taskTab)
  const taskContents = interrupted.getWebContents(taskTab.id) as unknown as InstanceType<
    typeof electron.FakeWebContents
  >
  interrupted.activateTab(taskTab.id)

  fault.failAfterSuccessfulReplacements(1)
  assert.equal(interrupted.destroyTask('task-c3'), true)
  assert.equal(taskContents.isDestroyed(), true)

  const intermediate = persistedComposite()
  assert.deepEqual(intermediate.browserTasks.tasks, [])
  assert.equal(
    intermediate.tabs.some(tab => tab.browserTaskId === 'task-c3'),
    false
  )
  assert.equal(intermediate.activeTabId, null)

  const recovered = new WorkstationBrowserRuntime()
  const state = recovered.ensure()
  assert.deepEqual(recovered.listTasks(), [])
  assert.equal(
    state.tabs.some(tab => tab.ownerTaskId === 'task-c3'),
    false
  )
  assert.equal(state.tabs.filter(tab => tab.id === ordinary.id).length, 1)
  assert.equal(state.activeTabId, ordinary.id)
  assert.throws(
    () => recovered.showTask('task-c3', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 }),
    /BrowserTask not found/
  )

  const canonical = persistedComposite()
  assert.deepEqual(canonical.browserTasks.tasks, [])
  assert.equal(
    canonical.tabs.some(tab => tab.browserTaskId === 'task-c3'),
    false
  )
  assert.deepEqual(
    canonical.tabs.map(tab => tab.id),
    state.tabs.map(tab => tab.id)
  )
  assert.equal(canonical.activeTabId, ordinary.id)
})

test('controller loads, verifies and removes only an extension from the Workstation store', async () => {
  const home = runtimeHome()
  process.env.HERMES_HOME = home
  const extensionId = 'cjpalhdlnbpafiamejdnhcphjbkeiagm'
  const extensionPath = path.join(home, 'workstation', 'extensions', extensionId)
  fs.mkdirSync(extensionPath, { recursive: true })
  fs.writeFileSync(path.join(extensionPath, 'manifest.json'), JSON.stringify({ name: 'Fixture', version: '1.0.0' }))
  const runtime = new WorkstationBrowserRuntime()
  runtime.ensure()

  const loaded = await (runtime as any).executeControlRequest({
    action: 'browser_extension_load', arguments: { extension_id: extensionId, path: extensionPath }
  })
  assert.equal(loaded.loaded, true)
  assert.equal(loaded.extension_id, extensionId)

  const verified = await (runtime as any).executeControlRequest({
    action: 'browser_extension_verify', arguments: { extension_id: extensionId }
  })
  assert.equal(verified.loaded, true)

  const removed = await (runtime as any).executeControlRequest({
    action: 'browser_extension_remove', arguments: { extension_id: extensionId }
  })
  assert.equal(removed.loaded, false)
  await runtime.destroy()
})

test('ownerTaskId is idempotent at the Chromium tab primitive', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()

  runtime.createTab('about:blank', false, 'task-a')
  const first = runtime.state().tabs.find(tab => tab.ownerTaskId === 'task-a')
  assert.ok(first)

  runtime.createTab('about:blank', false, 'task-a')
  const owned = runtime.state().tabs.filter(tab => tab.ownerTaskId === 'task-a')
  assert.equal(owned.length, 1)
  assert.equal(owned[0].id, first.id)

  await runtime.destroy()
})

test('runtime BrowserTask hide/park/show preserves one WebContents and its URL', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const window = hostWindow()
  const bounds = { x: 20, y: 80, width: 1000, height: 700 }

  runtime.createTask({ taskId: 'task-a' })
  const tab = runtime.state().tabs.find(candidate => candidate.ownerTaskId === 'task-a')
  assert.ok(tab)
  const contents = runtime.getWebContents(tab.id)
  assert.ok(contents)
  await contents.loadURL('https://example.test/current')

  const hidden = runtime.hideTask('task-a')
  assert.equal(hidden.status, 'hidden')
  assert.equal(contents.isDestroyed(), false)

  const shown = runtime.showTask('task-a', window as never, bounds)
  assert.equal(shown.status, 'visible')
  assert.equal(runtime.getWebContents(tab.id), contents)
  assert.equal(contents.getURL(), 'https://example.test/current')

  const parked = runtime.parkTask('task-a')
  assert.equal(parked.status, 'parked')
  runtime.showTask('task-a', window as never, bounds)
  assert.equal(runtime.getWebContents(tab.id), contents)
  assert.equal(contents.getURL(), 'https://example.test/current')

  assert.equal(runtime.state().tabs.filter(candidate => candidate.ownerTaskId === 'task-a').length, 1)
  assert.equal(runtime.destroyTask('task-a'), true)
  assert.equal(contents.isDestroyed(), true)
  assert.equal(runtime.listTasks().length, 0)

  await runtime.destroy()
})

test('controller actions preserve a visible task through Take Control and Release Control', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const window = hostWindow()
  const bounds = { x: 20, y: 80, width: 1000, height: 700 }

  runtime.createTask({ taskId: 'task-a' })
  const tab = runtime.state().tabs.find(candidate => candidate.ownerTaskId === 'task-a')
  assert.ok(tab)
  const contents = runtime.getWebContents(tab.id)
  assert.ok(contents)
  runtime.showTask('task-a', window as never, bounds)

  const executeControlRequest = (
    runtime as unknown as {
      executeControlRequest(request: Record<string, unknown>): Promise<Record<string, unknown>>
    }
  ).executeControlRequest.bind(runtime)

  runtime.takeControl()
  await executeControlRequest({ action: 'browser_snapshot', task_id: 'task-a', arguments: {} })
  assert.equal(runtime.state().controlOwner, 'human')
  assert.equal(runtime.state().attached, true)
  assert.equal(runtime.listTasks()[0]?.status, 'visible')
  assert.equal(runtime.getWebContents(tab.id), contents)

  runtime.releaseControl()
  await executeControlRequest({
    action: 'browser_navigate',
    task_id: 'task-a',
    arguments: { url: 'https://example.test/agent' }
  })
  assert.equal(runtime.state().controlOwner, 'agent')
  assert.equal(runtime.state().attached, true)
  assert.equal(runtime.listTasks()[0]?.status, 'visible')
  assert.equal(runtime.getWebContents(tab.id), contents)

  await runtime.destroy()
})

test('runtime replaces one crashed BrowserTask page and preserves logical ownership', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()

  runtime.createTask({ taskId: 'task-a' })
  const firstTab = runtime.state().tabs.find(candidate => candidate.ownerTaskId === 'task-a')
  assert.ok(firstTab)
  const crashed = runtime.getWebContents(firstTab.id) as unknown as InstanceType<typeof electron.FakeWebContents>
  assert.ok(crashed)
  crashed.simulateRenderProcessGone()

  const recovered = runtime.createTask({ taskId: 'task-a' })
  const owned = runtime.state().tabs.filter(candidate => candidate.ownerTaskId === 'task-a')
  assert.equal(recovered.recoveryState, 'recreated')
  assert.equal(owned.length, 1)
  assert.notEqual(owned[0].id, firstTab.id)
  assert.notEqual(runtime.getWebContents(owned[0].id), crashed)
  assert.equal(crashed.isDestroyed(), true)

  await runtime.destroy()
})

test('destroying the active BrowserTask activates the remaining isolated task page', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()

  runtime.createTask({ taskId: 'task-a' })
  runtime.createTask({ taskId: 'task-b' })
  const taskA = runtime.state().tabs.find(candidate => candidate.ownerTaskId === 'task-a')
  const taskB = runtime.state().tabs.find(candidate => candidate.ownerTaskId === 'task-b')
  assert.ok(taskA)
  assert.ok(taskB)
  runtime.activateTab(taskA.id)

  assert.equal(runtime.destroyTask('task-a'), true)
  const state = runtime.state()
  assert.equal(
    state.tabs.some(candidate => candidate.ownerTaskId === 'task-a'),
    false
  )
  assert.equal(state.tabs.filter(candidate => candidate.ownerTaskId === 'task-b').length, 1)
  assert.equal(state.activeTabId, taskB.id)
  assert.equal(
    runtime
      .listTasks()
      .map(task => task.taskId)
      .join(','),
    'task-b'
  )

  await runtime.destroy()
})

test('runtime restart keeps task metadata parked and recreates a page only on show', async () => {
  runtimeHome()
  const first = new WorkstationBrowserRuntime()
  first.createTask({ taskId: 'task-persisted', panelHost: 'panel-1' })
  first.showTask('task-persisted', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  await first.destroy()

  const second = new WorkstationBrowserRuntime()
  second.ensure()
  const restored = second.listTasks()
  assert.equal(restored.length, 1)
  assert.equal(restored[0].taskId, 'task-persisted')
  assert.equal(restored[0].status, 'parked')
  assert.equal(restored[0].recoveryState, 'restored')
  assert.equal(
    second.state().tabs.some(tab => tab.ownerTaskId === 'task-persisted'),
    false
  )

  const shown = second.showTask('task-persisted', hostWindow() as never, { x: 0, y: 0, width: 900, height: 600 })
  assert.equal(shown.recoveryState, 'recreated')
  assert.equal(second.state().tabs.filter(tab => tab.ownerTaskId === 'task-persisted').length, 1)

  await second.destroy()
})

test('getStandardChromeUserAgent outputs modern Chrome UA without Electron or hermes traces', () => {
  const ua = getStandardChromeUserAgent()
  assert.match(ua, /^Mozilla\/5\.0 /)
  assert.match(ua, /AppleWebKit\/537\.36/)
  assert.match(ua, /Chrome\/133\.0\.0\.0 Safari\/537\.36/)
  assert.equal(ua.includes('Electron'), false)
  assert.equal(ua.includes('hermes'), false)
})

test('runtime setVisible removes view from window contentView when false and restores on true', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const window = hostWindow()
  runtime.attach(window as never, { x: 0, y: 0, width: 900, height: 600 }, 'chat')

  assert.equal(window.contentView.children.length, 1)

  // When an overlay/menu is open: setVisible(false) removes child view from compositor
  runtime.setVisible(false)
  assert.equal(window.contentView.children.length, 0)

  // When overlay/menu closes: setVisible(true) restores child view
  runtime.setVisible(true)
  assert.equal(window.contentView.children.length, 1)

  await runtime.destroy()
})

test('runtime clearError resets lastError', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  ;(runtime as unknown as { recordError: (err: unknown) => void }).recordError('stale_or_unknown_ref')

  assert.equal(runtime.state().lastError, 'stale_or_unknown_ref')
  runtime.clearError()
  assert.equal(runtime.state().lastError, null)

  await runtime.destroy()
})

test('concurrent multi-task isolation: background actions do not steal active tab or hijack existing task tabs', async () => {
  runtimeHome()
  const runtime = new WorkstationBrowserRuntime()
  const window = hostWindow()
  const bounds = { x: 0, y: 0, width: 900, height: 600 }

  const executeControlRequest = (
    runtime as unknown as {
      executeControlRequest(request: Record<string, unknown>): Promise<Record<string, unknown>>
    }
  ).executeControlRequest.bind(runtime)

  // 1. Session 1 navigates to Site A
  runtime.attach(window as never, bounds, 'chat', 'task-sess-1')
  await executeControlRequest({
    action: 'browser_navigate',
    task_id: 'task-sess-1',
    arguments: { url: 'https://session1.test' }
  })

  const tab1 = runtime.state().tabs.find(t => t.ownerTaskId === 'task-sess-1')
  assert.ok(tab1, 'Tab 1 must exist for session 1')
  assert.equal(tab1.url, 'https://session1.test/')
  assert.equal(runtime.state().activeTabId, tab1.id)

  // 2. Session 2 navigates to Site B
  runtime.attach(window as never, bounds, 'chat', 'task-sess-2')
  await executeControlRequest({
    action: 'browser_navigate',
    task_id: 'task-sess-2',
    arguments: { url: 'https://session2.test' }
  })

  const tab2 = runtime.state().tabs.find(t => t.ownerTaskId === 'task-sess-2')
  assert.ok(tab2, 'Tab 2 must exist for session 2')
  assert.equal(tab2.url, 'https://session2.test/')
  assert.notEqual(tab1.id, tab2.id, 'Session 1 and Session 2 must own separate tabs')
  assert.equal(runtime.state().activeTabId, tab2.id, 'Session 2 should be the active tab')

  // Verify Tab 1 was not overwritten
  const tab1After = runtime.state().tabs.find(t => t.id === tab1.id)
  assert.equal(tab1After?.ownerTaskId, 'task-sess-1', 'Tab 1 must not be hijacked by Session 2')
  assert.equal(tab1After?.url, 'https://session1.test/')

  // 3. Session 1 performs background actions (snapshot, click, scroll) while Session 2 is active on-screen
  await executeControlRequest({
    action: 'browser_snapshot',
    task_id: 'task-sess-1',
    arguments: {}
  })

  // Active tab must still be Tab 2 (Session 2 view must NOT be stolen/swapped by Session 1 background work)
  assert.equal(runtime.state().activeTabId, tab2.id, 'Session 1 background action must not steal active tab from Session 2')

  // 4. Switching chat back to Session 1 via preferredTaskId activates Tab 1
  runtime.attach(window as never, bounds, 'chat', 'task-sess-1')
  assert.equal(runtime.state().activeTabId, tab1.id, 'Attaching to Session 1 should activate Tab 1')

  // Switching back to Session 2 activates Tab 2
  runtime.attach(window as never, bounds, 'chat', 'task-sess-2')
  assert.equal(runtime.state().activeTabId, tab2.id, 'Attaching to Session 2 should activate Tab 2')

  await runtime.destroy()
})
