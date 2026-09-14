/**
 * Integrated Desktop/Browser load evidence.
 *
 * This test intentionally runs the real dev Electron shell with the real
 * `hermes serve` backend from the mock-backend fixture. The inference server
 * is fake, but Chromium, BrowserTask lifecycle, the authenticated controller,
 * resource projection, and the renderer IPC bridge are real.
 *
 * `headless: true` is mandatory here: the test must never reveal a desktop
 * window while exercising native WebContentsView instances.
 */

import * as http from 'node:http'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from './test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

interface ControllerFile {
  url: string
  token: string
  runtime: string
}

interface ControllerResponse {
  success: boolean
  schema_version?: number
  result?: Record<string, unknown>
  runtime?: string
  task_id?: string | null
  generated_at?: string
  resources?: Array<Record<string, unknown>>
  events?: Array<Record<string, unknown>>
  error?: string
}

interface BrowserTaskRecord {
  taskId: string
  status?: string
  parked?: boolean
}

interface WorkstationBridge {
  attach: (bounds: Record<string, number>, host?: string) => Promise<Record<string, unknown>>
  setBounds: (bounds: Record<string, number>, expectedHost?: string) => Promise<Record<string, unknown>>
  transferViewport: (targetHost: string, bounds: Record<string, number>) => Promise<Record<string, unknown>>
  resources: () => Promise<{ runtime: string; resources: Array<Record<string, unknown>> }>
  events: (
    taskId?: string | null,
    limit?: number
  ) => Promise<{ runtime: string; task_id: string | null; events: Array<Record<string, unknown>> }>
  listTasks: () => Promise<BrowserTaskRecord[]>
  showTask: (taskId: string, bounds: Record<string, number>, host?: string) => Promise<BrowserTaskRecord>
  hideTask: (taskId: string) => Promise<unknown>
  parkTask: (taskId: string) => Promise<unknown>
  destroyTask: (taskId: string) => Promise<unknown>
}

interface DesktopBridgeWindow {
  hermesDesktop: { workstationBrowser: WorkstationBridge }
}

interface NativeViewportBounds {
  x: number
  y: number
  width: number
  height: number
}

interface NativeViewportChild {
  bounds: NativeViewportBounds
  webContentsId: number | null
  url: string
}

interface NativeViewportSnapshot {
  contentBounds: NativeViewportBounds
  children: NativeViewportChild[]
  maximized: boolean
}

interface SustainedLoadEvidence {
  accepted: true
  task_count: number
  requested_rounds: number
  completed_rounds: number
  requested_duration_ms: number
  observed_duration_ms: number
  chat_turns: number
}

function readBoundedInteger(name: string, fallback: number, minimum: number, maximum: number): number {
  const raw = process.env[name]?.trim()

  if (!raw) {
    return fallback
  }

  const value = Number(raw)

  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`)
  }

  return value
}

const TASK_COUNT = 4
const SUSTAINED_TASK_COUNT = readBoundedInteger('HERMES_DESKTOP_E2E_SUSTAINED_TASKS', 8, 1, 32)
const SUSTAINED_ROUNDS = readBoundedInteger('HERMES_DESKTOP_E2E_SUSTAINED_ROUNDS', 3, 1, 300)
const SUSTAINED_DURATION_MS = readBoundedInteger('HERMES_DESKTOP_E2E_SUSTAINED_DURATION_MS', 0, 0, 300_000)
const SUSTAINED_CHAT_TURNS = readBoundedInteger('HERMES_DESKTOP_E2E_SUSTAINED_CHAT_TURNS', 3, 1, 32)

let fixture: MockBackendFixture | null = null
let pageServer: http.Server | null = null
let pageBaseUrl = ''
let controller: ControllerFile | null = null
let sustainedLoadEvidence: SustainedLoadEvidence | null = null

async function startDeterministicPageServer(): Promise<void> {
  pageServer = http.createServer((request, response) => {
    const taskId = decodeURIComponent((request.url ?? '/').replace(/^\//, '')) || 'unknown'

    response.writeHead(200, {
      'content-type': 'text/html; charset=utf-8',
      'cache-control': 'no-store'
    })
    response.end(
      `<!doctype html><html><head><title>H013 ${taskId}</title></head><body><main data-task="${taskId}">H013 ${taskId}</main></body></html>`
    )
  })

  await new Promise<void>((resolve, reject) => {
    pageServer!.once('error', reject)
    pageServer!.listen(0, '127.0.0.1', () => resolve())
  })

  const address = pageServer.address()

  if (!address || typeof address === 'string') {
    throw new Error('H013 deterministic page server did not bind a TCP port')
  }

  pageBaseUrl = `http://127.0.0.1:${address.port}`
}

async function readControllerFile(filePath: string): Promise<ControllerFile> {
  const raw = JSON.parse(await fs.promises.readFile(filePath, 'utf8')) as Partial<ControllerFile>

  if (!raw.url || !raw.token || raw.runtime !== 'electron-chromium') {
    throw new Error('H013 controller file is incomplete')
  }

  return { url: raw.url, token: raw.token, runtime: raw.runtime }
}

async function controllerRequest(endpoint: string, init: RequestInit = {}): Promise<ControllerResponse> {
  if (!controller) {
    throw new Error('H013 controller is not ready')
  }

  const response = await fetch(`${controller.url}${endpoint}`, {
    ...init,
    headers: {
      authorization: `Bearer ${controller.token}`,
      ...(init.headers ?? {})
    }
  })
  const body = (await response.json()) as ControllerResponse

  if (!response.ok || body.success !== true) {
    throw new Error(`H013 controller request failed (${response.status}): ${body.error ?? 'unknown error'}`)
  }

  return body
}

function comparableResourceSnapshot(snapshot: { schema_version?: unknown; runtime?: unknown; resources?: unknown[] }): {
  schema_version: unknown
  runtime: unknown
  resources: unknown[]
} {
  const resources = Array.isArray(snapshot.resources)
    ? snapshot.resources.map(resource => {
        if (!resource || typeof resource !== 'object') {
          return resource
        }

        const comparable = { ...(resource as Record<string, unknown>) }

        if (comparable.resource_type === 'browser') {
          comparable.updated_at = '<generated>'
        }

        return comparable
      })
    : []

  return {
    schema_version: snapshot.schema_version,
    runtime: snapshot.runtime,
    resources
  }
}

async function browserAction(taskId: string, sessionId: string, url: string): Promise<ControllerResponse> {
  return controllerRequest('/v1/action', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      action: 'browser_navigate',
      arguments: { url },
      task_id: taskId,
      session_id: sessionId
    })
  })
}

async function nativeViewportSnapshot(app: MockBackendFixture['app']): Promise<NativeViewportSnapshot | null> {
  return app.evaluate(({ BrowserWindow }): NativeViewportSnapshot | null => {
    const window = BrowserWindow.getAllWindows()[0]

    if (!window || window.isDestroyed()) {
      return null
    }

    const children = window?.contentView.children ?? []

    return {
      contentBounds: window.getContentBounds(),
      maximized: window.isMaximized(),
      children: children.flatMap((child: unknown) => {
        const view = child as unknown as {
          getBounds?: () => NativeViewportBounds
          webContents?: { id?: number; getURL?: () => string }
        }

        if (typeof view.getBounds !== 'function') {
          return []
        }

        return [
          {
            bounds: view.getBounds(),
            webContentsId: typeof view.webContents?.id === 'number' ? view.webContents.id : null,
            url: typeof view.webContents?.getURL === 'function' ? view.webContents.getURL() : ''
          }
        ]
      })
    }
  })
}

function activeViewportChild(snapshot: NativeViewportSnapshot | null): NativeViewportChild {
  const content = snapshot?.contentBounds
  const activeChildren = (snapshot?.children ?? []).filter(child => {
    if (!content) {
      return false
    }

    return (
      child.bounds.x >= 0 &&
      child.bounds.y >= 0 &&
      child.bounds.x + child.bounds.width <= content.width &&
      child.bounds.y + child.bounds.height <= content.height
    )
  })
  const viewport = activeChildren[0]

  // Parked BrowserTasks may remain attached at the compositor edge to keep
  // Chromium warm. Only one native view may occupy the actual content area.
  expect(activeChildren, `native viewport snapshot: ${JSON.stringify(snapshot)}`).toHaveLength(1)

  return activeChildren[0] as NativeViewportChild
}

function expectViewportWithinContent(snapshot: NativeViewportSnapshot | null): void {
  const viewport = activeViewportChild(snapshot)
  const content = snapshot?.contentBounds

  expect(viewport?.bounds.x).toBeGreaterThanOrEqual(0)
  expect(viewport?.bounds.y).toBeGreaterThanOrEqual(0)
  expect(viewport?.bounds.width).toBeGreaterThan(0)
  expect(viewport?.bounds.height).toBeGreaterThan(0)
  expect((viewport?.bounds.x ?? 0) + (viewport?.bounds.width ?? 0)).toBeLessThanOrEqual(content?.width ?? 0)
  expect((viewport?.bounds.y ?? 0) + (viewport?.bounds.height ?? 0)).toBeLessThanOrEqual(content?.height ?? 0)
}

test.beforeAll(async () => {
  await startDeterministicPageServer()
  fixture = await setupMockBackend({ headless: true })
  await waitForAppReady(fixture, 120_000, false)

  const controlPath = path.join(fixture.sandbox.root, 'workstation', 'Runtime', 'browser-control.json')

  await expect.poll(() => fs.existsSync(controlPath), { timeout: 30_000 }).toBe(true)
  controller = await readControllerFile(controlPath)

  expect(controller.runtime).toBe('electron-chromium')
  const health = await controllerRequest('/health')
  expect(health.runtime).toBe('electron-chromium')
})

test.afterAll(async () => {
  const reportPath = process.env.HERMES_DESKTOP_E2E_REPORT?.trim()

  if (reportPath && sustainedLoadEvidence) {
    await fs.promises.mkdir(path.dirname(reportPath), { recursive: true })
    await fs.promises.writeFile(reportPath, `${JSON.stringify(sustainedLoadEvidence, null, 2)}\n`, 'utf8')
  }

  await fixture?.cleanup()
  fixture = null
  controller = null

  await new Promise<void>(resolve => {
    if (!pageServer) {
      resolve()
      return
    }

    pageServer.close(() => resolve())
  })
  pageServer = null
})

test('keeps four native Browser tasks aligned across controller and IPC while headless', async () => {
  if (!fixture || !controller) {
    throw new Error('H013 fixture was not initialized')
  }

  const tasks = Array.from({ length: TASK_COUNT }, (_, index) => ({
    taskId: `h013-task-${index}`,
    sessionId: `h013-session-${index}`,
    url: `${pageBaseUrl}/h013-task-${index}`
  }))

  const navigations = await Promise.all(tasks.map(task => browserAction(task.taskId, task.sessionId, task.url)))

  expect(navigations).toHaveLength(TASK_COUNT)
  for (const [index, navigation] of navigations.entries()) {
    expect(navigation.result?.url).toBe(tasks[index].url)
    expect(navigation.result?.title).toBe(`H013 ${tasks[index].taskId}`)
  }

  for (const task of tasks) {
    const snapshot = await controllerRequest('/v1/action', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        action: 'browser_snapshot',
        arguments: { full: false },
        task_id: task.taskId,
        session_id: task.sessionId
      })
    })

    expect(snapshot.result?.url).toBe(task.url)
    expect(snapshot.result?.title).toBe(`H013 ${task.taskId}`)
  }

  const controllerResources = await controllerRequest('/resources')
  const taskResources = (controllerResources.resources ?? []).filter(
    resource => resource.resource_type === 'browser_task'
  )
  const taskResourceIds = new Set(taskResources.map(resource => resource.task_id))

  expect(controllerResources.runtime).toBe('electron-chromium')
  expect(taskResources).toHaveLength(TASK_COUNT)
  expect(taskResourceIds).toEqual(new Set(tasks.map(task => task.taskId)))

  const controllerEvents = await controllerRequest('/events?limit=100')

  for (const resource of taskResources) {
    expect(resource.session_id).toBe(`h013-session-${String(resource.task_id).replace('h013-task-', '')}`)
    expect((resource.state as { tab_id?: string }).tab_id).toEqual(expect.any(String))
    expect((resource.state as { evidence?: string[] }).evidence).toEqual(
      expect.arrayContaining(['browser://controller'])
    )
  }

  const ipcSnapshot = await fixture.page.evaluate(async () => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop?.workstationBrowser

    if (!api) {
      throw new Error('H013 renderer IPC bridge is unavailable')
    }

    return {
      resources: await api.resources(),
      events: await api.events(null, 100),
      tasks: await api.listTasks()
    }
  })

  expect(ipcSnapshot.resources.runtime).toBe(controllerResources.runtime)
  expect(comparableResourceSnapshot(ipcSnapshot.resources)).toEqual(comparableResourceSnapshot(controllerResources))
  const resourceIdentity = (snapshot: { resources?: Array<Record<string, unknown>> }) =>
    (snapshot.resources ?? []).map(resource => [
      resource.resource_type,
      resource.resource_id,
      resource.task_id,
      resource.session_id
    ])

  expect(resourceIdentity(ipcSnapshot.resources)).toEqual(resourceIdentity(controllerResources))
  expect(ipcSnapshot.events.runtime).toBe('electron-chromium')
  expect(ipcSnapshot.events.task_id).toBe(controllerEvents.task_id)
  expect(ipcSnapshot.events.events).toEqual(controllerEvents.events)
  expect(ipcSnapshot.tasks).toHaveLength(TASK_COUNT)
  expect(ipcSnapshot.events.events).toEqual(expect.any(Array))

  const hubBounds = { x: 12, y: 18, width: 720, height: 520 }
  const chatBounds = { x: 300, y: 22, width: 500, height: 480 }
  const attached = await fixture.page.evaluate(async bounds => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
    return api.attach(bounds, 'hub')
  }, hubBounds)

  expect(attached.viewportHost).toBe('hub')
  const initialNativeViewport = await nativeViewportSnapshot(fixture.app)
  const initialNativeChild = activeViewportChild(initialNativeViewport)
  expect(initialNativeChild.bounds).toEqual(hubBounds)
  const initialNativeChildId = initialNativeChild.webContentsId

  const stale = await fixture.page.evaluate(async bounds => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
    return api.setBounds(bounds, 'chat')
  }, chatBounds)

  expect(stale.viewportHost).toBe('hub')
  const staleNativeViewport = await nativeViewportSnapshot(fixture.app)
  const staleNativeChild = activeViewportChild(staleNativeViewport)
  expect(staleNativeChild.webContentsId).toBe(initialNativeChildId)
  expect(staleNativeChild.bounds).toEqual(initialNativeChild.bounds)

  const resized = await fixture.page.evaluate(async bounds => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
    return api.setBounds(bounds, 'hub')
  }, chatBounds)

  expect(resized.viewportHost).toBe('hub')
  const resizedNativeViewport = await nativeViewportSnapshot(fixture.app)
  const resizedNativeChild = activeViewportChild(resizedNativeViewport)
  expect(resizedNativeChild.webContentsId).toBe(initialNativeChildId)
  expect(resizedNativeChild.bounds).toEqual(chatBounds)

  const transferred = await fixture.page.evaluate(async bounds => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
    return api.transferViewport('chat', bounds)
  }, chatBounds)

  expect(transferred.viewportHost).toBe('chat')
  const transferredNativeViewport = await nativeViewportSnapshot(fixture.app)
  const transferredNativeChild = activeViewportChild(transferredNativeViewport)
  expect(transferredNativeChild.webContentsId).toBe(initialNativeChildId)
  expect(transferredNativeChild.bounds).toEqual({
    x: expect.any(Number),
    y: expect.any(Number),
    width: expect.any(Number),
    height: expect.any(Number)
  })

  const restoredWindowViewport = await nativeViewportSnapshot(fixture.app)
  expect(restoredWindowViewport?.maximized).toBe(false)
  expectViewportWithinContent(restoredWindowViewport)

  const maximized = await fixture.app.evaluate(({ BrowserWindow }): boolean => {
    const window = BrowserWindow.getAllWindows()[0]

    if (!window || window.isDestroyed()) {
      throw new Error('H013 BrowserWindow disappeared before maximize')
    }

    window.maximize()
    const maximized = window.isMaximized()
    // Windows may reveal a hidden BrowserWindow as a side effect of native
    // maximize. Immediately re-hide it so this E2E never paints a desktop
    // surface while still exercising the real maximize transition.
    window.hide()

    return maximized
  })

  expect(maximized).toBe(true)
  await fixture.page.waitForTimeout(250)

  const maximizedViewport = await nativeViewportSnapshot(fixture.app)
  expect(maximizedViewport?.maximized).toBe(true)
  expect(activeViewportChild(maximizedViewport).webContentsId).toBe(initialNativeChildId)
  expectViewportWithinContent(maximizedViewport)

  const restored = await fixture.app.evaluate(({ BrowserWindow }): boolean => {
    const window = BrowserWindow.getAllWindows()[0]

    if (!window || window.isDestroyed()) {
      throw new Error('H013 BrowserWindow disappeared before restore')
    }

    window.unmaximize()
    const restored = window.isMaximized()
    window.hide()

    return restored
  })

  expect(restored).toBe(false)
  await fixture.page.waitForTimeout(250)

  const afterRestoreViewport = await nativeViewportSnapshot(fixture.app)
  expect(afterRestoreViewport?.maximized).toBe(false)
  expect(activeViewportChild(afterRestoreViewport).webContentsId).toBe(initialNativeChildId)
  expectViewportWithinContent(afterRestoreViewport)

  const hiddenWindows = await fixture.app.evaluate(({ BrowserWindow }): boolean[] =>
    BrowserWindow.getAllWindows().map((window: { isVisible: () => boolean }) => window.isVisible())
  )

  expect(hiddenWindows.length).toBeGreaterThan(0)
  expect(hiddenWindows.every(visible => visible === false)).toBe(true)

  await fixture.page.evaluate(async taskId => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
    await api.hideTask(taskId)
    await api.parkTask(taskId)
  }, tasks[0].taskId)

  const parkedTasks = await fixture.page.evaluate(() =>
    (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser.listTasks()
  )
  const parkedTask = parkedTasks.find(task => task.taskId === 'h013-task-0')

  expect(parkedTask?.status).toBe('parked')
  expect(parkedTask?.parked).toBe(true)

  for (const task of tasks) {
    await fixture.page.evaluate(
      taskId => (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser.destroyTask(taskId),
      task.taskId
    )
  }

  const finalTasks = await fixture.page.evaluate(() =>
    (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser.listTasks()
  )
  expect(finalTasks).toHaveLength(0)
})

test('sustains concurrent BrowserTasks and real backend chat turns while headless', async () => {
  test.setTimeout(Math.max(90_000, SUSTAINED_DURATION_MS + 60_000))

  if (!fixture || !controller) {
    throw new Error('H013 fixture was not initialized')
  }

  const hubBounds = { x: 12, y: 18, width: 720, height: 520 }
  const chatBounds = { x: 300, y: 22, width: 500, height: 480 }
  const tasks = Array.from({ length: SUSTAINED_TASK_COUNT }, (_, index) => ({
    taskId: `h013-sustained-task-${index}`,
    sessionId: `h013-sustained-session-${index}`
  }))

  const attached = await fixture.page.evaluate(async bounds => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
    return api.attach(bounds, 'hub')
  }, hubBounds)

  expect(attached.viewportHost).toBe('hub')

  const startedAt = Date.now()
  let completedRounds = 0

  for (let round = 0; round < SUSTAINED_ROUNDS; round += 1) {
    if (round > 0 && SUSTAINED_DURATION_MS > 0 && Date.now() - startedAt >= SUSTAINED_DURATION_MS) {
      break
    }

    const navigations = await Promise.all(
      tasks.map(task =>
        browserAction(task.taskId, task.sessionId, `${pageBaseUrl}/sustained-round-${round}/${task.taskId}`)
      )
    )

    expect(navigations).toHaveLength(SUSTAINED_TASK_COUNT)
    for (const [index, navigation] of navigations.entries()) {
      expect(navigation.result?.url).toBe(`${pageBaseUrl}/sustained-round-${round}/${tasks[index].taskId}`)
      expect(navigation.result?.title).toBe(`H013 sustained-round-${round}/${tasks[index].taskId}`)
    }

    const snapshots = await Promise.all(
      tasks.map(task =>
        controllerRequest('/v1/action', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            action: 'browser_snapshot',
            arguments: { full: false },
            task_id: task.taskId,
            session_id: task.sessionId
          })
        })
      )
    )

    expect(snapshots).toHaveLength(SUSTAINED_TASK_COUNT)
    for (const [index, snapshot] of snapshots.entries()) {
      expect(snapshot.result?.url).toBe(`${pageBaseUrl}/sustained-round-${round}/${tasks[index].taskId}`)
      expect(snapshot.result?.title).toBe(`H013 sustained-round-${round}/${tasks[index].taskId}`)
    }

    const shownTask = tasks[round % tasks.length]
    const shown = await fixture.page.evaluate(
      async ({ bounds, taskId, host }) => {
        const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
        return api.showTask(taskId, bounds, host)
      },
      {
        bounds: round % 2 === 0 ? hubBounds : chatBounds,
        taskId: shownTask.taskId,
        host: round % 2 === 0 ? 'hub' : 'chat'
      }
    )

    expect(shown.taskId).toBe(shownTask.taskId)
    expect(shown.status).toBe('visible')
    expectViewportWithinContent(await nativeViewportSnapshot(fixture.app))
    completedRounds += 1
  }

  expect(completedRounds).toBeGreaterThan(0)
  const observedDurationMs = Date.now() - startedAt

  const controllerResources = await controllerRequest('/resources')
  const controllerEvents = await controllerRequest('/events?limit=200')
  const taskResources = (controllerResources.resources ?? []).filter(
    resource => resource.resource_type === 'browser_task'
  )
  expect(taskResources).toHaveLength(SUSTAINED_TASK_COUNT)
  expect(new Set(taskResources.map(resource => resource.task_id))).toEqual(new Set(tasks.map(task => task.taskId)))

  const chatTurns = Math.min(SUSTAINED_CHAT_TURNS, completedRounds)

  for (let turn = 0; turn < chatTurns; turn += 1) {
    const prompt = `H013 sustained backend turn ${turn}`
    const before = fixture.mock.receivedPrompts.length
    const composer = fixture.page.locator('[contenteditable="true"]').first()

    await composer.click()
    await composer.type(prompt)
    await fixture.page.keyboard.press('Enter')
    await expect.poll(() => fixture?.mock.receivedPrompts.length ?? 0, { timeout: 60_000 }).toBeGreaterThan(before)
    await expect
      .poll(() => fixture?.page.locator('body').textContent() ?? '', { timeout: 60_000 })
      .toContain('boot chain is working')
  }

  const ipcSnapshot = await fixture.page.evaluate(async () => {
    const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
    return {
      resources: await api.resources(),
      events: await api.events(null, 200),
      tasks: await api.listTasks()
    }
  })

  const ipcTaskResources = ipcSnapshot.resources.resources.filter(resource => resource.resource_type === 'browser_task')
  expect(comparableResourceSnapshot(ipcSnapshot.resources)).toEqual(comparableResourceSnapshot(controllerResources))
  expect(ipcSnapshot.resources.resources.length).toBeGreaterThan(SUSTAINED_TASK_COUNT)
  expect(ipcTaskResources).toHaveLength(SUSTAINED_TASK_COUNT)
  expect(new Set(ipcTaskResources.map(resource => resource.task_id))).toEqual(new Set(tasks.map(task => task.taskId)))
  expect(ipcSnapshot.events.task_id).toBe(controllerEvents.task_id)
  expect(ipcSnapshot.events.events).toEqual(controllerEvents.events)
  expect(ipcSnapshot.events.events).toEqual(expect.any(Array))
  expect(ipcSnapshot.tasks).toHaveLength(SUSTAINED_TASK_COUNT)

  for (const task of tasks) {
    await fixture.page.evaluate(async taskId => {
      const api = (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser
      await api.hideTask(taskId)
      await api.parkTask(taskId)
      await api.destroyTask(taskId)
    }, task.taskId)
  }

  const finalTasks = await fixture.page.evaluate(() =>
    (window as unknown as DesktopBridgeWindow).hermesDesktop.workstationBrowser.listTasks()
  )
  expect(finalTasks).toHaveLength(0)

  sustainedLoadEvidence = {
    accepted: true,
    task_count: SUSTAINED_TASK_COUNT,
    requested_rounds: SUSTAINED_ROUNDS,
    completed_rounds: completedRounds,
    requested_duration_ms: SUSTAINED_DURATION_MS,
    observed_duration_ms: observedDurationMs,
    chat_turns: chatTurns
  }
})
