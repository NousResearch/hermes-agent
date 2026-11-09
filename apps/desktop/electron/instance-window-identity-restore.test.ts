/** Executes the actual main.ts constructor/capture/restore bodies, with only
 * Electron/OS effects stubbed. No hand-written restore callback or renderer
 * storage mirror. This proves main wiring, not real Electron lifecycle/UI. */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'

import ts from 'typescript'
import { test } from 'vitest'

import * as profile from './desktop-profile'
import * as urls from './session-windows'
import * as state from './window-session-state'

const GEOMETRY = { width: 500, height: 700, isMaximized: false }
const source = ts.createSourceFile('main.ts', readFileSync(new URL('./main.ts', import.meta.url), 'utf8'), ts.ScriptTarget.Latest, true)
const names = ['createInstanceWindow', 'captureWindowSessionState', 'restoreWindowSessionsOnce', 'liveInstanceWindowIds']
const functions = source.statements.filter((n): n is ts.FunctionDeclaration => ts.isFunctionDeclaration(n) && names.includes(n.name?.text ?? ''))
assert.equal(functions.length, names.length, 'all production seams must still exist')

const code = ts.transpileModule(functions.map(n => n.getText(source)).join('\n'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS }
}).outputText

function harness() {
  let nextId = 0
  let fresh = 0
  const all: FakeWindow[] = []
  const loaded = new Map<number, string>()
  const routes = new Map<number, unknown>()

  class FakeWindow {
    webContents = { id: ++nextId }
    events: Record<string, () => void> = {}
    constructor(public options: unknown) { all.push(this) }
    on(name: string, cb: () => void) { this.events[name] = cb }
    isDestroyed() { return false }
    getNormalBounds() { return { x: 0, y: 0, width: 500, height: 700 } }
    isMaximized() { return false }
    static getAllWindows() { return all }
  }

  const noop = () => {}

  const context: any = {
    ...state, ...urls, ...profile,
    BrowserWindow: FakeWindow, instanceWindows: new Set(), instanceWindowIds: new Map(), instanceWindowProfilePins: new Map(),
    windowConnectionRoutes: { get: (id: number) => routes.get(id) },
    crypto: { randomUUID: () => `fresh-${++fresh}` }, console: { error: noop }, mainWindow: null,
    sessionWindowLaunchInfo: new Map(), appQuitting: false, quitInProgress: false,
    primaryProfileKey: () => 'default', validateDesktopProfileRoute: noop, getAppIconPath: noop,
    registerChatWindow: noop, minimizeToTray: { registerWindow: noop },
    recordWindowConnectionRoute: (wc: { id: number }, route: unknown) => routes.set(wc.id, route),
    translucencyBackedWindows: new Set(), IS_MAC: false, wireWindowReveal: noop,
    bindWindowChromeEvents: noop, sendWindowStateChanged: noop, streamThrottle: { register: noop },
    wireCommonWindowHandlers: noop, zoomWiringForWindowKind: noop, installWindowRendererLifecycle: noop,
    rendererTeardownInProgress: noop, RENDERER_RELOAD_WINDOW_MS: 100, RENDERER_RELOAD_MAX: 2, rendererReloadTimesRef: {},
    attachRendererConsoleCapture: noop, rememberLog: noop,
    loadWindowUrl: (w: FakeWindow, url: string) => loaded.set(w.webContents.id, url),
    DEV_SERVER: 'http://localhost:5173', WINDOW_MIN_WIDTH: 420, WINDOW_MIN_HEIGHT: 620, WINDOW_BUTTON_POSITION: {},
    getTitleBarOverlayOptions: noop, chatWindowSurfaceOptions: () => ({}), PRELOAD_PATH: 'unused',
    screen: { getAllDisplays: () => [] }, savedSessionWindowExists: () => true, createSessionWindow: noop,
    nextInstanceBounds: () => GEOMETRY
  }

  vm.createContext(context)
  vm.runInContext(code, context)

  return {
    context,
    urls: () => [...loaded.values()].map(u => new URL(u)),
    open: (options?: unknown, id?: unknown, sourceWindow = null) => context.createInstanceWindow(options, sourceWindow, GEOMETRY, id),
    async restart(raw?: unknown) {
      const captured = context.captureWindowSessionState()

      const parsed = state.parseWindowSessionState(raw === undefined
        ? state.serializeWindowSessionState(captured.windows, captured.instanceWindows)
        : JSON.stringify(raw))

      for (const win of [...context.instanceWindows]) {win.events.closed()}
      assert.equal(context.instanceWindowIds.size, 0, 'closed windows release identities')
      assert.equal(context.instanceWindowProfilePins.size, 0, 'closed windows release pin metadata')
      loaded.clear()
      context.readWindowSessionState = () => parsed
      await context.restoreWindowSessionsOnce()

      return parsed
    }
  }
}

test('ordinary inherited-route peers stay unpinned across capture/serialize/restore and a second restart', async () => {
  const h = harness()
  // Source has a non-default remote route; ordinary New Window inherits boot
  // routing, but must NOT acquire the explicit-profile New-session default.
  const sourceWindow = h.open({ profile: 'work', connectionId: 'remote-a' }, 'source')
  h.open(undefined, 'peer-a', sourceWindow)
  h.open(undefined, 'peer-b', sourceWindow)
  const before = h.urls().filter(u => u.searchParams.get('windowId') !== 'source')
  assert.ok(before.every(u => !u.searchParams.has('profileWindow')))
  await h.restart()
  const peers = h.urls().filter(u => u.searchParams.get('windowId') !== 'source')
  assert.deepEqual(peers.map(u => u.searchParams.get('windowId')), ['peer-a', 'peer-b'])
  assert.ok(peers.every(u => u.searchParams.get('profile') === 'work' && u.searchParams.get('connectionId') === 'remote-a'))
  assert.ok(peers.every(u => !u.searchParams.has('profileWindow')), 'ordinary peers must not become pinned on restore')
  await h.restart()
  assert.ok(h.urls().filter(u => u.searchParams.get('windowId') !== 'source').every(u => !u.searchParams.has('profileWindow')))
})

test('explicit profile pins survive capture/serialize/restore', async () => {
  const h = harness()
  h.open({ profile: 'work', connectionId: 'remote-a' }, 'pinned')
  assert.equal(h.urls()[0].searchParams.get('profileWindow'), '1')
  await h.restart()
  assert.equal(h.urls()[0].searchParams.get('profileWindow'), '1')
  assert.equal(h.urls()[0].searchParams.get('profile'), 'work')
  assert.equal(h.urls()[0].searchParams.get('connectionId'), 'remote-a')
})

// Regression coverage for t_208bf984: every profile-persistence test above
// (and in session-windows.test.ts) exercises a REMOTE registry route
// (connectionId: 'remote-a'). In production the common non-default-profile
// case is a LOCAL profile (connectionId: null, no registry connection at
// all, e.g. a "maestro" profile running on this same machine) — a
// combination nothing here checked, closing exactly the kind of gap that let
// a profile-specific regression slip through 187/187 "passing" tests.
test('explicit LOCAL (connectionId: null) profile pin survives capture/serialize/restore', async () => {
  const h = harness()
  h.open({ profile: 'maestro', connectionId: null }, 'maestro-win')
  assert.equal(h.urls()[0].searchParams.get('profile'), 'maestro')
  assert.equal(h.urls()[0].searchParams.get('connectionId'), '')
  assert.equal(h.urls()[0].searchParams.get('profileWindow'), '1')
  await h.restart()
  assert.equal(h.urls()[0].searchParams.get('profile'), 'maestro', 'LOCAL profile must survive restart, not revert to default')
  assert.equal(h.urls()[0].searchParams.get('connectionId'), '')
  assert.equal(h.urls()[0].searchParams.get('profileWindow'), '1')
})

test('LOCAL (connectionId: null) inherited non-pinned profile peer survives capture/serialize/restore', async () => {
  const h = harness()
  // LOCAL maestro source, pinned explicitly…
  const sourceWindow = h.open({ profile: 'maestro', connectionId: null }, 'source')
  // …and an ordinary ⌘⇧N peer that only INHERITS the maestro route (not pinned).
  h.open(undefined, 'peer-a', sourceWindow)
  const peerBefore = h.urls().find(u => u.searchParams.get('windowId') === 'peer-a')!
  assert.equal(peerBefore.searchParams.get('profile'), 'maestro')
  assert.ok(!peerBefore.searchParams.has('profileWindow'))
  await h.restart()
  const peerAfter = h.urls().find(u => u.searchParams.get('windowId') === 'peer-a')!
  assert.equal(peerAfter.searchParams.get('profile'), 'maestro', 'inherited LOCAL profile must survive restart, not revert to default')
  assert.ok(!peerAfter.searchParams.has('profileWindow'), 'an inherited peer must not become pinned on restore')
})

test('legacy and malformed saved IDs reopen with fresh distinct isolated IDs, never the main bare-key identity', async () => {
  const h = harness()
  await h.restart({ version: 1, windows: [], instanceWindows: [
    { profile: 'work', connectionId: 'remote-a', geometry: GEOMETRY },
    { profile: 'work', connectionId: 'remote-a', geometry: GEOMETRY, windowId: '../bad' }
  ] })
  const ids = h.urls().map(u => u.searchParams.get('windowId'))
  assert.deepEqual(ids, ['fresh-1', 'fresh-2'])
  assert.equal(new Set(ids).size, 2)
  assert.ok(h.urls().every(u => u.searchParams.get('peer') === '1' && !u.searchParams.has('profileWindow')))
  await h.restart()
  assert.deepEqual(h.urls().map(u => u.searchParams.get('windowId')), ids)
})

test('duplicate saved rows are sanitized and a live constructor collision mints a fresh identity', async () => {
  const h = harness()
  await h.restart({ version: 1, windows: [], instanceWindows: [
    { profile: 'default', connectionId: null, geometry: GEOMETRY, windowId: 'same-id' },
    { profile: 'work', connectionId: null, geometry: GEOMETRY, windowId: 'same-id' }
  ] })
  assert.deepEqual(h.urls().map(u => u.searchParams.get('windowId')), ['same-id'])
  h.open(undefined, 'same-id')
  assert.deepEqual(h.urls().map(u => u.searchParams.get('windowId')), ['same-id', 'fresh-1'])
})

test('constructor itself rejects malformed and missing IDs even without the saved-state sanitizer', () => {
  const h = harness()

  for (const id of [undefined, null, '../bad', 'has space', 'x'.repeat(65)]) {h.open(undefined, id)}
  assert.deepEqual(h.urls().map(u => u.searchParams.get('windowId')), ['fresh-1', 'fresh-2', 'fresh-3', 'fresh-4', 'fresh-5'])
})
