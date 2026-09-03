// IPC surface for HUD mode (the chrome-free floating chat band). Extracted
// from main.ts; the HUD window handle and session-id latch stay injected
// because main.ts owns the window lifecycle and the close broadcast reads the
// latch when handing the session back to the app window.
import { type BrowserWindow, ipcMain, screen } from 'electron'

import type { HudAskPayload, HudLaunchOptions, HudPrefs, HudPrefsStatus } from './hud-ask'
import { createHudDragSession } from './hud-drag'
import { normalizeHudResizeBounds } from './hud-geometry'
import { hudWindowingView, resolveHudWindowing } from './hud-windowing'
import { hudFrostFor, type TranslucencyState } from './translucency'

function hudWindowing() {
  return resolveHudWindowing(process.platform, process.env, process.argv)
}

export interface HudIpcDeps {
  isMac: boolean
  /** Main's authoritative translucency state (Settings → Appearance). */
  getTranslucencyState: () => TranslucencyState
  getHudWindow: () => BrowserWindow | null
  openHudWindow: (sessionId: null | string, profile: null | string) => void
  closeHudWindow: () => void
  resetHudLayout: () => boolean
  setHudSessionId: (sessionId: null | string) => void
  /** Follow / ask prefs — main is authoritative (it owns the chord + hook). */
  getHudPrefs: () => HudPrefsStatus
  updateHudPrefs: (patch: Partial<HudPrefs>) => HudPrefsStatus
  /** The ask payload parked for a HUD renderer that was not up yet. */
  takePendingAsk: () => HudAskPayload | null
  getLaunchOptions: () => HudLaunchOptions
  /** Open a room in the MAIN window on the HUD's behalf. */
  openRoom: (groupId: string) => boolean
  /** A parsed HUD voice/typed command (place, come-here, follow, hide). */
  runCommand: (command: { anchor?: string; kind: string; on?: boolean }) => boolean
  /** The app window, where Bot Mode's room engine runs. */
  getMainWindow: () => BrowserWindow | null
}

export function registerHudIpc({
  isMac,
  getTranslucencyState,
  getHudWindow,
  openHudWindow,
  closeHudWindow,
  resetHudLayout,
  setHudSessionId,
  getHudPrefs,
  updateHudPrefs,
  takePendingAsk,
  getLaunchOptions,
  openRoom,
  runCommand,
  getMainWindow
}: HudIpcDeps) {
  const hudDrag = createHudDragSession()

  // The renderer needs this before first paint so X11 never installs the
  // Chromium drag region that steals modifier-drag gestures from the WM.
  // Main answers because it owns the actual Ozone backend selection.
  ipcMain.on('hermes:hud:native-drag', event => {
    event.returnValue = hudWindowing().move === 'native-drag'
  })

  ipcMain.on('hermes:hud:windowing', event => {
    event.returnValue = hudWindowingView(hudWindowing())
  })

  // X11/KWin window transfer: a renderer-driven grab is temporarily sticky so
  // the user can keep Ctrl+primary-button held while invoking KDE's desktop
  // switch shortcut. Clearing sticky on release makes Chromium assign the
  // window to `_NET_CURRENT_DESKTOP`, exactly like releasing a native titlebar
  // drag on the destination desktop. Native Wayland owns its move loop and
  // Windows/macOS stay out of this Linux-specific bridge.
  ipcMain.on('hermes:hud:workspace-transfer', (event, transferring) => {
    const hudWindow = getHudWindow()

    if (
      !hudWindow ||
      hudWindow.isDestroyed() ||
      event.sender !== hudWindow.webContents ||
      !hudWindowing().workspaceTransfer
    ) {
      return
    }

    try {
      hudWindow.setVisibleOnAllWorkspaces(Boolean(transferring))
    } catch {
      // Workspace APIs are window-manager capabilities — best effort.
    }
  })

  // Whether the band currently covers the window below the bar. The renderer
  // is the only party that can know this (it measures the transcript), and it
  // is half of the frost decision — the other half is the user's setting,
  // which main owns. Latched so a Settings change can re-decide without
  // waiting for the HUD to report again.
  let bandShowing = false
  let applied: null | string = null
  let appliedTo: BrowserWindow | null = null

  // Real frosted glass behind the band — the thing CSS backdrop-filter cannot do,
  // because Chromium composites a transparent window's page against nothing and
  // the desktop is not in its backdrop root. The material IS the window's content
  // view, so it frosts the whole rectangle; the HUD's layout leaves no dead
  // margins for that reason, and it only turns on while the band is showing
  // (idle HUD mode must be the bar and nothing else).
  //
  // macOS ONLY. Windows' equivalent (setBackgroundMaterial → the DWM backdrop)
  // is mutually exclusive with window transparency, so it is not called at all
  // here — see the note at the bottom of this function.
  //
  // Diffed before issuing: `setVibrancy` carries a 150ms animation that restarts
  // if re-issued, so a repeated call would keep the material from ever settling
  // (the same churn the chat windows' native-diff contract exists to prevent).
  //
  // The diff is keyed to the WINDOW as well as the value. A HUD respawn (the
  // profile switch in openHudWindow destroys and rebuilds it) hands back a
  // fresh window carrying no material, and a latch that only remembered the
  // value would recognise its own last answer and skip — leaving the new HUD
  // unfrosted until something else happened to change the signature.
  const applyHudFrost = () => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed()) {
      applied = null
      appliedTo = null

      return
    }

    const frost = hudFrostFor(getTranslucencyState(), bandShowing)
    const signature = `${frost.vibrancy ?? 'off'}:${frost.backgroundMaterial}`

    if (applied === signature && appliedTo === hudWindow) {
      return
    }

    applied = signature
    appliedTo = hudWindow

    if (isMac && typeof hudWindow.setVibrancy === 'function') {
      hudWindow.setVibrancy(frost.vibrancy)
    }

    // Windows: never touch setBackgroundMaterial on the HUD. Live-verified on
    // Win11 (Electron 40.10.2, RTX 4090): ANY setBackgroundMaterial call on a
    // transparent window — including 'none', which is what the idle HUD asks
    // for — permanently kills per-pixel alpha, and every transparent pixel
    // composites as opaque white. Neither 'auto' nor a follow-up
    // setBackgroundColor('#00000000') restores it. The DWM backdrop and window
    // transparency are mutually exclusive, so the Windows HUD keeps the CSS
    // tint the sheet already paints and skips the native frost entirely.
  }

  ipcMain.handle('hermes:hud:open', async (_event, request) => {
    openHudWindow(
      typeof request?.sessionId === 'string' ? request.sessionId : null,
      typeof request?.profile === 'string' ? request.profile : null
    )

    return { ok: true }
  })

  ipcMain.handle('hermes:hud:frost', (_event, showing) => {
    bandShowing = Boolean(showing)
    applyHudFrost()

    return { ok: true }
  })

  // Let clicks fall through the HUD wherever it isn't really there. An
  // always-on-top window eats every click inside its rectangle, and most of that
  // rectangle is a faded-out band over whatever the user is actually working in.
  // `forward` keeps mousemove flowing so the renderer can re-arm when the cursor
  // reaches the bar.
  ipcMain.on('hermes:hud:ignore-mouse', (_event, ignore) => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed()) {
      return
    }

    // On X11 ignore-mouse is a one-way door: setIgnoreMouseEvents(false)
    // cannot restore the input region afterwards. Veto the request there so
    // the HUD stays a normal solid window. Native Wayland and macOS/Windows
    // keep the per-element path.
    if (Boolean(ignore) && !hudWindowing().ignoreMouse) {
      return
    }

    hudWindow.setIgnoreMouseEvents(Boolean(ignore), { forward: true })
  })

  ipcMain.on('hermes:hud:begin-move', event => {
    const hudWindow = getHudWindow()

    if (
      !hudWindow ||
      hudWindow.isDestroyed() ||
      event.sender !== hudWindow.webContents ||
      !hudWindowing().clientPlacement
    ) {
      return
    }

    const [x, y] = hudWindow.getPosition()
    hudDrag.begin(screen.getCursorScreenPoint(), { x, y })
  })

  ipcMain.on('hermes:hud:end-move', event => {
    const hudWindow = getHudWindow()

    if (hudWindow && !hudWindow.isDestroyed() && event.sender !== hudWindow.webContents) {
      return
    }

    hudDrag.end()
  })

  ipcMain.on('hermes:hud:move-by', (event, delta) => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed() || event.sender !== hudWindow.webContents) {
      return
    }

    const width = Number(delta?.width)
    const height = Number(delta?.height)

    if (!Number.isFinite(width) || !Number.isFinite(height) || !hudWindowing().clientPlacement) {
      return
    }

    const origin = hudDrag.origin(screen.getCursorScreenPoint())

    if (!origin) {
      return
    }

    // Cursor − grab offset in Electron DIP (see hud-drag.ts). setBounds —
    // NOT setPosition: on Windows, a transparent frameless window silently
    // grows ~1px per setPosition call (worse at >100% DPI). The renderer
    // snapshots outerWidth/outerHeight when the composer drag arms and
    // re-pins to that size on every move (same pattern as the pet overlay).
    hudWindow.setBounds({
      x: origin.x,
      y: origin.y,
      width: Math.round(width),
      height: Math.round(height)
    })
  })

  // Resize from the HUD's edge/corner handles. The window is created non-resizable
  // (see spawnHudWindow — a transparent frameless window must not expose a
  // system resize hot-zone, or dragging grows it), which on Windows/Linux also
  // blocks programmatic setBounds sizing — so briefly flip resizable on while
  // the size actually changes, exactly like the pet overlay's wheel-scale does.
  ipcMain.on('hermes:hud:set-bounds', (event, bounds) => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed() || event.sender !== hudWindow.webContents || !bounds) {
      return
    }

    const nextBounds = normalizeHudResizeBounds(bounds)

    if (!nextBounds) {
      return
    }

    const win = hudWindow
    const { width, height } = nextBounds
    const [curW, curH] = win.getSize()
    const resizing = width !== curW || height !== curH
    const restoreResizeLock = resizing && !win.isResizable()

    try {
      if (restoreResizeLock) {
        win.setResizable(true)
      }

      win.setBounds(nextBounds)
    } catch {
      // The window may disappear between validation and the native call.
    } finally {
      if (restoreResizeLock && !win.isDestroyed()) {
        win.setResizable(false)
      }
    }
  })

  ipcMain.handle('hermes:hud:reset-layout', event => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed() || event.sender !== hudWindow.webContents) {
      return { ok: false }
    }

    return { ok: resetHudLayout() }
  })

  // The HUD renderer reporting which session it is on, so the close broadcast
  // can hand it back to the app window (see hudSessionId).
  ipcMain.on('hermes:hud:session', (event, sessionId) => {
    const hudWindow = getHudWindow()

    if (hudWindow && !hudWindow.isDestroyed() && event.sender === hudWindow.webContents) {
      setHudSessionId(typeof sessionId === 'string' && sessionId ? sessionId : null)
    }
  })

  ipcMain.handle('hermes:hud:close', async () => {
    closeHudWindow()

    return { ok: true }
  })

  // Follow / ask prefs. Readable and writable from ANY window — Settings
  // lives in the app window, the follow toggle in the HUD's controls row —
  // and main answers with the ground truth either way.
  ipcMain.handle('hermes:hud:prefs:get', () => getHudPrefs())

  ipcMain.handle('hermes:hud:prefs:set', (_event, patch) => {
    const record = patch && typeof patch === 'object' ? (patch as Record<string, unknown>) : {}
    const next: Partial<HudPrefs> = {}

    if (typeof record.follow === 'boolean') {
      next.follow = record.follow
    }

    if (typeof record.askOnRightClick === 'boolean') {
      next.askOnRightClick = record.askOnRightClick
    }

    if (typeof record.pets === 'boolean') {
      next.pets = record.pets
    }

    if (record.petByAgent && typeof record.petByAgent === 'object') {
      next.petByAgent = record.petByAgent as HudPrefs['petByAgent']
    }

    if (typeof record.askShortcut === 'string') {
      next.askShortcut = record.askShortcut
    }

    return updateHudPrefs(next)
  })

  // The HUD renderer, once its sheet is mounted, collects an ask that arrived
  // before it existed. HUD sender only: the payload names another app's window
  // and carries a screenshot path.
  ipcMain.handle('hermes:hud:ask-pending', event => {
    const hudWindow = getHudWindow()

    if (!hudWindow || hudWindow.isDestroyed() || event.sender !== hudWindow.webContents) {
      return null
    }

    return takePendingAsk()
  })

  ipcMain.handle('hermes:hud:launch-options', () => getLaunchOptions())

  ipcMain.handle('hermes:hud:open-room', (_event, request) => {
    const groupId = typeof request?.groupId === 'string' ? request.groupId.trim() : ''

    return { ok: groupId ? openRoom(groupId) : false }
  })

  // "HUD top left" / "HUD follow me" / "HUD come here" / "HUD hide" — parsed in
  // the renderer (src/lib/hud-voice-command.ts), carried out here. Any window
  // may send one: the voice loop runs in the HUD and in the app window alike.
  // ── The HUD talking into a room ─────────────────────────────────────────
  // Rooms live in the app window. The HUD asks main; main asks the primary
  // renderer over a request id and answers when the reply comes back (or
  // times out — a hung primary must not hang the HUD's composer).
  const HUD_ROOM_TIMEOUT_MS = 6000
  let roomRequestSeq = 0
  const roomRequests = new Map<string, (value: unknown) => void>()
  let hudWatchedRoom: null | string = null

  const askPrimary = (channel: string, payload: Record<string, unknown>): Promise<unknown> => {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return Promise.resolve(null)
    }

    const requestId = `hud-room-${++roomRequestSeq}`

    return new Promise(resolve => {
      const timer = setTimeout(() => {
        roomRequests.delete(requestId)
        resolve(null)
      }, HUD_ROOM_TIMEOUT_MS)

      roomRequests.set(requestId, value => {
        clearTimeout(timer)
        roomRequests.delete(requestId)
        resolve(value)
      })

      mainWindow.webContents.send(channel, { ...payload, requestId })
    })
  }

  const settle = (reply: unknown, pick: (record: Record<string, unknown>) => unknown) => {
    const record = reply && typeof reply === 'object' ? (reply as Record<string, unknown>) : {}
    const requestId = typeof record.requestId === 'string' ? record.requestId : ''
    const resolve = roomRequests.get(requestId)

    if (resolve) {
      resolve(pick(record))
    }
  }

  ipcMain.on('hermes:hud:room-feed-reply', (_event, reply) => settle(reply, record => record.feed ?? null))
  ipcMain.on('hermes:hud:room-post-reply', (_event, reply) => settle(reply, record => record.ok === true))

  ipcMain.handle('hermes:hud:room-feed', (_event, request) => {
    const groupId = typeof request?.groupId === 'string' ? request.groupId.trim() : ''

    return groupId ? askPrimary('hermes:hud:room-feed-request', { groupId }) : Promise.resolve(null)
  })

  ipcMain.handle('hermes:hud:room-post', async (_event, request) => {
    const groupId = typeof request?.groupId === 'string' ? request.groupId.trim() : ''
    const text = typeof request?.text === 'string' ? request.text.trim() : ''

    if (!groupId || !text) {
      return { ok: false }
    }

    return { ok: (await askPrimary('hermes:hud:room-post', { groupId, text })) === true }
  })

  // Which room the HUD is watching: the primary pushes that room's feed on
  // every change, and a respawned HUD renderer asks for it again on mount.
  ipcMain.handle('hermes:hud:watch-room', (_event, request) => {
    hudWatchedRoom = typeof request?.groupId === 'string' && request.groupId.trim() ? request.groupId.trim() : null
    const mainWindow = getMainWindow()

    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('hermes:hud:watch-room', { groupId: hudWatchedRoom })
    }

    return { groupId: hudWatchedRoom }
  })

  ipcMain.on('hermes:hud:room-feed-push', (_event, feed) => {
    const hudWindow = getHudWindow()
    const groupId = feed && typeof feed === 'object' ? (feed as { groupId?: unknown }).groupId : null

    if (hudWindow && !hudWindow.isDestroyed() && typeof groupId === 'string' && groupId === hudWatchedRoom) {
      hudWindow.webContents.send('hermes:hud:room-feed', feed)
    }
  })

  ipcMain.handle('hermes:hud:command', (_event, command) => {
    const record = command && typeof command === 'object' ? (command as Record<string, unknown>) : {}
    const kind = typeof record.kind === 'string' ? record.kind : ''

    if (!kind) {
      return { ok: false }
    }

    return {
      ok: runCommand({
        kind,
        ...(typeof record.anchor === 'string' ? { anchor: record.anchor } : {}),
        ...(typeof record.on === 'boolean' ? { on: record.on } : {})
      })
    }
  })

  // Main re-applies the frost when the translucency SETTING changes, since the
  // band's own report only fires when the band itself moves.
  return { applyHudFrost }
}
