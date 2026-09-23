import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { app, BrowserWindow, globalShortcut, screen, systemPreferences } from 'electron'

import { requestHudClose } from './hud-close'
import { cursorPointInWindow } from './hud-cursor'
import { startHudGameOverlayWatch } from './hud-game-overlay'
import { applyHudResetBounds, defaultHudBounds } from './hud-geometry'
import { applyHudElectronOverlay, promoteHudOverlay } from './hud-overlay'
import { snapHudBounds } from './hud-snap'
import { createHudSnapShortcut } from './hud-snap-shortcut'
import { buildHudWindowUrl } from './hud-url'
import { resolveHudWindowing } from './hud-windowing'
import { createQuickEntryShortcut, quickEntryWindowBounds, sanitizeQuickEntrySettings } from './quick-entry'
import { attachRendererConsoleCapture } from './renderer-log'
import { chatWindowWebPreferences } from './session-windows'
import { enumerateWindowsFrontToBack, enumerationFailed } from './window-below'
import { installWindowRendererLifecycle } from './window-renderer-lifecycle'
import { debounce } from './window-state'
import { zoomWiringForWindowKind } from './zoom'

// HUD and Quick Entry share ownership of transient always-on-top windows and
// their global accelerators. Main supplies its live windows and late services.
export function createDesktopShellOverlayRuntime(deps: {
  DEV_SERVER: string | undefined
  HUD_WINDOW_TITLE: string
  IS_MAC: boolean
  PRELOAD_PATH: string
  bindGeometryPersistence: (...args: any[]) => any
  focusWindow: (window: BrowserWindow | null) => void
  getMainWindow: () => BrowserWindow | null
  getStreamThrottle: () => { register: (window: BrowserWindow) => void }
  loadWindowUrl: (...args: any[]) => any
  rememberLog: (line: string) => void
  resolveRendererIndex: () => string
  wireCommonWindowHandlers: (...args: any[]) => any
  wireWindowReveal: (...args: any[]) => any
  writeFileAtomic: (...args: any[]) => any
}) {
  const {
    DEV_SERVER,
    HUD_WINDOW_TITLE,
    IS_MAC,
    PRELOAD_PATH,
    bindGeometryPersistence,
    focusWindow,
    getMainWindow,
    getStreamThrottle,
    loadWindowUrl,
    rememberLog,
    resolveRendererIndex,
    wireCommonWindowHandlers,
    wireWindowReveal,
    writeFileAtomic
  } = deps

  // ── HUD mode ────────────────────────────────────────────────────────────────
  //
  // The chrome-free floating chat: a transparent, frameless, always-on-top
  // window showing only the composer and its scrollback, so Hermes can be driven
  // while the user works in another app.
  //
  // Unlike the pet overlay / quick entry, this is a FULL app renderer with its
  // own gateway — the same thing createInstanceWindow() spawns, reshaped. That
  // is deliberate: the HUD renders the real chat surface, so its composer is the
  // app's composer (slash commands, attachments, queue, voice) instead of a
  // lookalike that drifts. Entering HUD mode hides the main window; leaving
  // restores it.
  let hudWindow: BrowserWindow | null = null

  // Whether closing the HUD should bring the main window back. Armed whenever a
  // live main window exists at HUD-open time, visible or not: the HUD hides the
  // app window itself, so a main window minimized or behind another app when
  // the HUD opened still needs a surface back — arming only on `isVisible()`
  // left the user with NO Hermes window after the second toggle (#88513).
  let hudRestoreMainWindow = false

  // The session the HUD is currently on, reported by its renderer whenever the
  // selection changes. Leaving HUD mode is a HANDOFF, not just a window close:
  // the gateway binds a session's event stream to exactly one socket, so the
  // turn the HUD started is streaming to the HUD's socket and the app window
  // hears nothing. The app has to re-resume that session to take the stream
  // back, and it can only do that if it knows which session to ask for — the
  // HUD may have switched sessions, or started a new one the app has never
  // seen. Main is the only party that outlives the HUD's renderer, so it holds
  // the id and hands it over in the close broadcast.
  let hudSessionId = null

  // The profile the live HUD renderer booted against (rides hudUrl's query
  // string). A renderer adopts its backend once at boot, so a retarget onto a
  // session from a DIFFERENT profile cannot be a same-window `goto` — the HUD
  // must be respawned against the new profile's backend (see openHudWindow).
  let hudProfile = null

  // A wide, short bar parked near the bottom of the active display — the shape
  // of a game chat frame, and where one belongs. Defaults only: once the user
  // moves or resizes the HUD, hud-state.json wins (same pattern as the main
  // window's window-state.json).
  const HUD_STATE_PATH = path.join(app.getPath('userData'), 'hud-state.json')

  function readHudState() {
    try {
      const raw = JSON.parse(fs.readFileSync(HUD_STATE_PATH, 'utf8'))

      if (
        [raw?.x, raw?.y, raw?.width, raw?.height].every(v => Number.isFinite(v)) &&
        raw.width >= 380 &&
        raw.height >= 160
      ) {
        return raw
      }
    } catch {
      // First run / unreadable — fall through to defaults.
    }

    return null
  }

  function persistHudState() {
    if (!hudWindow || hudWindow.isDestroyed()) {
      return
    }

    try {
      const { x, y, width, height } = hudWindow.getNormalBounds()
      fs.mkdirSync(path.dirname(HUD_STATE_PATH), { recursive: true })
      writeFileAtomic(HUD_STATE_PATH, JSON.stringify({ x, y, width, height }, null, 2))
    } catch (err) {
      rememberLog(`[hud-state] persist failed: ${err?.message || err}`)
    }
  }

  function resetHudWindowLayout(): boolean {
    if (!hudWindow || hudWindow.isDestroyed()) {
      return false
    }

    const win = hudWindow
    const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
    const bounds = defaultHudBounds(display?.workArea)

    if (!applyHudResetBounds(win, bounds)) {
      rememberLog('[hud-state] reset layout failed while applying native bounds')

      return false
    }

    persistHudState()

    return true
  }

  const schedulePersistHudState = debounce(persistHudState, 250)

  // How often Linux gets told where the cursor is. Fast enough that the bar is
  // solid before a click lands after the pointer arrives, cheap enough to leave
  // running for as long as the HUD is open — it is one `getCursorScreenPoint()`
  // and, when the answer has not changed, nothing else.
  const HUD_CURSOR_POLL_MS = 60

  // Snap-to-pointer — global ⌘⇧G while the HUD is open (tap, not hold).
  const HUD_SNAP_ANCHOR_Y = 48

  function hudWindowing() {
    return resolveHudWindowing(process.platform, process.env, process.argv)
  }

  function applyHudSnapToPointer() {
    if (!hudWindow || hudWindow.isDestroyed() || !hudWindowing().clientPlacement) {
      return
    }

    const cursor = screen.getCursorScreenPoint()
    const bounds = hudWindow.getBounds()
    const display = screen.getDisplayNearestPoint(cursor)
    const workArea = display?.workArea ?? bounds
    const anchor = { x: Math.round(bounds.width / 2), y: HUD_SNAP_ANCHOR_Y }

    const origin = snapHudBounds(
      cursor,
      anchor,
      { width: bounds.width, height: bounds.height },
      hudWindow.webContents.getZoomFactor(),
      workArea
    )

    // setBounds — NOT setPosition alone: on Windows, a transparent frameless
    // window silently grows ~1px per setPosition call (see move-by handler).
    // On native Wayland the compositor ignores the position half; the snap
    // shortcut is therefore a documented no-op there.
    hudWindow.setBounds({
      x: origin.x,
      y: origin.y,
      width: bounds.width,
      height: bounds.height
    })
  }

  const hudSnapShortcut = createHudSnapShortcut(globalShortcut, applyHudSnapToPointer)

  function registerHudSnapShortcut() {
    if (!hudSnapShortcut.register()) {
      rememberLog('[hud] snap shortcut unavailable — CommandOrControl+Shift+G may be owned by another app')
    }
  }

  /**
   * Feed the HUD renderer the cursor position on Linux.
   *
   * Everywhere else the renderer learns this from mousemove, which keeps arriving
   * while the window ignores the mouse because we pass `{ forward: true }`. That
   * option is macOS/Windows only. Without it a Linux HUD stops hearing the
   * pointer the moment it turns click-through, so it can never notice the pointer
   * coming back and stays transparent — the bar is there, and clicking it hits
   * whatever is behind. Main can still see the cursor, so it says so.
   *
   * Deliberately the same decision, just a different source for one input: the
   * renderer runs its usual hit test on the point it is handed. Re-deciding
   * anything here would put a second, drifting copy of the click-through rules in
   * the main process.
   */
  function startHudCursorFeed(win: BrowserWindow) {
    const windowing = hudWindowing()

    if (!windowing.cursorFeed) {
      if (!windowing.ignoreMouse) {
        try {
          win.setIgnoreMouseEvents(false)
        } catch {
          // best effort
        }
      }

      return
    }

    let last: string | null = null

    const timer = setInterval(() => {
      if (win.isDestroyed() || !win.isVisible()) {
        return
      }

      const point = cursorPointInWindow(screen.getCursorScreenPoint(), win.getBounds(), win.webContents.getZoomFactor())

      // Off-window is a real answer (it is what hands the mouse back), so it is
      // sent — once. Only an unchanged answer is dropped, to keep an idle cursor
      // from waking the renderer 16 times a second.
      const key = point ? `${Math.round(point.x)},${Math.round(point.y)}` : 'out'

      if (key === last) {
        return
      }

      last = key
      win.webContents.send('hermes:hud:cursor', point)
    }, HUD_CURSOR_POLL_MS)

    win.on('closed', () => clearInterval(timer))
  }

  /**
   * Watch for a fullscreen app under the HUD (the Discord-style game overlay
   * cue) and feed the answer to its renderer. Pure detection lives in
   * hud-game-overlay.ts; enumeration is the same front-to-back walk the
   * read_window_below tool uses. The renderer answers with the low-opacity
   * treatment (`data-hud-game`), so main stays out of the styling business.
   */
  function startHudGameOverlayFeed(win: BrowserWindow) {
    const titlesAvailable = IS_MAC ? systemPreferences.getMediaAccessStatus?.('screen') === 'granted' : true

    let last = { active: false, app: '' }

    const push = (state: { active: boolean; app: string }) => {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:hud:game-overlay', state)
      }
    }

    // Replay the latest state to every load of this window. The watch pushes only
    // on CHANGE and its first tick fires the moment the window is created — well
    // before the renderer has mounted its listener — so a HUD opened over a game
    // that is already fullscreen would hear the one and only message before it
    // could receive it, then sit at "no game" forever while main was certain it
    // had reported one. (Same reason quick entry caches its last state push.)
    // did-finish-load also covers HMR full reloads during development.
    win.webContents.on('did-finish-load', () => push(last))

    // The watch gives up after two failed enumerations and never says so, which
    // is how a HUD that cannot see the screen at all — no game cue, and
    // read_window_below failing beside it — leaves nothing in the log to explain
    // itself. Report the reason once; the null keeps the watch's contract.
    let reported = false

    const enumerate = async () => {
      const windows = await enumerateWindowsFrontToBack(process.pid, titlesAvailable)

      if (!enumerationFailed(windows)) {
        return windows
      }

      if (!reported) {
        reported = true
        console.warn(`[hermes] HUD cannot enumerate windows: ${windows.reason}`)
      }

      return null
    }

    const dispose = startHudGameOverlayWatch({
      enumerate,
      displayBounds: () => screen.getDisplayMatching(win.getBounds()).bounds,
      selfPid: process.pid,
      send: state => {
        last = state
        push(state)
      }
    })

    win.on('closed', dispose)
  }

  function hudBounds() {
    // Remembered spot first — validated against the LIVE displays so a HUD
    // parked on an unplugged monitor comes back on-screen instead of lost.
    const saved = readHudState()

    if (saved) {
      const onScreen = screen.getAllDisplays().some(d => {
        const a = d.workArea

        return (
          saved.x < a.x + a.width - 40 &&
          saved.x + saved.width > a.x + 40 &&
          saved.y < a.y + a.height - 40 &&
          saved.y + saved.height > a.y + 40
        )
      })

      if (onScreen) {
        return saved
      }
    }

    const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
    const area = display?.workArea

    return defaultHudBounds(area)
  }

  function hudUrl(sessionId, profile) {
    // The profile rides the query string next to `win=hud` (BEFORE the '#', so
    // HashRouter never sees it). The HUD renderer's gateway boot reads it and
    // adopts that backend instead of the primary — without it, a HUD opened on a
    // non-primary profile's conversation resolves the session id against the
    // wrong backend and falls back to the default profile's last session.
    return buildHudWindowUrl(sessionId, {
      devServer: DEV_SERVER,
      profile,
      rendererIndexPath: DEV_SERVER ? undefined : resolveRendererIndex()
    })
  }

  // Tell every window whether the HUD is up, so a toggle in any of them reads
  // the truth even when the HUD is closed from its own side (⌘W / its exit row).
  // Carries the HUD's session so the app window can re-home onto it on the way
  // out (see hudSessionId).
  function broadcastHudState(open) {
    const payload = { open, sessionId: hudSessionId }

    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:hud:changed', payload)
      }
    }
  }

  function spawnHudWindow(sessionId, profile) {
    const win = new BrowserWindow({
      ...hudBounds(),
      minWidth: 380,
      minHeight: 160,
      title: HUD_WINDOW_TITLE,
      frame: false,
      transparent: true,
      // NOT resizable. A transparent frameless window on Windows keeps a
      // system-level edge resize hot-zone while `resizable` is on — the OS
      // interprets pointer capture near the edge as a resize gesture, so the
      // window grows a few px every drag (worse at >100% DPI scaling). The
      // composer drag calls setPosition, which must move the window, not resize
      // it. Resizing is done by the renderer's edge/corner handles through
      // `hermes:hud:set-bounds`, which flips resizable on for the call — the
      // same pattern the pet overlay uses for its wheel-scale.
      resizable: false,
      // macOS AppKit's constrainFrameRect clamps setBounds to the current
      // display unless this is on. The HUD is moved by renderer-driven
      // setBounds (not a native titlebar drag), so without it the bar cannot
      // be dragged onto another monitor. No-op on Windows/Linux.
      enableLargerThanScreen: true,
      movable: true,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      // Keep the interactive macOS HUD as an ordinary NSWindow. NSPanel defaults
      // hidesOnDeactivate to true, which removes the HUD while the user works in
      // another app; the floating/all-spaces setup below supplies overlay behavior.
      skipTaskbar: !IS_MAC,
      hasShadow: false,
      alwaysOnTop: true,
      // Clips the vibrancy layer to the HUD's silhouette rather than a hard
      // rectangle — the frost stops where the window's corners do.
      roundedCorners: true,
      // Vibrancy must keep rendering while the window is BLURRED: streaming under
      // another app is the whole feature, and the default 'followWindow' kills
      // the frost the moment something else takes focus.
      visualEffectState: 'active',
      hiddenInMissionControl: IS_MAC,
      show: false,
      backgroundColor: '#00000000',
      // The full chat webPreferences — this window streams a real transcript, so
      // it needs everything a chat window needs (preload bridge, autoplay for
      // voice, the shared throttling contract).
      webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
    })

    applyHudElectronOverlay(win, process.platform)
    win.setHiddenInMissionControl?.(true)

    // Linux intentionally starts on ONE virtual desktop. During a renderer
    // grab, hermes:hud:workspace-transfer temporarily makes the X11 window
    // sticky; releasing it assigns the HUD to KDE's then-current desktop.

    // Streaming into a window that is ALWAYS blurred (the user is in another
    // app) is the entire feature, so it gets the same stream-aware unthrottling
    // every chat window does.
    getStreamThrottle().register(win)
    wireCommonWindowHandlers(win, zoomWiringForWindowKind('chat'))

    // Remember where the user parks and sizes it (debounced — these fire many
    // times mid-drag).
    bindGeometryPersistence(win, schedulePersistHudState)

    startHudCursorFeed(win)
    startHudGameOverlayFeed(win)

    wireWindowReveal(win, {
      show: () => {
        win.show()
        win.focus()
      },
      onRevealed: () => {
        // Step the app aside: the HUD IS the surface now.
        if (hudRestoreMainWindow && getMainWindow() && !getMainWindow().isDestroyed()) {
          getMainWindow().hide()
        }

        // Compositor overlay adapters (Hyprland float+pin today). Electron
        // alwaysOnTop is already set; this is the dialect some WMs actually hear.
        void promoteHudOverlay({ title: HUD_WINDOW_TITLE })
      }
    })

    win.on('closed', () => {
      if (hudWindow === win) {
        hudWindow = null
      } else if (hudWindow && !hudWindow.isDestroyed()) {
        // Superseded by a profile respawn: the replacement owns the shortcut,
        // the main-window restore and the toggles. Nothing to hand back.
        return
      }

      // Whether the close came from closeHudWindow() or from the window's own
      // side (a crashed renderer, a native close), this is the one teardown:
      // release the global snap shortcut, put the app back so the user is never
      // left with no surface, and correct every window's toggle.
      hudSnapShortcut.dispose()
      restoreMainWindowFromHud()
      broadcastHudState(false)
    })

    attachRendererConsoleCapture(win, 'hud', rememberLog)
    // Log-only lifecycle (#81290): the HUD is a compact auxiliary surface the
    // user can re-toggle; a dead renderer should be diagnosable, not resurrected.
    installWindowRendererLifecycle(win, { kind: 'hud', callbacks: { log: rememberLog } })
    loadWindowUrl(win, hudUrl(sessionId, profile), 'HUD')

    return win
  }

  // Put the app window back, and give it the keyboard. `focusWindow`, not a bare
  // `show()`: show() alone leaves a minimized window minimized, and on macOS a
  // shown-but-not-key window means the user is looking at the app with the
  // caret still belonging to whatever the HUD was floating over.
  function restoreMainWindowFromHud() {
    if (!hudRestoreMainWindow) {
      return
    }

    hudRestoreMainWindow = false
    focusWindow(getMainWindow())
  }

  // Take the HUD window down. The 'closed' handler stays attached so ONE path
  // owns the teardown (snap shortcut, main-window restore, close broadcast)
  // whether the window went via the exit button, ⌘W, a profile respawn, or the
  // grace deadline — detaching it before close() was how a renderer that never
  // answered the close left an always-on-top HUD nobody could dismiss and no
  // broadcast to correct the toggles.
  function destroyHudWindow(win: BrowserWindow) {
    if (hudWindow === win) {
      hudWindow = null
    }

    requestHudClose(win)
  }

  function openHudWindow(sessionId, profile) {
    const profileKey = typeof profile === 'string' && profile.trim() ? profile.trim() : null

    if (hudWindow && !hudWindow.isDestroyed()) {
      // Pointed at another PROFILE: the live renderer is bound to the old
      // profile's backend, and a renderer adopts its backend exactly once at
      // boot — an in-place goto would resolve the id against the wrong backend
      // (the #82285 fallback). Respawn against the right one. The old window's
      // 'closed' handler sees `hudWindow` already pointing at the replacement,
      // so it neither restores main nor broadcasts a false "closed".
      if (profileKey && hudProfile !== profileKey) {
        const previous = hudWindow

        hudSessionId = sessionId || null
        hudProfile = profileKey
        hudWindow = spawnHudWindow(sessionId, profileKey)
        previous.destroy()
        broadcastHudState(true)
        registerHudSnapShortcut()

        return hudWindow
      }

      // Already up, but pointed somewhere else — switch it rather than just
      // raising it. Asking for HUD mode from another tab means "put THIS
      // conversation in the HUD", and a plain focus leaves the wrong one there.
      if (sessionId && sessionId !== hudSessionId) {
        hudSessionId = sessionId
        hudWindow.webContents.send('hermes:hud:goto', sessionId)
        // Keep every window's idea of where the HUD is pointed in step, so the
        // toggle keeps reading "switch" vs "dismiss" correctly.
        broadcastHudState(true)
      }

      focusWindow(hudWindow)

      return hudWindow
    }

    hudRestoreMainWindow = Boolean(getMainWindow() && !getMainWindow().isDestroyed())
    hudSessionId = sessionId || null
    hudProfile = profileKey
    hudWindow = spawnHudWindow(sessionId, profileKey)
    broadcastHudState(true)
    registerHudSnapShortcut()

    return hudWindow
  }

  function closeHudWindow() {
    const win = hudWindow

    if (win && !win.isDestroyed()) {
      destroyHudWindow(win)

      return
    }

    // No live HUD (a renderer that died, a toggle racing the close): still
    // release what an open HUD holds, so the toggles read right.
    hudWindow = null
    hudSnapShortcut.dispose()
    restoreMainWindowFromHud()
    broadcastHudState(false)
  }

  // ── Quick Entry ─────────────────────────────────────────────────────────────
  //
  // A global shortcut summons a small frameless always-on-top composer from
  // anywhere, so a prompt can be fired without raising the whole app. The window
  // carries NO gateway connection: it hands its text to us, we forward it to the
  // PRIMARY renderer, and that renderer submits through the same prompt path the
  // normal composer uses (see store/quick-entry + hooks/use-quick-entry-bridge).
  //
  // Main owns the OS registration and the persisted preference (it must restore
  // the shortcut on a cold launch without the renderer ever visiting Settings),
  // same authority split as keep-awake. Registration failure is surfaced, never
  // swallowed: a chord another app already owns comes back as `error: 'taken'`.
  const QUICK_ENTRY_CONFIG_PATH = path.join(app.getPath('userData'), 'quick-entry.json')

  let quickEntryWindow = null

  // Latest state push from the primary renderer (connection + recent sessions),
  // replayed to a quick window that spawns after the push happened.
  let quickEntryLastState = null

  function readQuickEntrySettings() {
    try {
      return sanitizeQuickEntrySettings(JSON.parse(fs.readFileSync(QUICK_ENTRY_CONFIG_PATH, 'utf8')))
    } catch {
      // Missing / unreadable / malformed → shipped defaults (enabled, default chord).
      return sanitizeQuickEntrySettings(undefined)
    }
  }

  function writeQuickEntrySettings(settings) {
    try {
      fs.mkdirSync(path.dirname(QUICK_ENTRY_CONFIG_PATH), { recursive: true })
      fs.writeFileSync(QUICK_ENTRY_CONFIG_PATH, JSON.stringify(settings, null, 2), 'utf8')
    } catch (error) {
      rememberLog(`[quick-entry] write failed: ${error.message}`)
    }
  }

  function quickEntryUrl() {
    if (DEV_SERVER) {
      return `${DEV_SERVER.endsWith('/') ? DEV_SERVER.slice(0, -1) : DEV_SERVER}/?win=quick#/`
    }

    return `${pathToFileURL(resolveRendererIndex()).toString()}?win=quick#/`
  }

  function spawnQuickEntryWindow() {
    const cursor = screen.getCursorScreenPoint()
    const display = screen.getDisplayNearestPoint(cursor)
    const bounds = quickEntryWindowBounds(display?.workArea)

    const win = new BrowserWindow({
      ...bounds,
      frame: false,
      transparent: true,
      resizable: false,
      movable: true,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      // Same rationale as the pet overlay: on Windows/Linux keep the helper out
      // of the taskbar/alt-tab list; on macOS use an NSPanel so the frameless
      // capture window never becomes the app's cmd-tab anchor.
      skipTaskbar: !IS_MAC,
      hasShadow: true,
      alwaysOnTop: true,
      type: IS_MAC ? 'panel' : undefined,
      hiddenInMissionControl: IS_MAC,
      show: false,
      backgroundColor: '#00000000',
      webPreferences: {
        preload: PRELOAD_PATH,
        contextIsolation: true,
        sandbox: true,
        nodeIntegration: false,
        devTools: true
      }
    })

    win.setAlwaysOnTop(true, IS_MAC ? 'floating' : 'screen-saver')
    win.setHiddenInMissionControl?.(true)

    try {
      win.setVisibleOnAllWorkspaces(
        true,
        IS_MAC ? { visibleOnFullScreen: true, skipTransformProcessType: true } : undefined
      )
    } catch {
      // Not supported everywhere — best effort.
    }

    // Opts out of global UI zoom for the same reason as the pet overlay: it sizes
    // its own OS window and a zoomed composer would overflow it.
    wireCommonWindowHandlers(win, zoomWiringForWindowKind('quickEntry'))

    // Log-only renderer lifecycle (#81290): a dead quick-entry window must never
    // resurrect itself over the app, but its loss belongs in desktop.log.
    installWindowRendererLifecycle(win, { kind: 'quick', callbacks: { log: rememberLog } })

    // Hide on blur. The window must never hold the user's focus captive — losing
    // focus is the cheapest, least surprising dismiss (matches Spotlight).
    win.on('blur', () => {
      if (!win.isDestroyed()) {
        win.hide()
      }
    })

    win.on('closed', () => {
      if (quickEntryWindow === win) {
        quickEntryWindow = null
      }
    })

    // Replay the last known gateway state as soon as the page can hear it — a
    // freshly spawned quick window must not sit "disconnected" when the primary
    // renderer already reported a live gateway.
    win.webContents.on('did-finish-load', () => {
      if (!win.isDestroyed() && quickEntryLastState) {
        win.webContents.send('hermes:quick-entry:state', quickEntryLastState)
      }
    })

    attachRendererConsoleCapture(win, 'quick-entry', rememberLog)
    loadWindowUrl(win, quickEntryUrl(), 'Quick entry')

    return win
  }

  // Move the (already-open) window to the display the cursor is on, so the chord
  // summons it where the user is looking rather than where they last were.
  function repositionQuickEntryWindow(win) {
    try {
      const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
      win.setBounds(quickEntryWindowBounds(display?.workArea))
    } catch (error) {
      rememberLog(`[quick-entry] reposition failed: ${error.message}`)
    }
  }

  function showQuickEntryWindow() {
    if (!quickEntryWindow || quickEntryWindow.isDestroyed()) {
      // Reveal the window this call created, not whatever `quickEntryWindow`
      // points at by the time the event lands.
      const win = spawnQuickEntryWindow()
      quickEntryWindow = win

      wireWindowReveal(win, {
        show: () => {
          win.show()
          win.focus()
        }
      })

      return
    }

    repositionQuickEntryWindow(quickEntryWindow)
    quickEntryWindow.show()
    quickEntryWindow.focus()
    // Re-summoned: tell the renderer to clear any stale draft and refocus.
    quickEntryWindow.webContents.send('hermes:quick-entry:shown')
  }

  function hideQuickEntryWindow() {
    if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
      quickEntryWindow.hide()
    }
  }

  // The chord toggles: pressing it while the composer is up puts it away, so one
  // gesture does exactly one thing in both directions.
  function toggleQuickEntryWindow() {
    if (quickEntryWindow && !quickEntryWindow.isDestroyed() && quickEntryWindow.isVisible()) {
      hideQuickEntryWindow()

      return
    }

    showQuickEntryWindow()
  }

  const quickEntryShortcut = createQuickEntryShortcut(globalShortcut, toggleQuickEntryWindow)

  function applyQuickEntrySettings(settings) {
    const state = quickEntryShortcut.apply(settings)

    if (!settings.enabled) {
      // Turning the feature off must not leave an orphan always-on-top window.
      if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
        quickEntryWindow.close()
      }

      quickEntryWindow = null
    }

    if (state.error === 'taken') {
      rememberLog(`[quick-entry] shortcut ${state.shortcut} is already taken by another application`)
    } else if (state.error === 'invalid') {
      rememberLog(`[quick-entry] shortcut ${state.shortcut} is not a valid accelerator`)
    }

    return { ...state, enabled: settings.enabled }
  }

  function closeQuickEntryWindow() {
    quickEntryShortcut.dispose()

    if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
      quickEntryWindow.close()
    }

    quickEntryWindow = null
  }

  // Quit bypasses the ordinary HUD handoff: restoring the main window here
  // would raise it while the app is already shutting down.
  function closeHudWindowForQuit() {
    hudSnapShortcut.dispose()

    if (hudWindow && !hudWindow.isDestroyed()) {
      hudWindow.removeAllListeners('closed')
      hudWindow.destroy()
    }

    hudWindow = null
  }

  function getHudWindow() {
    return hudWindow
  }

  function setHudSessionId(value: string | null) {
    hudSessionId = value
  }

  function currentQuickEntryShortcutState() {
    return quickEntryShortcut.current()
  }

  function pushQuickEntryState(payload: any) {
    quickEntryLastState = payload ?? null

    if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
      quickEntryWindow.webContents.send('hermes:quick-entry:state', payload)
    }
  }

  return {
    applyQuickEntrySettings,
    closeHudWindow,
    closeHudWindowForQuit,
    closeQuickEntryWindow,
    currentQuickEntryShortcutState,
    getHudWindow,
    hideQuickEntryWindow,
    openHudWindow,
    pushQuickEntryState,
    readQuickEntrySettings,
    resetHudWindowLayout,
    setHudSessionId,
    writeQuickEntrySettings
  }
}
