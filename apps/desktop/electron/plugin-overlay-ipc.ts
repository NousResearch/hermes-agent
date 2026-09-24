// IPC surface for the plugin overlay window — a generic, transparent,
// always-on-top host for plugin contributions that want to leave the app
// window (the Dash mascot card is the first user). The window hosts ONE
// contribution: whichever `area: 'pluginOverlay'` contribution its plugin
// registered in the overlay renderer. Window handles stay injected because
// main.ts owns their lifecycle (same pattern as pet-overlay-ipc.ts).
//
// PRODUCT POLICY — one overlay at a time (declared, not accidental):
// the app hosts a SINGLE overlay window. Opening plugin B while plugin A is
// hosted closes A's window and respawns for B — A's renderer state (in-flight
// input, scroll) is discarded. Persistence is already per-plugin
// (plugin-overlay-state.json is keyed by pluginId), and the IPC deps are
// keyed by plugin id, so a future Map<pluginId, BrowserWindow> refactor for
// concurrent overlays is mechanical.
//
// TRUST POSTURE: `open`/`close` are accepted from the MAIN window's renderer
// (where plugins run and ctx.os.openOverlay lives) and — for close — from the
// overlay window itself (a pop-in button). Every other channel is gated to
// the overlay window's webContents. A plugin cannot open or evict another
// plugin's overlay from an arbitrary window.
import { type BrowserWindow, ipcMain } from 'electron'

import { applyBoundsWithResizeFlip, type ResizeFlipWindow } from './resize-flip'

export type { PluginOverlayBounds } from './plugin-overlay-geometry'
import type { PluginOverlayBounds } from './plugin-overlay-geometry'

export interface PluginOverlayOpenRequest {
  pluginId: string
  /** The posture the overlay should take: 'mascot' (non-activating sprite
   *  pinned to the app window — Dash's auto-start) or 'card' (interactive
   *  Q&A). Defaults to 'card'. */
  mode?: PluginOverlayMode
  /** Optional bounds. `screen: false` (default) means VIEWPORT space: the
   *  pop-out button passes the pane's in-window rect and main converts it to
   *  screen space via the main window's content origin (pet overlay parity).
   *  `screen: true` means screen space, used as-is. */
  bounds?: PluginOverlayBounds
  screen?: boolean
}

export type PluginOverlayMode = 'mascot' | 'card'

export interface PluginOverlayIpcDeps {
  /** The main app window (viewport→screen conversion + close broadcast). */
  getMainWindow: () => BrowserWindow | null
  /** The overlay window, or null while closed. */
  getOverlayWindow: () => BrowserWindow | null
  /** The plugin id the current overlay window hosts, or null. */
  getOverlayPluginId: () => string | null
  /** The overlay's current posture: mascot (non-activating sprite pinned to
   *  the app window) or card (interactive Q&A card). */
  getOverlayMode: () => PluginOverlayMode | null
  /** Spawn (or re-show at bounds) the overlay window for a plugin id.
   *  Returns the window, or null when spawning failed. */
  openOverlay: (pluginId: string, mode: PluginOverlayMode, bounds?: PluginOverlayBounds) => BrowserWindow | null
  closeOverlay: () => void
  /** Flip the EXISTING window's posture. Main owns the geometry switch —
   *  the renderer only says which posture it needs. */
  setOverlayMode: (mode: PluginOverlayMode) => void
  /** Posture-aware bounds enforcement for renderer-reported geometry: the
   *  mascot clamps into the main window's rect at fixed size; the card
   *  clamps to minimums. Main implements both. */
  snapBounds: (bounds: PluginOverlayBounds) => PluginOverlayBounds
  /** Apply (or clear, with []) the window's X11 shape. Main owns the
   *  resizable-flip dance; the IPC handler only validates the rects. */
  setShape: (rects: Array<{ x: number; y: number; width: number; height: number }>) => void
  /** Persist the reported bounds to disk (main-side, userData), keyed by the
   *  CURRENTLY HOSTED plugin (main's own latch — the renderer-supplied id is
   *  never trusted for persistence). */
  persistBounds: (pluginId: string, bounds: PluginOverlayBounds) => void
  /** Read persisted bounds for a plugin id + posture (may be null). */
  readBounds: (pluginId: string, mode: PluginOverlayMode) => PluginOverlayBounds | null
}

export function registerPluginOverlayIpc({
  getMainWindow,
  getOverlayWindow,
  getOverlayPluginId,
  getOverlayMode,
  openOverlay,
  closeOverlay,
  setOverlayMode,
  snapBounds,
  setShape,
  persistBounds,
  readBounds
}: PluginOverlayIpcDeps): void {
  const overlayOwns = (event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent): boolean => {
    const win = getOverlayWindow()

    return Boolean(win && !win.isDestroyed() && event.sender === win.webContents)
  }

  const mainOwns = (event: Electron.IpcMainEvent | Electron.IpcMainInvokeEvent): boolean => {
    const win = getMainWindow()

    return Boolean(win && !win.isDestroyed() && event.sender === win.webContents)
  }

  // The plugin host window asks for a plugin overlay. OPEN comes from the MAIN
  // renderer (plugin's ctx.os.openOverlay). If a window for this plugin
  // already exists, re-show it at the requested bounds and reuse it — one
  // overlay per plugin (see PRODUCT POLICY above).
  ipcMain.handle('hermes:plugin-overlay:open', async (event, request: PluginOverlayOpenRequest) => {
    if (!mainOwns(event)) {
      return { ok: false }
    }

    const pluginId = String(request?.pluginId || '').trim()

    if (!pluginId) {
      return { ok: false }
    }

    // Posture: 'mascot' (non-activating sprite, auto-on-register) or 'card'
    // (interactive Q&A). Unknown → card (the safe default: an unexpected
    // mascot spawn is just a small sprite, an unexpected card is a full UI).
    const mode: PluginOverlayMode = request?.mode === 'mascot' ? 'mascot' : 'card'

    // Viewport→screen conversion (pet-overlay-ipc.ts parity): a fresh pop-out
    // passes the pane's in-window rect; add the main window's content origin
    // so the overlay lands where it sat in-window. Screen-space requests
    // (remembered/dragged spots) are used as-is.
    let screenBounds: PluginOverlayBounds | undefined = request?.bounds

    if (screenBounds && !request?.screen) {
      const mainWindow = getMainWindow()

      try {
        if (mainWindow && !mainWindow.isDestroyed()) {
          const content = mainWindow.getContentBounds()
          screenBounds = {
            x: content.x + (screenBounds.x || 0),
            y: content.y + (screenBounds.y || 0),
            width: screenBounds.width,
            height: screenBounds.height
          }
        }
      } catch {
        // Fall back to raw bounds if the window geometry is unavailable.
      }
    }

    try {
      const win = openOverlay(pluginId, mode, screenBounds)

      return { ok: Boolean(win) }
    } catch {
      // Spawn failure must be observable — the PluginOs contract resolves a
      // result instead of throwing, and an unconditional { ok: true } lied.
      return { ok: false }
    }
  })

  ipcMain.handle('hermes:plugin-overlay:close', async event => {
    // Main renderer (ctx.os.closeOverlay) OR the overlay itself (pop-in
    // button) may close. Any other sender is refused.
    if (!mainOwns(event) && !overlayOwns(event)) {
      return { ok: false }
    }

    closeOverlay()

    return { ok: true }
  })

  // The OVERLAY window reports its own content geometry. TWO channels, one
  // writer per event (the pet overlay's contract, verbatim):
  //   set-bounds     → TRANSIENT: live drag/resize — main snaps the window,
  //                    never persists.
  //   report-bounds  → DURABLE: drag/resize END — main snaps + persists under
  //                    its own hosted-plugin latch.
  // The overlay renderer is the geometry authority: it knows its content and
  // drives the drag/resize gestures.
  ipcMain.on('hermes:plugin-overlay:set-bounds', (event, payload) => {
    const win = getOverlayWindow()

    if (!win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    const bounds = payload?.bounds as PluginOverlayBounds | undefined

    if (!bounds || ![bounds.x, bounds.y, bounds.width, bounds.height].every(Number.isFinite)) {
      return
    }

    // Posture-aware snap: mascot → fixed size clamped into the app window;
    // card → min-clamped free bounds. Main is the geometry authority.
    applyBoundsWithResizeFlip(win as ResizeFlipWindow, snapBounds(bounds))
  })

  ipcMain.on('hermes:plugin-overlay:report-bounds', (event, payload) => {
    const win = getOverlayWindow()

    if (!win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    const bounds = payload?.bounds as PluginOverlayBounds | undefined

    if (!bounds || ![bounds.x, bounds.y, bounds.width, bounds.height].every(Number.isFinite)) {
      return
    }

    const next = snapBounds(bounds)

    applyBoundsWithResizeFlip(win as ResizeFlipWindow, next)

    // Persist under main's OWN latch — a renderer-supplied pluginId is a
    // spoofable key; main already holds the authoritative hosted id.
    const hostedId = getOverlayPluginId()

    if (hostedId) {
      persistBounds(hostedId, next)
    }
  })

  // WINDOW SHAPE — the no-compositor transparency path. Without a running
  // compositor, Chromium creates `transparent: true` windows with a 24-bit
  // visual (no alpha channel), so the transparent background renders BLACK.
  // The X11 SHAPE extension clips a window server-side to a set of rects —
  // no alpha needed. A mascot contribution reports its opaque sprite regions
  // here; main carves the window so the desktop shows through everywhere
  // else. The card posture clears the shape (full window, opaque surface).
  ipcMain.on('hermes:plugin-overlay:set-shape', (event, payload) => {
    const win = getOverlayWindow()

    if (!win || win.isDestroyed() || event.sender !== win.webContents) {
      return
    }

    const raw = Array.isArray(payload?.rects) ? payload.rects : null

    if (!raw) {
      return
    }

    // Validate + round: integers only (X rects), non-negative, bounded count.
    interface ShapeRectCandidate {
      x?: unknown
      y?: unknown
      width?: unknown
      height?: unknown
    }

    const candidates = (raw as ShapeRectCandidate[]).slice(0, 64)

    const rects: Array<{ x: number; y: number; width: number; height: number }> = []

    for (const r of candidates) {
      if (!r || typeof r !== 'object') {
        continue
      }

      const { x, y, width, height } = r

      if (![x, y, width, height].every(n => typeof n === 'number' && Number.isFinite(n))) {
        continue
      }

      const rx = Math.max(0, Math.round(x as number))
      const ry = Math.max(0, Math.round(y as number))
      const rw = Math.max(0, Math.round(width as number))
      const rh = Math.max(0, Math.round(height as number))

      if (rw > 0 && rh > 0) {
        rects.push({ x: rx, y: ry, width: rw, height: rh })
      }
    }

    // setShape on Linux requires a resizable window (Electron quirk) — main
    // owns the flip + the reveal gate.
    setShape(rects)
  })

  // The overlay renderer flips its own posture: a click on the mascot asks
  // main to expand it into the interactive card, the card's ✕ asks to
  // shrink back. Main owns the geometry switch (focusable/always-on-top/
  // size), the renderer only states which posture it needs.
  ipcMain.on('hermes:plugin-overlay:set-mode', (event, mode) => {
    if (!overlayOwns(event)) {
      return
    }

    if (mode !== 'mascot' && mode !== 'card') {
      return
    }

    setOverlayMode(mode)
  })

  // The overlay asks which plugin it was spawned for + its current posture
  // and remembered bounds.
  ipcMain.handle('hermes:plugin-overlay:whoami', async event => {
    if (!overlayOwns(event)) {
      return { pluginId: null, mode: null, bounds: null }
    }

    const pluginId = getOverlayPluginId()
    const mode = getOverlayMode()

    return {
      pluginId,
      mode,
      bounds: pluginId && mode ? readBounds(pluginId, mode) : null
    }
  })

  // Close notification (pet pop-in parity): when the overlay goes away on its
  // own (evicted by another plugin's open, ⌘W, crash), tell the main renderer
  // so a pop-out toggle never stays stale. main.ts's 'closed' handler calls
  // this via the deps.
}
