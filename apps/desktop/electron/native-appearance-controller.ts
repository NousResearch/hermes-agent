import fs from 'node:fs'
import path from 'node:path'

import type { BrowserWindow, NativeTheme } from 'electron'

import { titleBarOverlayOptions } from './titlebar-overlay-width'
import {
  backgroundMaterialFor,
  defaultTranslucencyState,
  glassActive,
  normalizeState as normalizeTranslucency,
  opacityNeedsSetting,
  vibrancyFor as vibrancyForTranslucency,
  windowBackingOptions,
  windowOpacityFor,
  windowOpacityOptions
} from './translucency'

interface NativeAppearanceOptions {
  userDataDir: string
  nativeTheme: NativeTheme
  getAllWindows: () => BrowserWindow[]
  log: (message: string) => void
  isMac: boolean
  isWindows: boolean
  isWsl: boolean
  darwinMajor: number
  glassSupported: boolean
  titlebarHeight: number
}

export function createNativeAppearanceController({
  userDataDir,
  nativeTheme,
  getAllWindows,
  log: rememberLog,
  isMac: IS_MAC,
  isWindows: IS_WINDOWS,
  isWsl: IS_WSL,
  darwinMajor: DARWIN_MAJOR,
  glassSupported: GLASS_SUPPORTED,
  titlebarHeight: TITLEBAR_HEIGHT
}: NativeAppearanceOptions) {
  let rendererTitleBarTheme = null

  // Force the NATIVE window appearance (vibrancy material, titlebar, the
  // pre-first-paint window background) to follow the APP theme instead of the
  // OS appearance. With `vibrancy` set, macOS paints an NSVisualEffectView that
  // tracks the window's effective appearance and ignores `backgroundColor` —
  // so a dark-themed app on a light-mode Mac flashes a white material on every
  // new window until the renderer covers it. The renderer reports its mode via
  // 'hermes:native-theme' ('dark' | 'light' | 'system'); we pin
  // nativeTheme.themeSource to it and persist the value so cold launches paint
  // correctly before the renderer has even loaded.
  const NATIVE_THEME_CONFIG_PATH = path.join(userDataDir, 'native-theme.json')
  const THEME_SOURCES = new Set(['dark', 'light', 'system'])

  function readPersistedThemeSource() {
    try {
      const parsed = JSON.parse(fs.readFileSync(NATIVE_THEME_CONFIG_PATH, 'utf8'))

      if (parsed && THEME_SOURCES.has(parsed.themeSource)) {
        return parsed.themeSource
      }
    } catch {
      // Missing / malformed → follow the OS like a fresh install.
    }

    return 'system'
  }

  function writePersistedThemeSource(mode) {
    try {
      fs.mkdirSync(path.dirname(NATIVE_THEME_CONFIG_PATH), { recursive: true })
      fs.writeFileSync(NATIVE_THEME_CONFIG_PATH, JSON.stringify({ themeSource: mode }, null, 2), 'utf8')
    } catch (error) {
      rememberLog(`[theme] write native theme failed: ${error.message}`)
    }
  }

  nativeTheme.themeSource = readPersistedThemeSource()

  // Window translucency (see-through window). One lever, 0–100; 0 = off (the
  // default). Two modes share the lever (see electron/translucency.ts and
  // store/translucency): 'clear' maps it to the native window opacity so the
  // desktop shows through the whole window; 'glass' keeps the window opaque
  // and lets the renderer thin its surfaces over a platform material instead
  // — a matte blur with full-contrast text. macOS uses vibrancy; Windows 11
  // uses DWM acrylic/mica/tabbed. Persisted so a cold launch applies it at
  // window creation, before the renderer reports its value.
  // macOS + Windows only; `setOpacity` is a no-op on Linux.
  const TRANSLUCENCY_CONFIG_PATH = path.join(userDataDir, 'translucency.json')

  function readPersistedTranslucency() {
    try {
      return normalizeTranslucency(JSON.parse(fs.readFileSync(TRANSLUCENCY_CONFIG_PATH, 'utf8')), GLASS_SUPPORTED)
    } catch {
      // Nothing persisted yet — a first launch. Glass ships on, so the FIRST
      // window has to be created with the glass backing already: a window born
      // opaque cannot reliably be swapped to glass afterwards (see
      // windowBackingOptions). nativeTheme is the only appearance signal main
      // has this early; the renderer's first resolved send corrects it.
      return defaultTranslucencyState(nativeTheme.shouldUseDarkColors ? 'dark' : 'light', GLASS_SUPPORTED, IS_WINDOWS)
    }
  }

  function writePersistedTranslucency(state) {
    try {
      fs.mkdirSync(path.dirname(TRANSLUCENCY_CONFIG_PATH), { recursive: true })
      fs.writeFileSync(TRANSLUCENCY_CONFIG_PATH, JSON.stringify(state, null, 2), 'utf8')
    } catch (error) {
      rememberLog(`[translucency] write failed: ${error.message}`)
    }
  }

  let translucencyState = readPersistedTranslucency()

  // Chat windows whose webContents backing follows translucency (primary,
  // instance peers, session windows). The HUD / pet overlay / quick entry /
  // wake indicator are `transparent: true` windows that own their backgrounds —
  // painting a themed backing onto them would turn them into opaque rectangles.
  const translucencyBackedWindows = new WeakSet()

  // Set a live window's native opacity, but only when the state asks it to fade
  // — or when the window is already faded and is on its way back to opaque. The
  // window's own opacity is the record of whether that door was ever opened; see
  // opacityNeedsSetting for why it matters that it stays shut.
  function applyWindowOpacity(win) {
    const opacity = windowOpacityFor(translucencyState)

    if (typeof win.setOpacity === 'function' && opacityNeedsSetting(opacity, win.getOpacity?.() ?? 1)) {
      win.setOpacity(opacity)
    }
  }

  // Re-apply translucency to a live window (runtime toggle, no recreation).
  // Opacity goes through applyWindowOpacity, which knows when the call is worth
  // making at all. The backing swap is the glass half: Chromium composites the
  // page against the window backing BEFORE the OS composites the window, so
  // glass needs the backing dropped for the platform material to reach it, and
  // every other state needs the opaque themed backing (anti-flash, and it is
  // what makes clear mode fade to the desktop instead of to black).
  //
  // `changed` says which native properties actually need touching. Dragging the
  // intensity slider emits ~100 updates, and in glass mode NONE of them change
  // anything native — the tint is painted by the renderer and windowOpacityFor
  // answers off `fade`, not `intensity`, there. Re-issuing setVibrancy on every
  // tick restarts its 150ms animation before macOS can settle the material,
  // which reads as jank and flattens the frost levels into each other. Windows
  // setBackgroundMaterial is instantaneous but still skipped on tint-only ticks.
  // The glass Fade lever is the one glass drag that does reach main, and it
  // costs exactly what a Clear drag costs: one setOpacity.
  //
  // CAUTION (measured, macOS 26 / Electron 40): a runtime
  // setBackgroundColor('#00000000') is silently LOST on a window whose
  // compositor hasn't been up for a few seconds — including calls from
  // 'ready-to-show' and 'did-finish-load'. Cold launches therefore must not
  // rely on this path: windows are BORN with the right backing
  // (windowBackingOptions at each creation site). This path only has to cover
  // live toggles from Settings, where the window is long settled.
  function applyWindowTranslucency(win, changed = { backing: true, material: true, opacity: true }) {
    if (!win || win.isDestroyed()) {
      return
    }

    try {
      // Backing swap + material are scoped to registered chat windows (see
      // translucencyBackedWindows above).
      if (translucencyBackedWindows.has(win)) {
        if (changed.backing && typeof win.setBackgroundColor === 'function') {
          win.setBackgroundColor(glassActive(translucencyState) ? '#00000000' : getWindowBackgroundColor())
        }

        if (changed.material) {
          // Glass frost level = the platform material. Animate the macOS hop so
          // a deliberate frost switch feels continuous — which only works if we
          // don't re-issue it on unrelated updates. Windows has no equivalent
          // animation option; setBackgroundMaterial is instantaneous.
          if (IS_MAC && typeof win.setVibrancy === 'function') {
            win.setVibrancy(vibrancyForTranslucency(translucencyState), { animationDuration: 150 })
          }

          if (IS_WINDOWS && GLASS_SUPPORTED && typeof win.setBackgroundMaterial === 'function') {
            win.setBackgroundMaterial(backgroundMaterialFor(translucencyState))
          }
        }
      }

      if (changed.opacity) {
        applyWindowOpacity(win)
      }
    } catch (error) {
      rememberLog(`[translucency] apply failed: ${error.message}`)
    }
  }

  // Constructor options every chat window shares for its translucency surface:
  // the platform material, the webContents backing, and a native opacity only if
  // the state actually fades — all under the CURRENT state. Glass omits
  // backgroundColor so the material shows from the first frame (Electron hands a
  // translucent window a transparent default backing, and runtime swaps are lost
  // early in a window's life — see applyWindowTranslucency); otherwise the opaque
  // themed anti-flash backing.
  //
  // Call sites also register the window in translucencyBackedWindows so a live
  // toggle can re-apply. The HUD, pet overlay, quick entry and wake indicator
  // are `transparent: true` windows that own their backgrounds and are
  // deliberately not chat windows.
  function chatWindowSurfaceOptions() {
    return {
      vibrancy: IS_MAC ? vibrancyForTranslucency(translucencyState) : undefined,
      // Pin the material to its ACTIVE appearance: several NSVisualEffectView
      // materials collapse to a shared inactive look when the window blurs
      // (measured on macOS 26: sidebar, popover and under-window composited
      // pixel-identically once unfocused), which would quietly erase the
      // user's frost choice whenever they click elsewhere. Only observable
      // under glass — everywhere else the page buries the material.
      visualEffectState: IS_MAC ? ('active' as const) : undefined,
      // NOT `transparent: true` on Windows. The backdrop material already makes
      // the window translucent on its own: `IsTranslucent` answers yes off
      // `background_material_` alone, which is what gives the page its transparent
      // default backing, and `SetBackgroundMaterial` flips widget translucency
      // live, so a Clear→Glass toggle needs no recreate either way. Its one gate
      // is a frameless window, and `titleBarStyle: 'hidden'` already makes
      // `has_frame()` false here.
      //
      // What `transparent` adds on top is permanent and unwanted: it pins the
      // widget to kTranslucent for the window's whole life, so even glass-OFF
      // windows pay a DirectComposition redraw per frame (electron#39895), and it
      // opts into the documented transparent-window limits — including that a
      // RESIZABLE transparent window is unsupported and breaks (electron#48421).
      // Every chat window is resizable.
      backgroundMaterial: IS_WINDOWS && GLASS_SUPPORTED ? backgroundMaterialFor(translucencyState) : undefined,
      ...windowOpacityOptions(translucencyState),
      ...windowBackingOptions(translucencyState, getWindowBackgroundColor())
    }
  }

  function isHexColor(value) {
    return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value)
  }

  // Background color to paint a window with BEFORE its renderer loads, so a new
  // (or reopened) window doesn't flash white/light in dark mode. Prefer the theme
  // the renderer last reported; fall back to the OS preference on first launch.
  function getWindowBackgroundColor() {
    if (rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.background)) {
      return rendererTitleBarTheme.background
    }

    return nativeTheme.shouldUseDarkColors ? '#111111' : '#f7f7f7'
  }

  // Transparent WCO — renderer chrome shows through. rgba(0,0,0,0) can fall back
  // to GetFrameColor() on some Electron builds; rgba(1,0,0,0) is the escape hatch.
  const TITLEBAR_OVERLAY_COLOR = 'rgba(1, 0, 0, 0)'

  // WSLg returns false: the RDP host paints nothing for a frameless window and
  // Electron's own overlay drifts its hit-region under RAIL, so the renderer
  // paints its own min/max/close (wslg-window-controls.tsx) over the
  // hermes:window-control IPC channel. See titleBarOverlayOptions.
  function getTitleBarOverlayOptions() {
    return titleBarOverlayOptions({
      platform: IS_MAC ? 'mac' : IS_WINDOWS ? 'windows' : IS_WSL ? 'wslg' : 'linux',
      darwinMajor: DARWIN_MAJOR,
      titlebarHeight: TITLEBAR_HEIGHT,
      color: TITLEBAR_OVERLAY_COLOR,
      foreground:
        rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.foreground) ? rendererTitleBarTheme.foreground : null,
      dark: nativeTheme.shouldUseDarkColors
    })
  }

  // Push refreshed overlay options to a live window after a theme/appearance
  // change. No-op only on plain (non-WSL) Linux, where getTitleBarOverlayOptions()
  // returns false; the try/catch additionally guards builds where
  // setTitleBarOverlay isn't supported.
  function applyTitleBarOverlay(win) {
    const options = getTitleBarOverlayOptions()

    if (!options || typeof options !== 'object') {
      return
    }

    try {
      win?.setTitleBarOverlay?.(options)
    } catch {
      // Overlay not supported on this platform/build — leave the frameless
      // titlebar as-is.
    }
  }

  let nativeThemeListenerInstalled = false
  let translucencyWriteTimer: ReturnType<typeof setTimeout> | null = null

  // The intensity slider is a hot path. Coalesce the synchronous persistence
  // write while applying only the native properties whose values changed.
  function scheduleTranslucencyWrite() {
    if (translucencyWriteTimer) {
      clearTimeout(translucencyWriteTimer)
    }

    translucencyWriteTimer = setTimeout(() => {
      translucencyWriteTimer = null
      writePersistedTranslucency(translucencyState)
    }, 250)
  }

  // Flush before quit so a setting changed during the debounce window survives.
  function flushTranslucencyWrite() {
    if (translucencyWriteTimer) {
      clearTimeout(translucencyWriteTimer)
      translucencyWriteTimer = null
      writePersistedTranslucency(translucencyState)
    }
  }

  function setTitleBarTheme(payload) {
    if (!payload || !isHexColor(payload.background) || !isHexColor(payload.foreground)) {
      return
    }

    rendererTitleBarTheme = {
      background: payload.background,
      foreground: payload.foreground
    }

    // All open chat windows share the app theme. The overlay helper no-ops on
    // platforms without a native overlay.
    for (const win of getAllWindows()) {
      applyTitleBarOverlay(win)
    }
  }

  function setNativeTheme(mode) {
    if (!THEME_SOURCES.has(mode)) {
      return
    }

    if (nativeTheme.themeSource !== mode) {
      nativeTheme.themeSource = mode
      writePersistedThemeSource(mode)
    }
  }

  function setTranslucency(payload, onChanged: () => void) {
    const next = normalizeTranslucency(payload, GLASS_SUPPORTED)
    const previous = translucencyState

    if (
      next.intensity === previous.intensity &&
      next.fade === previous.fade &&
      next.mode === previous.mode &&
      next.material === previous.material &&
      next.scope === previous.scope
    ) {
      return
    }

    translucencyState = next

    // Which native properties actually moved. scope is renderer-only.
    const changed = {
      backing: glassActive(previous) !== glassActive(next),
      material: vibrancyForTranslucency(previous) !== vibrancyForTranslucency(next),
      opacity: windowOpacityFor(previous) !== windowOpacityFor(next)
    }

    scheduleTranslucencyWrite()

    // The HUD is a transparent window with its own frost. It is intentionally
    // outside the chat backing fan-out and self-diffs in its own controller.
    onChanged()

    if (changed.backing || changed.material || changed.opacity) {
      for (const win of getAllWindows()) {
        applyWindowTranslucency(win, changed)
      }
    }
  }

  function installNativeThemeListener() {
    if (!nativeThemeListenerInstalled) {
      nativeThemeListenerInstalled = true
      nativeTheme.on('updated', () => {
        for (const win of getAllWindows()) {
          applyTitleBarOverlay(win)
        }
      })
    }
  }

  return {
    chatWindowSurfaceOptions,
    flushTranslucencyWrite,
    getTitleBarOverlayOptions,
    getTranslucencyState: () => translucencyState,
    installNativeThemeListener,
    registerChatWindow: (win: BrowserWindow) => translucencyBackedWindows.add(win),
    setNativeTheme,
    setTitleBarTheme,
    setTranslucency
  }
}
