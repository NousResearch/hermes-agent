/**
 * `host.window` — the typed door for the window-level appearance channels.
 *
 * Three surfaces on Settings → Appearance are not renderer preferences: they
 * drive the main process or the window chrome, so they cannot live in the
 * `host.settings` key table (that table is desktop-wide and renderer-owned).
 * Today a plugin reaches them through `window.hermesDesktop.*` — raw preload
 * IPC, which the plugin rules do not allow and which breaks when the bridge
 * changes shape.
 *
 * Every method here wraps the channel the native Settings page already uses,
 * so a plugin write is the same call the Appearance page makes: no new reach,
 * no new authority.
 *
 * Scope, stated once because it differs from `host.settings`:
 *
 * - **zoom** is per window. The bridge reports the window the renderer is in,
 *   and a change applies to that window only — same as the ⌘+/⌘− shortcuts.
 * - **translucency** is a renderer-owned book, desktop-wide, mirrored to the
 *   main process. The store is the source of truth; `set` patches it field by
 *   field through the same setters the Settings page uses.
 * - **minimizeToTray** is a machine-level preference owned by the main
 *   process. `status.available` reports whether the OS backs a tray at all;
 *   note the trap the native page also documents: the tray only exists *after*
 *   enabling, so gating a control on `available` alone can deadlock — gate on
 *   `available || enabled`.
 */

import type { TranslucencyState } from '@hermes/shared/translucency'

import {
  $translucency,
  setTranslucency,
  setTranslucencyFade,
  setTranslucencyMaterial,
  setTranslucencyMode,
  setTranslucencyScope
} from '@/store/translucency'

/** The zoom of the window this renderer is in. */
export interface PluginZoomState {
  /** Electron zoom level (-1, 0, 1, …). */
  level: number
  /** The same zoom as a percentage (100 = default). */
  percent: number
}

export interface PluginTrayStatus {
  enabled: boolean
  available: boolean
}

type DesktopBridge = NonNullable<Window['hermesDesktop']>

const bridge = (): DesktopBridge | undefined =>
  typeof window === 'undefined' ? undefined : (window.hermesDesktop as DesktopBridge | undefined)

const zoomBridge = (): DesktopBridge['zoom'] | undefined => bridge()?.zoom

const trayBridge = (): DesktopBridge['minimizeToTray'] | undefined => bridge()?.minimizeToTray

export const hostWindow = {
  zoom: {
    /** False on a desktop build that predates the zoom channel. */
    available(): boolean {
      return typeof zoomBridge()?.get === 'function'
    },

    /** Current zoom of this window, or null when the build has no bridge. */
    async get(): Promise<null | PluginZoomState> {
      const zoom = zoomBridge()

      if (typeof zoom?.get !== 'function') {
        return null
      }

      return zoom.get()
    },

    /** Set this window's zoom. No-op when the build has no bridge. */
    setPercent(percent: number): void {
      const zoom = zoomBridge()

      if (typeof zoom?.setPercent !== 'function') {
        return
      }

      zoom.setPercent(percent)
    },

    /**
     * Follow every zoom change in this window — the shortcuts, the View menu,
     * the Settings page and this door. Emits nothing on a build without the
     * bridge, so a caller can register unconditionally.
     */
    subscribe(listener: (state: PluginZoomState) => void): () => void {
      const zoom = zoomBridge()

      if (typeof zoom?.onChanged !== 'function') {
        return () => {}
      }

      return zoom.onChanged(listener)
    }
  },

  translucency: {
    /** The resolved state for the appearance currently painted. */
    get(): TranslucencyState {
      return $translucency.get()
    },

    /**
     * Patch the book through the native setters — intensity and fade are
     * clamped to the same band the Settings page uses, mode/material/scope
     * are normalized to their enums. Fields left out are untouched.
     */
    set(patch: Partial<TranslucencyState>): void {
      if (patch.mode !== undefined) {
        setTranslucencyMode(patch.mode)
      }

      if (patch.material !== undefined) {
        setTranslucencyMaterial(patch.material)
      }

      if (patch.scope !== undefined) {
        setTranslucencyScope(patch.scope)
      }

      if (patch.intensity !== undefined) {
        setTranslucency(patch.intensity)
      }

      if (patch.fade !== undefined) {
        setTranslucencyFade(patch.fade)
      }
    },

    /** Emits the current state now, then after every native or plugin write. */
    subscribe(listener: (state: TranslucencyState) => void): () => void {
      // nanostores hands its listener (value, oldValue, changed); a plugin
      // contract stays one-argument.
      return $translucency.subscribe(state => listener(state))
    }
  },

  minimizeToTray: {
    /** False on a build without the channel or on a platform with no tray. */
    available(): boolean {
      return typeof trayBridge()?.get === 'function'
    },

    /** Current status, or null when the build has no bridge. */
    async get(): Promise<null | PluginTrayStatus> {
      const tray = trayBridge()

      if (typeof tray?.get !== 'function') {
        return null
      }

      return tray.get()
    },

    /**
     * Turn minimize-to-tray on or off. Resolves with the resulting status
     * (the main process may refuse on a platform with no tray), or null when
     * the build has no bridge.
     */
    async set(on: boolean): Promise<null | PluginTrayStatus> {
      const tray = trayBridge()

      if (typeof tray?.set !== 'function') {
        return null
      }

      return tray.set(on)
    },

    /** Follow changes from the Settings page, the tray menu and this door. */
    subscribe(listener: (status: PluginTrayStatus) => void): () => void {
      const tray = trayBridge()

      if (typeof tray?.onChanged !== 'function') {
        return () => {}
      }

      return tray.onChanged(listener)
    }
  }
}
