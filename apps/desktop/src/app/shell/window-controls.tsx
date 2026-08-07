import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $connection } from '@/store/session'

import { titlebarButtonClass } from './titlebar'

// Renderer-drawn min/max/close for 'app-drawn' window chrome (frame: false,
// non-macOS — see computeWindowChromeOptions in electron/main.ts). The cluster
// sits in the exact slot the native Window Controls Overlay would occupy, so
// the titlebar reservation math in contrib/wiring.tsx swaps one width for the
// other.

// Right-edge reservation the system titlebar tools need while the cluster is
// visible: 3 buttons + 2 gaps (gap-x-1) + the cluster's right padding (pr-1.5),
// in the same CSS-var arithmetic wiring.tsx uses for the system cluster.
export const WINDOW_CONTROLS_WIDTH = 'calc(3 * (var(--titlebar-control-size) + 0.25rem) + 0.375rem)'

interface WindowControlsOverlayLike {
  visible: boolean
  getTitlebarAreaRect: () => DOMRect
  addEventListener: (type: 'geometrychange', cb: () => void) => void
  removeEventListener: (type: 'geometrychange', cb: () => void) => void
}

// True when Chromium exposes the Window Controls Overlay API — i.e. Electron's
// WCO is active (native Windows AND plain Linux). Absent on macOS (traffic
// lights are native, never a WCO) and on WSLg without host-drawn controls —
// the exact condition behind the "missing window controls" bug reports.
export function isWindowControlsOverlayAvailable(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    (navigator as Navigator & { windowControlsOverlay?: WindowControlsOverlayLike | null }).windowControlsOverlay !=
      null
  )
}

/**
 * Whether THIS window should draw its own window controls.
 *
 * - The persisted 'app-drawn' setting wins outright.
 * - Otherwise fall back to app-drawn when there is no native controls overlay
 *   at all — but never on macOS, where the traffic lights are native and the
 *   renderer must keep clear of them (windowButtonPosition is non-null there).
 */
export function useAppDrawnChrome(): boolean {
  const connection = useStore($connection)

  if (connection?.windowChromeMode === 'app-drawn') {
    return true
  }

  return !isWindowControlsOverlayAvailable() && connection?.windowButtonPosition === null
}

// Live maximize state so the center button flips maximize ⇄ restore. Rides the
// existing hermes:window-state-changed push (main sends it on maximize,
// unmaximize, minimize, restore, show/hide and fullscreen changes).
export function useWindowMaximized(): boolean {
  const [isMaximized, setIsMaximized] = useState(false)

  useEffect(() => {
    return (
      window.hermesDesktop?.onWindowStateChanged?.(state => {
        setIsMaximized(Boolean(state.isMaximized))
      })
    )
  }, [])

  return isMaximized
}

export function WindowControls() {
  const { t } = useI18n()
  const appDrawn = useAppDrawnChrome()
  const isMaximized = useWindowMaximized()

  if (!appDrawn) {
    return null
  }

  const controls = window.hermesDesktop?.windowControls

  // Chrome-band button styling: icons follow --titlebar-foreground, hovers
  // follow the chrome hover tokens (falling back to the default palette via
  // context.tsx). Close keeps a destructive hover so it reads as dangerous
  // even on a busy band.
  const controlButtonClass = cn(
    titlebarButtonClass,
    'text-(--titlebar-foreground) hover:text-(--titlebar-foreground) hover:bg-(--titlebar-control-hover)'
  )

  const closeButtonClass = cn(controlButtonClass, 'hover:bg-(--titlebar-control-close-hover) hover:text-white')

  return (
    <div
      aria-label={t.shell.windowControls}
      className="fixed right-0 top-0 z-70 flex h-(--titlebar-height) flex-row items-center justify-end gap-x-1 pr-1.5 pointer-events-auto select-none [-webkit-app-region:no-drag]"
    >
      <Tip label={t.shell.windowButtons.minimize}>
        <Button
          aria-label={t.shell.windowButtons.minimize}
          className={controlButtonClass}
          onClick={() => controls?.minimize()}
          onPointerDown={event => event.stopPropagation()}
          size="icon-titlebar"
          type="button"
          variant="ghost"
        >
          <Codicon name="chrome-minimize" />
        </Button>
      </Tip>
      <Tip label={isMaximized ? t.shell.windowButtons.restore : t.shell.windowButtons.maximize}>
        <Button
          aria-label={isMaximized ? t.shell.windowButtons.restore : t.shell.windowButtons.maximize}
          className={controlButtonClass}
          onClick={() => controls?.toggleMaximize()}
          onPointerDown={event => event.stopPropagation()}
          size="icon-titlebar"
          type="button"
          variant="ghost"
        >
          <Codicon name={isMaximized ? 'chrome-restore' : 'chrome-maximize'} />
        </Button>
      </Tip>
      <Tip label={t.shell.windowButtons.close}>
        <Button
          aria-label={t.shell.windowButtons.close}
          className={closeButtonClass}
          onClick={() => controls?.close()}
          onPointerDown={event => event.stopPropagation()}
          size="icon-titlebar"
          type="button"
          variant="ghost"
        >
          <Codicon name="chrome-close" />
        </Button>
      </Tip>
    </div>
  )
}
