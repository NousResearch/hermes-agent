import type { CSSProperties, PointerEvent } from 'react'
import { useLocation } from 'react-router-dom'

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import { appViewForPath, isOverlayView } from '../routes'

import { TITLEBAR_HEIGHT } from './titlebar'

interface WslgWindowControlsProps {
  isFullscreen: boolean
  isMaximized: boolean
}

// Full-height caption buttons sized to match the native Windows cluster
// (~46px wide × full titlebar height). `h-full` fills the cluster box, whose
// height is pinned to TITLEBAR_HEIGHT below — NOT var(--titlebar-height), which
// the contrib shell zeroes for content subtrees (controller.tsx), collapsing
// the buttons if inherited.
const buttonClass =
  'grid h-full w-[46px] place-items-center border-0 bg-transparent p-0 text-muted-foreground transition-colors duration-75 select-none [-webkit-app-region:no-drag] focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring hover:bg-white/10 hover:text-foreground active:bg-white/15'

// Match the native titlebar tools: stopPropagation (NOT preventDefault) on
// pointerdown. preventDefault on pointerdown suppresses the synthesized click
// under WSLg's XWayland/RAIL compositor, so the buttons render but never fire.
// stopPropagation keeps the drag region from swallowing the press while leaving
// the click intact; keyboard-focus reassertion after maximize is handled on the
// main side in performWindowControl (win.focus()).
const stopTitlebarDrag = (event: PointerEvent<HTMLButtonElement>) => event.stopPropagation()

export function WslgWindowControls({ isFullscreen, isMaximized }: WslgWindowControlsProps) {
  const location = useLocation()
  const controls = window.hermesDesktop?.windowControls

  if (!controls || isFullscreen || isOverlayView(appViewForPath(location.pathname))) {
    return null
  }

  return (
    <div
      aria-label="Window controls"
      className="fixed right-0 top-0 z-80 flex items-stretch overflow-hidden bg-(--ui-chat-surface-background) text-[10px]"
      // Pin the real titlebar height: the shared --titlebar-height var is
      // contextually zeroed inside the contrib shell, so read the constant.
      style={{ height: `${TITLEBAR_HEIGHT}px` } as CSSProperties}
    >
      <button
        aria-label="Minimize window"
        className={buttonClass}
        onClick={controls.minimize}
        onPointerDown={stopTitlebarDrag}
        type="button"
      >
        <Codicon name="chrome-minimize" size={10} />
      </button>
      <button
        aria-label={isMaximized ? 'Restore window' : 'Maximize window'}
        className={buttonClass}
        onClick={controls.toggleMaximize}
        onPointerDown={stopTitlebarDrag}
        type="button"
      >
        <Codicon name={isMaximized ? 'chrome-restore' : 'chrome-maximize'} size={10} />
      </button>
      <button
        aria-label="Close window"
        className={cn(
          buttonClass,
          'hover:bg-[#c42b1c] hover:text-white active:bg-[#b3271a] active:text-white dark:hover:bg-[#c42b1c]'
        )}
        onClick={controls.close}
        onPointerDown={stopTitlebarDrag}
        type="button"
      >
        <Codicon name="chrome-close" size={10} />
      </button>
    </div>
  )
}
