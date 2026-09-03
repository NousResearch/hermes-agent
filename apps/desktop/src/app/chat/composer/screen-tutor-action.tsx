import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import {
  $screenTutor,
  dismissScreenAnnotations,
  toggleScreenAnnotationsFrozen,
  toggleScreenTutor
} from '@/store/screen-tutor'

import { ACTIVE_ICON_BTN, GHOST_ICON_BTN } from './control-classes'

export function ScreenTutorAction({ disabled, target }: { disabled: boolean; target: string }) {
  const state = useStore($screenTutor)
  const available = Boolean(window.hermesDesktop?.screenTutor?.capture)
  const active = state.armedTarget === target
  const capturing = state.status === 'capturing'

  if (!available) {
    return null
  }

  const label = state.error
    ? `Screen Tutor capture failed: ${state.error}`
    : active
      ? 'Screen annotations armed — the next send analyzes one fresh screenshot'
      : 'Screen annotations — analyze and mark up the current display'

  return (
    <div className="flex items-center gap-0.5">
      <Tip label={label} side="top">
        <Button
          aria-label={label}
          aria-pressed={active}
          className={cn(active ? ACTIVE_ICON_BTN : GHOST_ICON_BTN, 'relative')}
          disabled={disabled || capturing}
          onClick={() => toggleScreenTutor(target)}
          type="button"
          variant="ghost"
        >
          <Codicon className={capturing ? 'animate-pulse' : undefined} name="target" />
          {(active || state.overlay.visible) && (
            <span aria-hidden className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-cyan-300" />
          )}
        </Button>
      </Tip>
      {state.overlay.visible && (
        <>
          <Tip label={state.overlay.frozen ? 'Resume annotation timer' : 'Keep annotations on screen'} side="top">
            <Button
              aria-label={state.overlay.frozen ? 'Resume annotation timer' : 'Keep annotations on screen'}
              aria-pressed={state.overlay.frozen}
              className={cn(state.overlay.frozen ? ACTIVE_ICON_BTN : GHOST_ICON_BTN, 'size-6')}
              onClick={toggleScreenAnnotationsFrozen}
              type="button"
              variant="ghost"
            >
              <Codicon name="pin" />
            </Button>
          </Tip>
          <Tip label="Clear screen annotations" side="top">
            <Button
              aria-label="Clear screen annotations"
              className={cn(GHOST_ICON_BTN, 'size-6')}
              onClick={dismissScreenAnnotations}
              type="button"
              variant="ghost"
            >
              <Codicon name="close" />
            </Button>
          </Tip>
        </>
      )}
    </div>
  )
}
