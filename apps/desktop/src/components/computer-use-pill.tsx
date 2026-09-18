import { useStore } from '@nanostores/react'
import { AnimatePresence, motion } from 'motion/react'
import type { ComponentProps } from 'react'
import { memo } from 'react'

import { AlertTriangle, AppWindow, Check } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $activeComputerUse,
  clearComputerUseState,
  type ComputerUseActiveState,
  sessionComputerUse
} from '@/store/computer-use'
import { $activeSessionId } from '@/store/session'

export interface ComputerUseStatusPillProps extends ComponentProps<'div'> {
  /** Optional specific session id. If omitted, tracks the active session. */
  sessionId?: string | null
  /** Optional variant style: 'titlebar' (compact, 26px height) or 'floating' (slightly more padded). */
  variant?: 'titlebar' | 'floating'
}

function StatusIndicator({ phase }: { phase: ComputerUseActiveState['phase'] }) {
  if (phase === 'completed') {
    return (
      <span className="flex size-3.5 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
        <Check className="size-2.5 stroke-[2.5]" />
      </span>
    )
  }

  if (phase === 'error') {
    return (
      <span className="flex size-3.5 shrink-0 items-center justify-center rounded-full bg-destructive/15 text-destructive">
        <AlertTriangle className="size-2.5 stroke-[2.5]" />
      </span>
    )
  }

  if (phase === 'drafting') {
    return (
      <span className="relative flex size-2 shrink-0 items-center justify-center">
        <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" />
      </span>
    )
  }

  // running phase: pulsing radar ping
  return (
    <span className="relative flex size-2 shrink-0 items-center justify-center">
      <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-400 opacity-75" />
      <span className="relative inline-flex size-1.5 rounded-full bg-emerald-500" />
    </span>
  )
}

export const ComputerUseStatusPill = memo(function ComputerUseStatusPill({
  className,
  sessionId,
  variant = 'titlebar',
  ...props
}: ComputerUseStatusPillProps) {
  const activeSessionId = useStore($activeSessionId)
  const currentSessionId = sessionId ?? activeSessionId

  const store = sessionId ? sessionComputerUse(sessionId) : $activeComputerUse
  const state = useStore(store)

  const isVisible = state != null && state.phase !== 'idle'

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          animate={{ opacity: 1, scale: 1, y: 0 }}
          aria-atomic="true"
          aria-label={
            state.phase === 'error'
              ? `Computer Use error: ${state.error || 'Failed'}`
              : state.phase === 'completed'
                ? `Computer Use completed: ${state.targetSummary || 'Done'}`
                : `Computer Use running: ${state.targetSummary || 'Active'}`
          }
          aria-live="polite"
          className={cn(
            'group inline-flex items-center gap-1.5 select-none rounded-full backdrop-blur-md [-webkit-app-region:no-drag]',
            variant === 'titlebar'
              ? 'h-6.5 px-2.5 text-[11px] max-w-[280px]'
              : 'h-8 px-3 text-xs shadow-md max-w-[340px]',
            state.phase === 'error'
              ? 'border-destructive/35 bg-destructive/10 text-destructive cursor-pointer hover:bg-destructive/15'
              : state.phase === 'completed'
                ? 'border-emerald-500/35 bg-card/90 text-foreground shadow-xs'
                : 'border-border/80 bg-card/90 text-foreground shadow-xs',
            'border',
            className
          )}
          data-computer-use-phase={state.phase}
          data-computer-use-pill
          exit={{ opacity: 0, scale: 0.95, y: -4 }}
          initial={{ opacity: 0, scale: 0.95, y: -4 }}
          key="computer-use-status-pill"
          onClick={state.phase === 'error' ? () => clearComputerUseState(currentSessionId) : undefined}
          role="status"
          title={
            state.phase === 'error'
              ? `Computer Use error: ${state.error || 'Failed'} (Click to dismiss)`
              : state.phase === 'completed'
                ? `Computer Use completed: ${state.targetSummary || 'Done'}${
                    state.durationSeconds != null ? ` (${state.durationSeconds.toFixed(1)}s)` : ''
                  }`
                : `Computer Use: ${state.targetSummary || (state.app ? state.app : 'Operating desktop')}`
          }
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          {...(props as any)}
        >
          <StatusIndicator phase={state.phase} />

          <span className="flex items-center gap-1 min-w-0">
            <span
              className={cn(
                'font-mono text-[9px] font-semibold tracking-wider uppercase shrink-0',
                state.phase === 'error'
                  ? 'text-destructive'
                  : 'text-emerald-600 dark:text-emerald-400'
              )}
            >
              Computer Use
            </span>

            <span className="text-muted-foreground/40 font-mono text-[10px]">·</span>

            <span className="font-medium text-foreground truncate min-w-0">
              {state.phase === 'error' ? (
                <span className="text-destructive truncate">{state.error || 'Action failed'}</span>
              ) : state.phase === 'completed' ? (
                <span className="truncate">
                  {state.targetSummary || 'Done'}
                  {state.durationSeconds != null ? ` (${state.durationSeconds.toFixed(1)}s)` : ''}
                </span>
              ) : (
                state.targetSummary || (state.app ? state.app : 'Operating desktop')
              )}
            </span>
          </span>

          {state.app && state.phase !== 'error' && (
            <AppWindow className="size-3 text-muted-foreground/60 shrink-0 ml-0.5" />
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
})

export { ComputerUseStatusPill as ComputerUsePill }
