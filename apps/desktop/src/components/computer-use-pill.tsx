import { useStore } from '@nanostores/react'
import { AnimatePresence, motion } from 'motion/react'
import { memo } from 'react'

import { useI18n } from '@/i18n'
import { AlertTriangle, AppWindow, Check, X } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $activeComputerUse,
  clearComputerUseState,
  type ComputerUseActiveState,
  type ComputerUseLabels,
  formatComputerUseTarget,
  sessionComputerUse
} from '@/store/computer-use'
import { $activeSessionId } from '@/store/session'

export interface ComputerUseStatusPillProps {
  className?: string
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

function pillDetail(state: ComputerUseActiveState, labels: ComputerUseLabels): string {
  switch (state.phase) {
    case 'drafting':
      return labels.preparing

    case 'error':
      return state.error || labels.failed

    case 'completed':
      return state.durationSeconds != null
        ? `${formatComputerUseTarget(state, labels)} (${state.durationSeconds.toFixed(1)}s)`
        : formatComputerUseTarget(state, labels)

    default:
      return formatComputerUseTarget(state, labels)
  }
}

export const ComputerUseStatusPill = memo(function ComputerUseStatusPill({
  className,
  sessionId,
  variant = 'titlebar'
}: ComputerUseStatusPillProps) {
  const { t } = useI18n()
  const labels = t.titlebar.computerUse
  const activeSessionId = useStore($activeSessionId)
  const currentSessionId = sessionId ?? activeSessionId

  const store = sessionId ? sessionComputerUse(sessionId) : $activeComputerUse
  const state = useStore(store)
  const detail = state ? pillDetail(state, labels) : ''

  return (
    <AnimatePresence>
      {state && (
        <motion.div
          animate={{ opacity: 1, scale: 1, y: 0 }}
          aria-atomic="true"
          className={cn(
            'group inline-flex items-center gap-1.5 select-none rounded-full backdrop-blur-md [-webkit-app-region:no-drag]',
            variant === 'titlebar'
              ? 'h-6.5 px-2.5 text-[11px] max-w-[280px]'
              : 'h-8 px-3 text-xs shadow-md max-w-[340px]',
            state.phase === 'error'
              ? 'border-destructive/35 bg-destructive/10 text-destructive'
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
          role="status"
          title={`${labels.name}: ${detail}`}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
        >
          <StatusIndicator phase={state.phase} />

          <span className="flex items-center gap-1 min-w-0">
            <span
              className={cn(
                'font-mono text-[9px] font-semibold tracking-wider uppercase shrink-0',
                state.phase === 'error' ? 'text-destructive' : 'text-emerald-600 dark:text-emerald-400'
              )}
            >
              {labels.name}
            </span>

            <span aria-hidden className="text-muted-foreground/40 font-mono text-[10px]">
              ·
            </span>

            <span
              className={cn(
                'truncate min-w-0 font-medium',
                state.phase === 'error' ? 'text-destructive' : 'text-foreground'
              )}
            >
              {detail}
            </span>
          </span>

          {state.phase === 'error' ? (
            <button
              aria-label={labels.dismiss}
              className="-mr-1 flex size-4 shrink-0 items-center justify-center rounded-full text-destructive/80 hover:bg-destructive/15 hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-destructive/50"
              onClick={() => clearComputerUseState(currentSessionId)}
              title={labels.dismiss}
              type="button"
            >
              <X className="size-3" />
            </button>
          ) : (
            state.app && <AppWindow aria-hidden className="size-3 text-muted-foreground/60 shrink-0 ml-0.5" />
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
})
