import type { ReactNode } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorState } from '@/components/ui/error-state'
import { Tip } from '@/components/ui/tooltip'

interface ContribBoundaryProps {
  children: ReactNode
  /** Contribution key, shown in the fallback + console tag. */
  id: string
  /** `chip` = inline bar item (tiny fallback); `pane` = zone body. */
  variant?: 'chip' | 'pane'
}

/**
 * The blast wall between a contribution's `render()` and the app. Plugin
 * code throwing during render (bad import, undefined component, logic bug)
 * degrades to a small inline error in ITS slot — the surrounding bar/zone,
 * other plugins, and the app keep working. Every surface that mounts
 * contribution renders wraps them in this.
 *
 * The pane fallback uses the app's canonical `ErrorState` (same icon/title/body
 * as the React boundary and dialog errors) so a crashed contribution reads like
 * every other failure, not a raw stack dump.
 */
export function ContribBoundary({ children, id, variant = 'pane' }: ContribBoundaryProps) {
  return (
    <ErrorBoundary
      fallback={({ error, reset }) =>
        variant === 'chip' ? (
          <Tip label={`${id}: ${error.message}`}>
            <button
              className="inline-flex items-center gap-1 rounded px-1.5 text-[0.6875rem] text-destructive transition-colors hover:bg-(--chrome-action-hover)"
              onClick={reset}
              type="button"
            >
              <Codicon name="warning" size="0.7rem" />
              {id}
            </button>
          </Tip>
        ) : (
          <div className="grid h-full place-items-center p-6">
            <ErrorState description={error.message} title={`“${id}” failed to render`}>
              <Button className="justify-self-center" onClick={reset} size="sm" variant="outline">
                <Codicon name="refresh" size="0.8rem" />
                Retry
              </Button>
            </ErrorState>
          </div>
        )
      }
      label={`contrib:${id}`}
    >
      {children}
    </ErrorBoundary>
  )
}
