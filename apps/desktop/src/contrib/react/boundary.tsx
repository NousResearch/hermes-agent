import { createElement, useState, type ReactNode } from 'react'

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

interface ContribRenderProps {
  render: () => ReactNode
}

/** Mount a contribution callback as a component so its hooks and errors belong
 * to the contribution, not to whichever host surface happened to call it. */
export function ContribRender({ render }: ContribRenderProps) {
  return createElement(render)
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
 *
 * Renderer-side retry storm guard (the issue this file existed to fix):
 * a hot-mounted contribution that throws `Maximum call stack size exceeded`
 * on every render — discovered 2026-07-25 where the workspace slot cycled
 * between render — throw — ErrorBoundary — fallback — slot subscription
 * re-renders it — repeat — would spin the renderer at ~30% CPU forever
 * even after Retry was capped. To break the loop we count crashes via
 * `useState`. Once the limit is hit, the boundary short-circuits and
 * returns a permanent post-mortem element in place of `children`. The slot's
 * `useSyncExternalStore` subscriber, the parent reconcile, the boundary's
 * own state churn, or any other path can no longer re-mount the broken
 * subtree because the subtree is no longer being rendered at all.
 */
const MAX_CONTRIB_RETRIES = 1

export function ContribBoundary({ children, id, variant = 'pane' }: ContribBoundaryProps) {
  const [crashCount, setCrashCount] = useState(0)

  // If we've hit the crash cap, render a permanent post-mortem and DO NOT
  // touch `children`. Without this branch, React would keep calling
  // ErrorBoundary's render with the broken subtree attached, which is what
  // fed the renderer CPU spin.
  if (crashCount > MAX_CONTRIB_RETRIES) {
    if (variant === 'chip') {
      return (
        <span
          className="inline-flex items-center gap-1 rounded px-1.5 text-[0.6875rem] text-muted-foreground"
          title={`${id}: crashed ${crashCount} times — slot locked`}
        >
          <Codicon name="warning" size="0.7rem" />
          {id} (locked)
        </span>
      )
    }
    return (
      <div className="grid h-full place-items-center p-6">
        <ErrorState
          description={`${id} crashed ${crashCount} times in a row — slot locked to stop the renderer retry loop. Reload the window (Ctrl+R) once the underlying issue is fixed.`}
          title={`«${id}» failed to render`}
        />
      </div>
    )
  }

  const handleReset = (baseReset: () => void) => () => {
    if (crashCount >= MAX_CONTRIB_RETRIES) {
      console.warn(
        `[contrib:${id}] retry limit (${MAX_CONTRIB_RETRIES}) reached; refusing to re-enter broken render to break auto-retry loop`
      )
      return
    }

    baseReset()
  }

  return (
    <ErrorBoundary
      onError={() => {
        // Each caught render failure counts toward the cap; once exceeded,
        // `crashCount` updates trigger a re-render that swaps the broken
        // boundary out for the permanent post-mortem above.
        setCrashCount(c => c + 1)
      }}
      fallback={({ error, reset }) => {
        const retriesExhausted = crashCount >= MAX_CONTRIB_RETRIES

        if (variant === 'chip') {
          return (
            <Tip
              label={
                retriesExhausted
                  ? `${id}: retries exhausted (${error.message})`
                  : `${id}: ${error.message}`
              }
            >
              <button
                className="inline-flex items-center gap-1 rounded px-1.5 text-[0.6875rem] text-destructive transition-colors hover:bg-(--chrome-action-hover)"
                disabled={retriesExhausted}
                onClick={handleReset(reset)}
                type="button"
              >
                <Codicon name="warning" size="0.7rem" />
                {id}
                {retriesExhausted ? ' (locked)' : ''}
              </button>
            </Tip>
          )
        }

        return (
          <div className="grid h-full place-items-center p-6">
            <ErrorState
              description={
                retriesExhausted
                  ? `After ${MAX_CONTRIB_RETRIES} retry the contribution still failed (${error.message}). Reload the window (Ctrl+R) once the underlying issue is fixed.`
                  : error.message
              }
              title={`«${id}» failed to render`}
            >
              {retriesExhausted ? (
                <p className="text-xs text-muted-foreground text-center">
                  Retries exhausted. Locking the slot — the broken contribution will not be re-rendered again on this page.
                </p>
              ) : (
                <Button
                  className="justify-self-center"
                  onClick={handleReset(reset)}
                  size="sm"
                  variant="outline"
                >
                  <Codicon name="refresh" size="0.8rem" />
                  Retry
                </Button>
              )}
            </ErrorState>
          </div>
        )
      }}
      label={`contrib:${id}`}
    >
      {children}
    </ErrorBoundary>
  )
}
