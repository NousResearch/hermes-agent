import type { ReactNode } from 'react'

import { ErrorBoundary } from '@/components/error-boundary'

interface ReasoningRenderBoundaryProps {
  children: ReactNode
  text: string
}

/** Keep pathological model-generated reasoning from taking down its message pane.
 * Rich markdown remains the primary surface; if that renderer throws, the raw
 * reasoning is still readable and later reasoning parts mount normally. */
export function ReasoningRenderBoundary({ children, text }: ReasoningRenderBoundaryProps) {
  return (
    <ErrorBoundary
      fallback={() => (
        <div
          className="wrap-anywhere whitespace-pre-wrap text-xs leading-snug text-muted-foreground/85"
          data-render-fallback="reasoning"
          data-slot="aui_reasoning-text"
        >
          {text}
        </div>
      )}
      label="reasoning-markdown"
    >
      {children}
    </ErrorBoundary>
  )
}
