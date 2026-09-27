import { type ReactNode, useCallback, useRef, useState } from 'react'

import { useResizeObserver } from '@/hooks/use-resize-observer'

interface PaneBodyProps {
  hidden: boolean
  children: ReactNode
}

/** Collapse the zone, not its guests. Retaining the last visible viewport lets
 * background browser input use the same page coordinates while the restore
 * rail occupies only a sliver of the layout. Never detach/reparent a webview. */
export function PaneBody({ hidden, children }: PaneBodyProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState<{ width: number; height: number }>()

  // Rides the app's ONE shared observer (see use-resize-observer.ts); the
  // hidden body is a fixed box, so only visible measurements are remembered.
  const remember = useCallback(
    (entries: readonly ResizeObserverEntry[]) => {
      const rect = entries[0]?.contentRect

      if (!hidden && rect && rect.width > 0 && rect.height > 0) {
        setSize(previous =>
          previous?.width === rect.width && previous.height === rect.height
            ? previous
            : { width: rect.width, height: rect.height }
        )
      }
    },
    [hidden]
  )

  useResizeObserver(remember, ref)

  return (
    <div
      className="relative min-h-0 min-w-0 flex-1 overflow-hidden"
      ref={ref}
      style={hidden ? { position: 'absolute', visibility: 'hidden', pointerEvents: 'none', ...size } : undefined}
    >
      {children}
    </div>
  )
}
