import { type ReactNode, useLayoutEffect, useRef, useState } from 'react'

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

  useLayoutEffect(() => {
    const body = ref.current

    if (hidden || !body) {
      return
    }

    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect

      if (width > 0 && height > 0) {
        setSize(previous => (previous?.width === width && previous.height === height ? previous : { width, height }))
      }
    })

    observer.observe(body)

    return () => observer.disconnect()
  }, [hidden])

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
