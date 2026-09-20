import type { PointerEvent as ReactPointerEvent } from 'react'

import { cn } from '@/lib/utils'

// A one-pixel split handle with forgiving geometry: the visual hairline sits on
// the boundary, while an invisible 8px band (this element plus its bleed span)
// owns the drag. Sizes live in the IDE layout store, so the handle reports
// absolute-from-start deltas against the size captured at pointer-down.

interface IdeSplitHandleProps {
  axis: 'x' | 'y'
  /** Positive pointer movement SHRINKS the pane (right/top-docked panes). */
  invert?: boolean
  label: string
  setSize: (size: number) => void
  size: () => number
}

export function IdeSplitHandle({ axis, invert, label, setSize, size }: IdeSplitHandleProps) {
  const isX = axis === 'x'

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()

    const element = event.currentTarget
    const start = isX ? event.clientX : event.clientY
    const startSize = size()

    element.setPointerCapture?.(event.pointerId)

    const move = (moveEvent: PointerEvent) => {
      const raw = (isX ? moveEvent.clientX : moveEvent.clientY) - start

      setSize(startSize + (invert ? -raw : raw))
    }

    const finish = () => {
      element.releasePointerCapture?.(event.pointerId)
      element.removeEventListener('pointermove', move)
      element.removeEventListener('pointerup', finish)
      element.removeEventListener('pointercancel', finish)
    }

    element.addEventListener('pointermove', move)
    element.addEventListener('pointerup', finish)
    element.addEventListener('pointercancel', finish)
  }

  return (
    <div
      aria-label={label}
      aria-orientation={isX ? 'vertical' : 'horizontal'}
      className={cn(
        'group relative z-10 shrink-0 touch-none',
        isX ? 'w-0 cursor-col-resize' : 'h-0 cursor-row-resize'
      )}
      onPointerDown={onPointerDown}
      role="separator"
    >
      <span aria-hidden className={cn('absolute', isX ? 'inset-y-0 -left-1 w-2' : 'inset-x-0 -top-1 h-2')} />
      <span
        aria-hidden
        className={cn(
          'pointer-events-none absolute bg-(--ui-stroke-tertiary) group-hover:bg-(--ui-accent)',
          isX ? 'inset-y-0 left-0 w-px' : 'inset-x-0 top-0 h-px'
        )}
      />
    </div>
  )
}
