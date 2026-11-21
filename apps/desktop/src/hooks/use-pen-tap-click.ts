import { useRef } from 'react'
import type * as React from 'react'

type PenTapHandlers = Pick<
  React.HTMLAttributes<HTMLElement>,
  'onPointerCancel' | 'onPointerDown' | 'onPointerUp' | 'onTouchEnd'
>

function touchEndedInside(event: React.TouchEvent<HTMLElement>): boolean {
  const touch = event.changedTouches?.[0]

  if (!touch) {
    return true
  }

  const rect = event.currentTarget.getBoundingClientRect()

  return (
    touch.clientX >= rect.left &&
    touch.clientX <= rect.right &&
    touch.clientY >= rect.top &&
    touch.clientY <= rect.bottom
  )
}

/**
 * iPadOS Safari sends pointer and touch events for an Apple Pencil tap on a menu
 * row, but no mousedown/mouseup/click (#150605). Radix menu rows select and open
 * submenus only on `click`, so a Pencil tap did nothing.
 *
 * When a pen tap ends with a `touchend` on the row, cancel the touchend (so the
 * browser cannot also send a compatibility click) and click the row ourselves:
 * one activation either way. Pens that send no touch events, as on Windows and
 * in Chromium, keep their native click. A pen gesture the browser takes over
 * (`pointercancel`) or one that ends off the row does not click.
 */
export function usePenTapClick({ onPointerCancel, onPointerDown, onPointerUp, onTouchEnd }: PenTapHandlers) {
  const penTapRef = useRef(false)

  return {
    onPointerCancel: (event: React.PointerEvent<HTMLElement>) => {
      onPointerCancel?.(event)
      penTapRef.current = false
    },
    onPointerDown: (event: React.PointerEvent<HTMLElement>) => {
      onPointerDown?.(event)
      penTapRef.current = false
    },
    onPointerUp: (event: React.PointerEvent<HTMLElement>) => {
      onPointerUp?.(event)
      penTapRef.current = event.pointerType === 'pen'
    },
    onTouchEnd: (event: React.TouchEvent<HTMLElement>) => {
      onTouchEnd?.(event)

      if (!penTapRef.current) {
        return
      }

      penTapRef.current = false

      if (event.defaultPrevented || !touchEndedInside(event)) {
        return
      }

      event.preventDefault()
      event.currentTarget.click()
    }
  } satisfies PenTapHandlers
}
