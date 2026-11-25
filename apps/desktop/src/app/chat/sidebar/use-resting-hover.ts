import { type PointerEvent, useEffect, useRef, useState } from 'react'

/** How long a resting pointer takes to open a sidebar popover, and how long it may travel between
 *  its anchor and the popover before it closes. */
const HOVER_OPEN_MS = 300
const HOVER_CLOSE_MS = 200

/** A sidebar popover a resting mouse opens: 'hover' is that kind of open, which must not move focus
 *  and ends when the pointer has left both the anchor and the popover; 'click' is a deliberate open
 *  (a click, the keyboard), which the pointer leaving does not end. Spread `hoverHandlers` on both
 *  the anchor and the popover's content. */
export function useRestingHover() {
  const [open, setOpen] = useState<'click' | 'hover' | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const schedule = (next: 'hover' | null, delay: number) => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setOpen(current => (current === 'click' ? current : next)), delay)
  }

  const cancel = () => window.clearTimeout(timer.current)

  useEffect(() => cancel, [])

  const close = () => {
    cancel()
    setOpen(null)
  }

  const hoverHandlers = {
    onPointerEnter: (event: PointerEvent<HTMLElement>) => {
      if (event.pointerType === 'mouse') {
        schedule('hover', open ? 0 : HOVER_OPEN_MS)
      }
    },
    onPointerLeave: (event: PointerEvent<HTMLElement>) => {
      if (event.pointerType === 'mouse') {
        schedule(null, HOVER_CLOSE_MS)
      }
    }
  }

  return { cancel, close, hoverHandlers, open, setOpen }
}
