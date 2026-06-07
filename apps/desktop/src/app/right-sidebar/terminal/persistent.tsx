import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { TERMINAL_BG } from './selection'

import { TerminalTab } from './index'

/**
 * One xterm Terminal mounted at the layout root and CSS-overlayed onto
 * whichever `<TerminalSlot />` is active. Moving the host DOM detaches xterm's
 * WebGL renderer (it observes its own attachment) and resets the screen, so
 * the host stays put and we chase the slot's bounding rect with position:fixed.
 */

const $slot = atom<HTMLElement | null>(null)

const SLOT_CLASS = 'relative flex min-h-0 min-w-0 flex-1 flex-col'

export function TerminalSlot({ className = SLOT_CLASS }: { className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const el = ref.current

    if (!el) {
      return
    }

    $slot.set(el)

    return () => {
      if ($slot.get() === el) {
        $slot.set(null)
      }
    }
  }, [])

  return <div className={className} ref={ref} />
}

interface PersistentTerminalProps {
  cwd: string
  onAddSelectionToChat: (text: string, label?: string) => void
}

interface Rect {
  top: number
  left: number
  width: number
  height: number
}

const sameRect = (a: Rect | null, b: Rect) =>
  !!a && a.top === b.top && a.left === b.left && a.width === b.width && a.height === b.height

export function PersistentTerminal({ cwd, onAddSelectionToChat }: PersistentTerminalProps) {
  const slot = useStore($slot)
  const [rect, setRect] = useState<Rect | null>(null)
  const [ready, setReady] = useState(false)

  useLayoutEffect(() => {
    if (!slot) {
      setRect(null)

      return
    }

    let prev: Rect | null = null
    let frame = 0
    let stableFrames = 0

    const measure = (): boolean => {
      const r = slot.getBoundingClientRect()
      // floor top/left + ceil right/bottom: overlay always covers the slot's
      // full pixel footprint, so half-pixel rects can't leak page bg through.
      const top = Math.floor(r.top)
      const left = Math.floor(r.left)
      const next: Rect = { top, left, width: Math.ceil(r.right) - left, height: Math.ceil(r.bottom) - top }

      if (sameRect(prev, next)) {
        return false
      }

      prev = next
      setRect(next)

      if (next.width > 0 && next.height > 0) {
        setReady(true)
      }

      return true
    }

    // Track the slot through a transition, then STOP. The old implementation
    // re-ran getBoundingClientRect() every frame forever, which forces a layout
    // flush ~60×/sec for the whole app lifetime — a constant main-thread tax
    // that amplifies any other jank into a perceptible freeze on slower/Windows
    // machines. Instead, re-measure each frame only until the rect has held
    // steady for a few frames, then idle until the next resize/scroll/layout
    // change re-arms the burst.
    const settle = () => {
      stableFrames = 0

      if (frame) {
        return
      }

      const step = () => {
        stableFrames = measure() ? 0 : stableFrames + 1

        if (stableFrames >= 3) {
          frame = 0

          return
        }

        frame = requestAnimationFrame(step)
      }

      frame = requestAnimationFrame(step)
    }

    measure()
    settle()

    // Re-arm the burst on the events that can move/resize the slot. A plain
    // ResizeObserver alone misses position-only shifts (a sibling pane resizing
    // moves the slot without changing its size), so pair it with window resize
    // and capture-phase scroll.
    const reArm = () => settle()
    const resizeObserver = new ResizeObserver(reArm)
    resizeObserver.observe(slot)
    window.addEventListener('resize', reArm)
    window.addEventListener('scroll', reArm, { capture: true, passive: true })

    return () => {
      if (frame) {
        cancelAnimationFrame(frame)
      }

      resizeObserver.disconnect()
      window.removeEventListener('resize', reArm)
      window.removeEventListener('scroll', reArm, { capture: true })
    }
  }, [slot])

  const visible = Boolean(rect && rect.width > 0 && rect.height > 0)

  const style: CSSProperties = {
    position: 'fixed',
    top: rect?.top ?? 0,
    left: rect?.left ?? 0,
    width: rect?.width ?? 0,
    height: rect?.height ?? 0,
    display: 'flex',
    flexDirection: 'column',
    visibility: visible ? 'visible' : 'hidden',
    pointerEvents: visible ? 'auto' : 'none',
    zIndex: 4,
    backgroundColor: TERMINAL_BG,
    contain: 'layout size paint'
  }

  // Defer mount until real dims — booting xterm at 0×0 starts the shell at
  // 80×24, then the first ResizeObserver SIGWINCH redraws the prompt on a
  // new line. After first measurement we keep it mounted forever.
  return (
    <div aria-hidden={!visible} style={style}>
      {ready && <TerminalTab cwd={cwd} onAddSelectionToChat={onAddSelectionToChat} />}
    </div>
  )
}
