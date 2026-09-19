import { animate, AnimatePresence, motion, useIsPresent, useMotionValue, useReducedMotion } from 'motion/react'
import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { cn } from '@/lib/utils'

export type CardStackDirection = 'left' | 'right'

export interface CardStackAction {
  active: boolean
  busy: boolean
  /** Feedback and departure are shared; persistence stays with the consumer. */
  depart: (action: () => void | Promise<void>) => Promise<void>
}

interface CardStackProps<T> {
  items: readonly T[]
  getKey: (item: T) => string
  children: (item: T, action: CardStackAction) => ReactNode
  /** Surface only. All placements share the geometry and gesture controller. */
  surfaceClassName: string
  expanded?: boolean
  swipeDirections?: readonly CardStackDirection[]
  onSwipe?: (item: T, direction: CardStackDirection, action: CardStackAction) => void
}

// Measured from Cursor's shipped AgentTranscriptApprovalStack: one silhouette,
// 7px/96% promotion and a 14px upward clearance. Shared by every consumer.
const GAP = 7
const SCALE = 0.96
const EASE = [0.23, 1, 0.32, 1] as const
const PROMOTION = { duration: 0.22, ease: EASE }
const DEPARTURE = { duration: 0.18, ease: EASE }
const RETURN = { type: 'spring' as const, stiffness: 460, damping: 34, mass: 0.8 }
const NO_SWIPES: readonly CardStackDirection[] = []

export function CardStack<T>({
  items,
  getKey,
  children,
  surfaceClassName,
  expanded = false,
  swipeDirections = NO_SWIPES,
  onSwipe
}: CardStackProps<T>) {
  const [height, setHeight] = useState(0)
  const [frontKey, setFrontKey] = useState<string | undefined>(() => items[0] && getKey(items[0]))
  // Keep the item being read when newer items arrive. Cursor's live projection
  // keeps its previous entry id until that entry leaves the pending list.
  const front = items.find(item => getKey(item) === frontKey) ?? items[0]
  const currentKey = front && getKey(front)
  useLayoutEffect(() => {
    setFrontKey(currentKey)
  }, [currentKey])

  const shown = expanded ? items : front ? [front] : []
  const reduced = useReducedMotion()

  return (
    <div
      className={cn('relative isolate min-w-0', expanded && 'flex flex-col gap-2')}
      data-expanded={expanded}
      data-slot="card-stack"
      data-stack-count={items.length}
      style={{
        paddingTop: !expanded && items.length ? 8 : 0,
        minHeight: items.length || height ? height + (expanded ? 0 : 8) : undefined
      }}
    >
      {!expanded && items.length > 1 && (
        <motion.div
          animate={{ opacity: 0.6 }}
          aria-hidden="true"
          className={cn('pointer-events-none absolute inset-x-0', surfaceClassName)}
          data-glass-opaque=""
          data-slot="card-stack-edge"
          initial={{ opacity: 0 }}
          style={{
            top: 8,
            height,
            transform: `translateY(-${GAP}px) scale(${SCALE})`,
            transformOrigin: 'top center',
            zIndex: 0
          }}
          transition={reduced ? { duration: 0 } : PROMOTION}
        />
      )}
      <AnimatePresence
        initial={false}
        onExitComplete={() => {
          if (!items.length) {
            setHeight(0)
          }
        }}
      >
        {shown.map((item, index) => (
          <StackCard
            expanded={expanded}
            index={index}
            itemKey={getKey(item)}
            key={getKey(item)}
            onMeasure={setHeight}
            onSwipe={onSwipe ? (side, action) => onSwipe(item, side, action) : undefined}
            surfaceClassName={surfaceClassName}
            swipeDirections={swipeDirections}
          >
            {action => children(item, action)}
          </StackCard>
        ))}
      </AnimatePresence>
    </div>
  )
}

function StackCard({
  children,
  itemKey,
  index,
  onMeasure,
  expanded,
  surfaceClassName,
  swipeDirections,
  onSwipe
}: {
  children: (action: CardStackAction) => ReactNode
  itemKey: string
  index: number
  onMeasure: (height: number) => void
  expanded: boolean
  surfaceClassName: string
  swipeDirections: readonly CardStackDirection[]
  onSwipe?: (direction: CardStackDirection, action: CardStackAction) => void
}) {
  const present = useIsPresent()
  const reduced = useReducedMotion()
  const active = present && (expanded || index === 0)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const locked = useRef(false)
  const node = useRef<HTMLDivElement>(null)
  const content = useRef<HTMLDivElement>(null)

  const pointer = useRef<{
    id: number
    x: number
    y: number
    lastX: number
    lastTime: number
    velocity: number
    moved: boolean
  } | null>(null)

  const x = useMotionValue(0)
  const y = useMotionValue(0)

  useLayoutEffect(() => {
    if (index !== 0 || !content.current) {
      return
    }

    const element = content.current
    const measure = () => onMeasure(element.offsetHeight + 2)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)

    return () => observer.disconnect()
  }, [index, onMeasure])

  useEffect(
    () => () => {
      x.stop()
      y.stop()
    },
    [x, y]
  )

  const action: CardStackAction = {
    active,
    busy,
    async depart(commit) {
      if (locked.current || !active) {
        return
      }

      locked.current = true
      setBusy(true)
      setDragging(false)

      try {
        // Immediate press feedback. Removal drives the shared keyed exit and
        // promotion together; a slow or failed reply never leaves a blank pile.
        animate(y, reduced ? 0 : 1, { duration: 0.08 })
        await commit()
      } catch (error) {
        locked.current = false
        setBusy(false)
        animate(x, 0, reduced ? { duration: 0 } : RETURN)
        animate(y, 0, { duration: reduced ? 0 : 0.12 })
        throw error
      }
    }
  }

  const resetDrag = () => {
    pointer.current = null
    setDragging(false)
    animate(x, 0, reduced ? { duration: 0 } : RETURN)
  }

  return (
    <motion.div
      animate={{ opacity: 1, y: 0, scale: 1 }}
      aria-hidden={!active || undefined}
      className={cn('min-w-0', expanded ? 'relative' : 'absolute inset-x-0')}
      data-slot="card-stack-front"
      data-stack-active={active}
      data-stack-busy={busy}
      data-stack-key={itemKey}
      exit={{ opacity: 0, y: reduced ? 0 : -14, transition: reduced ? { duration: 0 } : DEPARTURE }}
      inert={!active || busy}
      initial={reduced ? { opacity: 1 } : { opacity: 0.6, y: -GAP, scale: SCALE }}
      ref={node}
      style={{
        zIndex: present ? 1 : 2,
        top: !expanded ? 8 : undefined,
        transformOrigin: 'top center'
      }}
      transition={reduced ? { duration: 0 } : PROMOTION}
    >
      <motion.div
        className={cn(surfaceClassName, onSwipe && active && 'cursor-grab', dragging && 'cursor-grabbing')}
        data-glass-opaque=""
        data-slot="card-stack-surface"
        onLostPointerCapture={() => {
          if (pointer.current) {
            resetDrag()
          }
        }}
        onPointerCancel={resetDrag}
        onPointerDown={event => {
          if (
            !active ||
            locked.current ||
            !onSwipe ||
            event.button !== 0 ||
            (event.target as HTMLElement).closest('button,a,input,textarea,select,summary,pre,[contenteditable]')
          ) {
            return
          }

          pointer.current = {
            id: event.pointerId,
            x: event.clientX,
            y: event.clientY,
            lastX: event.clientX,
            lastTime: event.timeStamp,
            velocity: 0,
            moved: false
          }
        }}
        onPointerMove={event => {
          const start = pointer.current

          if (!start || start.id !== event.pointerId) {
            return
          }

          const delta = event.clientX - start.x

          if (!start.moved) {
            if (Math.abs(event.clientY - start.y) > Math.abs(delta) && Math.abs(event.clientY - start.y) > 6) {
              resetDrag()

              return
            }

            if (Math.abs(delta) < 6) {
              return
            }

            if (window.getSelection()?.toString()) {
              resetDrag()

              return
            }

            start.moved = true
            event.currentTarget.setPointerCapture(event.pointerId)
            setDragging(true)
          }

          const side = delta < 0 ? 'left' : 'right'
          const allowed = swipeDirections.includes(side)
          x.set(allowed ? delta : delta / (1.5 + Math.abs(delta) / 20))
          start.velocity = (event.clientX - start.lastX) / Math.max(1, event.timeStamp - start.lastTime)
          start.lastX = event.clientX
          start.lastTime = event.timeStamp
        }}
        onPointerUp={event => {
          const start = pointer.current

          if (!start || start.id !== event.pointerId) {
            return
          }

          const delta = event.clientX - start.x
          const side = delta < 0 ? 'left' : 'right'
          const threshold = Math.min(96, (node.current?.offsetWidth ?? 360) * 0.22)
          const fast = event.timeStamp - start.lastTime < 80 && Math.abs(start.velocity) > 0.5 && Math.abs(delta) > 18
          pointer.current = null
          setDragging(false)

          if (start.moved && swipeDirections.includes(side) && (Math.abs(delta) > threshold || fast)) {
            onSwipe?.(side, action)
          } else {
            animate(x, 0, reduced ? { duration: 0 } : RETURN)
          }
        }}
        style={{ x, y, touchAction: onSwipe ? 'pan-y' : undefined }}
      >
        <div ref={content}>{children(action)}</div>
      </motion.div>
    </motion.div>
  )
}
