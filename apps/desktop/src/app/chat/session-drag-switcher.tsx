import { useQuery } from '@tanstack/react-query'
import { type FC, type ReactNode, useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { Codicon } from '@/components/ui/codicon'
import { listAllProfileSessions } from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import { cn } from '@/lib/utils'
import { $selectedStoredSessionId } from '@/store/session'

import { sessionRoute } from '../routes'

const MAX_SESSIONS = 50
const DRAG_THRESHOLD_PX = 80
const MIN_HORIZONTAL_PX = 30
const DIRECTION_BIAS_RATIO = 2 // |dx| must be > 2 * |dy|

type DragDirection = 'left' | 'right'

interface DragState {
  direction: DragDirection
  /** 0…1 — how far past the activation threshold we are */
  progress: number
  targetSessionId: string
  targetTitle: string
}

/**
 * Fetches up to 50 recent sessions and enables horizontal mouse-drag
 * (or trackpad swipe) in the chat area to switch between them.
 *
 * Gesture: hold mouse button, drag left or right past 80px — a directional
 * overlay appears showing the target session title. Release to commit the
 * switch (navigate to that session). Drag less than threshold → no-op.
 */
export const SessionDragSwitcher: FC<{ children: ReactNode }> = ({ children }) => {
  const navigate = useNavigate()
  const [drag, setDrag] = useState<DragState | null>(null)

  // --- fetch up to 50 recent sessions ---
  const sessionsQuery = useQuery({
    queryFn: () => listAllProfileSessions(MAX_SESSIONS, 0, 'exclude', 'recent'),
    queryKey: ['session-drag-switcher', 'sessions'],
    staleTime: 30_000 // don't refetch on every drag
  })

  // --- drag tracking refs (avoid re-render per pixel) ---
  const startXRef = useRef(0)
  const startYRef = useRef(0)
  const startTimeRef = useRef(0)
  const trackingRef = useRef(false)

  const resolveTarget = useCallback(
    (direction: DragDirection): { id: string; title: string } | null => {
      const currentId = $selectedStoredSessionId.get()
      const sessionsList = sessionsQuery.data?.sessions ?? []
      const idx = sessionsList.findIndex(s => s.id === currentId)

      if (sessionsList.length < 2) return null

      const nextIdx =
        direction === 'left'
          ? idx <= 0
            ? sessionsList.length - 1
            : idx - 1
          : idx >= sessionsList.length - 1
            ? 0
            : idx + 1

      const target = sessionsList[nextIdx]
      if (!target) return null

      return { id: target.id, title: sessionTitle(target) }
    },
    [sessionsQuery.data]
  )

  // --- mouse handlers ---
  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Only primary button (left click)
      if (e.button !== 0) return
      // Don't start if there's an active text selection
      const sel = window.getSelection()
      if (sel && sel.toString().length > 0) return
      // Don't start drag if user is interacting with an input/textarea/button/link
      const target = e.target as HTMLElement
      if (
        target.closest('input, textarea, button, [role="button"], a, select, [contenteditable="true"]')
      ) {
        return
      }

      startXRef.current = e.clientX
      startYRef.current = e.clientY
      startTimeRef.current = Date.now()
      trackingRef.current = true
    },
    []
  )

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!trackingRef.current) return

      const sessionsList = sessionsQuery.data?.sessions ?? []
      if (sessionsList.length < 2) return

      const dx = e.clientX - startXRef.current
      const dy = e.clientY - startYRef.current

      // Not enough horizontal movement yet
      if (Math.abs(dx) < MIN_HORIZONTAL_PX) return

      // Must be primarily horizontal
      if (Math.abs(dx) < Math.abs(dy) * DIRECTION_BIAS_RATIO) return

      const absDx = Math.abs(dx)

      if (absDx < DRAG_THRESHOLD_PX) {
        setDrag(null)
        return
      }

      const direction: DragDirection = dx < 0 ? 'left' : 'right'
      const target = resolveTarget(direction)
      if (!target) {
        setDrag(null)
        return
      }

      // Progress: how far past threshold, clamped to 0–1
      const maxExtra = 100
      const progress = Math.min((absDx - DRAG_THRESHOLD_PX) / maxExtra, 1)

      setDrag({ direction, progress, targetSessionId: target.id, targetTitle: target.title })
    },
    [sessionsQuery.data, resolveTarget]
  )

  const onMouseUp = useCallback(() => {
    if (!trackingRef.current) return
    trackingRef.current = false

    setDrag(current => {
      if (current) {
        // Commit the switch
        navigate(sessionRoute(current.targetSessionId))
      }
      return null
    })
  }, [navigate])

  // Clean up if mouse button is released outside the element
  useEffect(() => {
    const onGlobalMouseUp = () => {
      if (trackingRef.current) {
        trackingRef.current = false
        setDrag(null)
      }
    }

    window.addEventListener('mouseup', onGlobalMouseUp)
    return () => window.removeEventListener('mouseup', onGlobalMouseUp)
  }, [])

  return (
    <div className="relative flex min-h-0 flex-1 flex-col" onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}>
      {children}

      {/* Directional overlay */}
      {drag && (
        <div
          className={cn(
            'pointer-events-none absolute inset-y-0 z-50 flex w-64 flex-col items-center justify-center transition-opacity duration-100',
            drag.direction === 'left'
              ? 'right-0 bg-gradient-to-l from-(--ui-chat-surface-background)/90 to-transparent'
              : 'left-0 bg-gradient-to-r from-(--ui-chat-surface-background)/90 to-transparent'
          )}
          style={{ opacity: 0.3 + drag.progress * 0.7 }}
        >
          <Codicon
            className="mb-2 text-(--ui-text-secondary)"
            name={drag.direction === 'left' ? 'chevron-left' : 'chevron-right'}
            size="2rem"
          />
          <span className="max-w-[14rem] truncate text-center text-sm font-medium text-(--ui-text-secondary)">
            {drag.targetTitle}
          </span>
        </div>
      )}
    </div>
  )
}
