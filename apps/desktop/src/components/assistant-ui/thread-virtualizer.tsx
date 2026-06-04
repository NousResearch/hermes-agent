import { ThreadPrimitive, useAuiEvent, useAuiState } from '@assistant-ui/react'
import { useVirtualizer, type Virtualizer } from '@tanstack/react-virtual'
import { type ComponentProps, type FC, memo, type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef } from 'react'

import { cn } from '@/lib/utils'
import { setThreadScrolledUp } from '@/store/thread-scroll'

const ESTIMATED_ITEM_HEIGHT = 220
const OVERSCAN = 4
const AT_BOTTOM_THRESHOLD = 4
const POST_RUN_BOTTOM_LOCK_MS = 1_200

type ThreadMessageComponents = ComponentProps<typeof ThreadPrimitive.MessageByIndex>['components']

type MessageGroup = { id: string; index: number; kind: 'standalone' } | { id: string; indices: number[]; kind: 'turn' }

interface VirtualizedThreadProps {
  clampToComposer: boolean
  components: ThreadMessageComponents
  emptyPlaceholder?: ReactNode
  loadingIndicator?: ReactNode
  sessionKey?: string | null
}

function buildGroups(signature: string): MessageGroup[] {
  if (!signature) {
    return []
  }

  const messages = signature.split('\n').map(row => {
    const [index, id, role] = row.split(':')

    return { id, index: Number(index), role }
  })

  const groups: MessageGroup[] = []

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]

    if (message.role !== 'user') {
      groups.push({ id: message.id, index: message.index, kind: 'standalone' })

      continue
    }

    const indices = [message.index]

    while (i + 1 < messages.length && messages[i + 1].role !== 'user') {
      indices.push(messages[++i].index)
    }

    groups.push({ id: message.id, indices, kind: 'turn' })
  }

  return groups
}

const VirtualizedThreadInner: FC<VirtualizedThreadProps> = ({
  clampToComposer,
  components,
  emptyPlaceholder,
  loadingIndicator,
  sessionKey
}) => {
  const messageSignature = useAuiState(s =>
    s.thread.messages.map((message, index) => `${index}:${message.id}:${message.role}`).join('\n')
  )
  const isRunning = useAuiState(s => s.thread.isRunning)

  const groups = useMemo(() => buildGroups(messageSignature), [messageSignature])
  const renderEmpty = groups.length === 0 && Boolean(emptyPlaceholder)
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  // Shared ref so scrollToFn can check whether the user is parked at the
  // bottom without needing a ref from inside useThreadScrollAnchor.
  const stickyBottomRef = useRef(true)

  const virtualizer = useVirtualizer({
    count: groups.length,
    estimateSize: () => ESTIMATED_ITEM_HEIGHT,
    getItemKey: index => groups[index]?.id ?? index,
    getScrollElement: () => scrollerRef.current,
    // Seed the rect so the initial range mounts something before
    // `observeElementRect` reports the real layout (it overrides this).
    initialRect: { height: 600, width: 800 },
    overscan: OVERSCAN,
    // When the virtualizer adjusts scroll due to item measurement changes,
    // skip the adjustment if the user is at the bottom. Our ResizeObserver +
    // pinToBottom loop handles scroll anchoring; letting the virtualizer also
    // adjust creates a feedback loop where the two fight each other,
    // producing visible rubber-banding (the view snaps to the composer
    // then jumps back up).
    scrollToFn: (offset, _options, instance) => {
      const el = instance.scrollElement
      if (!el) {
        return
      }

      if (stickyBottomRef.current) {
        const maxScroll = el.scrollHeight - el.clientHeight
        const distFromBottom = maxScroll - el.scrollTop

        if (distFromBottom <= AT_BOTTOM_THRESHOLD && offset < maxScroll) {
          return
        }
      }

      ;(el as HTMLElement).scrollTo(0, offset)
    }
  })

  useThreadScrollAnchor({
    enabled: !renderEmpty,
    groupCount: groups.length,
    isRunning,
    scrollerRef,
    sessionKey: sessionKey ?? null,
    stickyBottomRef,
    virtualizer
  })

  const virtualItems = virtualizer.getVirtualItems()
  const totalSize = virtualizer.getTotalSize()
  const paddingTop = virtualItems[0]?.start ?? 0
  const paddingBottom = Math.max(0, totalSize - (virtualItems.at(-1)?.end ?? 0))

  return (
    <div
      className="relative min-h-0 max-w-full overflow-hidden contain-[layout_paint]"
      style={{ height: clampToComposer ? 'var(--thread-viewport-height)' : '100%' }}
    >
      <div
        className="size-full overflow-x-hidden overflow-y-auto overscroll-contain"
        data-slot="aui_thread-viewport"
        ref={scrollerRef}
      >
        {renderEmpty ? (
          <div
            className="mx-auto grid h-full w-full max-w-(--composer-width) grid-rows-[minmax(0,1fr)_auto] min-w-0 gap-(--conversation-turn-gap) px-6 py-8"
            data-slot="aui_thread-content"
          >
            {emptyPlaceholder}
          </div>
        ) : (
          <div
            className={cn(
              'mx-auto flex w-full max-w-(--composer-width) min-w-0 flex-col px-6 pt-[calc(var(--titlebar-height)+1.5rem)]'
            )}
            data-slot="aui_thread-content"
          >
            {/* Natural-flow virtualization: mounted items render as normal
                flex siblings so `position: sticky` on the human bubble
                resolves against the scroller without transform interference.
                Padding spacers reserve scroll space for unmounted items. */}
            <div style={{ paddingBottom: `${paddingBottom}px`, paddingTop: `${paddingTop}px` }}>
              {virtualItems.map(virtualItem => {
                const group = groups[virtualItem.index]

                if (!group) {
                  return null
                }

                return (
                  <div
                    className="flex min-w-0 flex-col gap-(--conversation-turn-gap) pb-(--conversation-turn-gap)"
                    data-index={virtualItem.index}
                    key={virtualItem.key}
                    ref={virtualizer.measureElement}
                  >
                    {group.kind === 'turn' ? (
                      <div
                        className="composer-human-ai-pair-container relative flex min-w-0 flex-col gap-(--conversation-turn-gap)"
                        data-slot="aui_turn-pair"
                      >
                        {group.indices.map(index => (
                          <ThreadPrimitive.MessageByIndex components={components} index={index} key={index} />
                        ))}
                      </div>
                    ) : (
                      <ThreadPrimitive.MessageByIndex components={components} index={group.index} />
                    )}
                  </div>
                )
              })}
            </div>
            {loadingIndicator}
            {clampToComposer && (
              <div
                aria-hidden="true"
                className="shrink-0"
                data-slot="aui_composer-clearance"
                style={{ height: 'var(--thread-last-message-clearance)' }}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export const VirtualizedThread = memo(VirtualizedThreadInner)

interface ScrollAnchorOptions {
  enabled: boolean
  groupCount: number
  isRunning: boolean
  scrollerRef: React.RefObject<HTMLDivElement | null>
  sessionKey: string | null
  stickyBottomRef: React.MutableRefObject<boolean>
  virtualizer: Virtualizer<HTMLDivElement, Element>
}

function useThreadScrollAnchor({
  enabled,
  groupCount,
  isRunning,
  scrollerRef,
  sessionKey,
  stickyBottomRef,
  virtualizer
}: ScrollAnchorOptions) {
  // `stickyBottomRef` = parked at bottom, content growth should follow. Cleared on
  // user-driven upward scroll; re-armed when they reach bottom again.
  // This is a shared ref — scrollToFn reads it to prevent the virtualizer's
  // measurement adjustments from fighting our pinToBottom.
  const lastTopRef = useRef(0)
  const lastHeightRef = useRef(0)
  const lastClientHeightRef = useRef(0)
  // Counter that tracks how many scroll events we expect to be ours rather
  // than the user's. `pinToBottom` writes `el.scrollTop`, which fires an
  // async `scroll` event; without this guard the on-scroll handler can race
  // with the programmatic write (because content also grew, the *resulting*
  // scrollTop can be lower than `lastTopRef` from the previous frame) and
  // misread the programmatic pin as the user scrolling up — which disarms
  // sticky-bottom and the user's just-submitted message slides above the
  // fold. See `apps/desktop/scripts/measure-jump.mjs` for the repro
  // (distFromBottom 0 → 49 within one frame, sticking forever).
  const programmaticScrollPendingRef = useRef(0)
  const prevSessionKeyRef = useRef(sessionKey)
  const prevGroupCountRef = useRef(0)

  const pinToBottom = useCallback(() => {
    const el = scrollerRef.current

    if (!el) {
      return
    }

    // Hold the disarm gate across the scroll event the next line will fire.
    programmaticScrollPendingRef.current += 1
    el.scrollTop = el.scrollHeight
    lastTopRef.current = el.scrollTop
    lastHeightRef.current = el.scrollHeight
    lastClientHeightRef.current = el.clientHeight
  }, [scrollerRef])

  const jumpToBottom = useCallback(() => {
    stickyBottomRef.current = true

    if (groupCount > 0) {
      virtualizer.scrollToIndex(groupCount - 1, { align: 'end', behavior: 'auto' })
    }

    requestAnimationFrame(() => {
      if (stickyBottomRef.current) {
        pinToBottom()
      }
    })
  }, [groupCount, pinToBottom, stickyBottomRef, virtualizer])

  useEffect(() => () => setThreadScrolledUp(false), [])

  // Track at-bottom state, dim composer when scrolled up, disarm on user
  // scroll/wheel/touch.
  useEffect(() => {
    const el = scrollerRef.current

    if (!el) {
      return undefined
    }

    const disarm = () => {
      stickyBottomRef.current = false
      programmaticScrollPendingRef.current = 0
    }

    const onScroll = () => {
      const top = el.scrollTop

      // If this scroll event is the consequence of `pinToBottom` writing
      // `el.scrollTop`, treat it as ours: don't disarm. The RO + rAF pin
      // loop will re-pin on the next frame if the browser clamped us
      // short of bottom (because content grew in the same frame).
      // Without this guard the post-pin scrollTop gets misread as the
      // user scrolling up, disarming sticky-bottom permanently and
      // leaving the just-submitted message below the fold.
      if (programmaticScrollPendingRef.current > 0) {
        programmaticScrollPendingRef.current -= 1
        lastTopRef.current = top
        lastHeightRef.current = el.scrollHeight
        lastClientHeightRef.current = el.clientHeight
        // Always re-arm — sticky-bottom should hold through clamp races.
        stickyBottomRef.current = true
        const atBottom = el.scrollHeight - (top + el.clientHeight) <= AT_BOTTOM_THRESHOLD
        setThreadScrolledUp(!atBottom)

        return
      }

      // Disarm only when `scrollTop` decreases while both content height and
      // viewport height are stable. A bare `top < lastTopRef.current` check is
      // unsafe: virtualizer measurement, streaming markdown, composer resizing,
      // window resizing, and toolbar/status updates can all move scrollTop as a
      // layout side effect. Wheel-up and touchmove still disarm immediately via
      // their own listeners below, so real user intent remains covered.
      const heightGrew = el.scrollHeight > lastHeightRef.current
      const clientHeightChanged = Math.abs(el.clientHeight - lastClientHeightRef.current) > 1
      if (!heightGrew && !clientHeightChanged && top + 1 < lastTopRef.current) {
        stickyBottomRef.current = false
      }

      lastTopRef.current = top
      lastHeightRef.current = el.scrollHeight
      lastClientHeightRef.current = el.clientHeight

      const atBottom = el.scrollHeight - (top + el.clientHeight) <= AT_BOTTOM_THRESHOLD

      if (atBottom) {
        stickyBottomRef.current = true
      }

      setThreadScrolledUp(!atBottom)
    }

    const onWheel = (event: WheelEvent) => {
      if (event.deltaY < 0) {
        disarm()
      }
    }

    el.addEventListener('scroll', onScroll, { passive: true })
    el.addEventListener('wheel', onWheel, { passive: true })
    el.addEventListener('touchmove', disarm, { passive: true })

    return () => {
      el.removeEventListener('scroll', onScroll)
      el.removeEventListener('wheel', onWheel)
      el.removeEventListener('touchmove', disarm)
    }
  }, [scrollerRef, stickyBottomRef])

  // Follow content growth (streaming, item measurements, loading indicator)
  // while armed. During fast streaming the ResizeObserver can fire many
  // times per frame as Streamdown re-tokenizes; coalesce to one pin per
  // animation frame so we don't run the scroll-event/re-pin chain
  // (~20+ ms self in `Virtualizer.getMaxScrollOffset`) several times per
  // token.
  useEffect(() => {
    if (!enabled || !isRunning) {
      return undefined
    }

    const el = scrollerRef.current

    if (!el) {
      return undefined
    }

    let pinRafScheduled = false
    const schedulePin = () => {
      if (pinRafScheduled || !stickyBottomRef.current) {
        return
      }
      pinRafScheduled = true
      requestAnimationFrame(() => {
        pinRafScheduled = false
        if (stickyBottomRef.current) {
          pinToBottom()
        }
      })
    }

    const observer = new ResizeObserver(schedulePin)

    // Observe ONLY the content (firstElementChild), not the scroller `el`
    // itself. Resizes of the viewport/scroller (window resize, devtools
    // panel toggle) shouldn't trigger a pin — only content growth should.
    if (el.firstElementChild) {
      observer.observe(el.firstElementChild)
    }

    return () => observer.disconnect()
  }, [enabled, isRunning, pinToBottom, scrollerRef, stickyBottomRef])

  // Jump to bottom on session change OR when an empty thread first gets
  // content. Both share the same intent and the same effect.
  useEffect(() => {
    const sessionChanged = prevSessionKeyRef.current !== sessionKey
    const becameNonEmpty = prevGroupCountRef.current === 0 && groupCount > 0

    prevSessionKeyRef.current = sessionKey
    prevGroupCountRef.current = groupCount

    if (enabled && (sessionChanged || becameNonEmpty)) {
      jumpToBottom()
    }
  }, [enabled, groupCount, jumpToBottom, sessionKey])

  // Pre-paint pin: when groupCount increases while armed (optimistic user
  // message insert, streaming assistant turn arriving, etc.), pin BEFORE
  // the browser commits the layout to screen. Using useLayoutEffect rather
  // than useEffect so this runs synchronously after React commits the DOM
  // mutation but before the browser paints. Without this, there's a ~50ms
  // visual window where the new message sits below the fold while we wait
  // for the ResizeObserver / scroll event chain to fire and re-pin.
  //
  // We pin TWICE in this critical path — once synchronously, then once on
  // the next rAF. The second pin catches the case where React mounts the
  // new message in the second commit (after our layout effect ran), which
  // grows scrollHeight again; without the rAF pin the user briefly sees a
  // ~15 px gap below the new message until the RO catches up. Streaming
  // tokens use the rate-limited RO path only; only the group-count change
  // (which fires once per user submit / new turn arrival) pays for the
  // extra pin.
  const prevGroupCountForLayoutRef = useRef(groupCount)
  useLayoutEffect(() => {
    if (!enabled) {
      return
    }
    if (groupCount > prevGroupCountForLayoutRef.current && stickyBottomRef.current) {
      // Defer to rAF so that browser scroll/wheel events from the current
      // frame are processed first.  Without this deferral, a trackpad
      // scroll-up during streaming can race with this effect: the wheel
      // event hasn't fired yet so stickyBottomRef is still true, and the
      // immediate pinToBottom() would snap the viewport back to bottom
      // against the user's intent.
      requestAnimationFrame(() => {
        if (stickyBottomRef.current) {
          pinToBottom()
        }
      })
    }
    prevGroupCountForLayoutRef.current = groupCount
  }, [enabled, groupCount, pinToBottom, stickyBottomRef])

  // Completion swaps streaming placeholders/plain code for final rendered DOM
  // (notably Shiki-highlighted code). Keep following the bottom briefly after
  // `isRunning` flips false so that final measurement pass cannot strand the
  // viewport near the top of a large code block.
  const prevIsRunningForLayoutRef = useRef(isRunning)
  useLayoutEffect(() => {
    const finishedRun = prevIsRunningForLayoutRef.current && !isRunning
    prevIsRunningForLayoutRef.current = isRunning

    if (!enabled || !finishedRun || !stickyBottomRef.current) {
      return undefined
    }

    const lockUntil = performance.now() + POST_RUN_BOTTOM_LOCK_MS
    let lockRaf: number | null = null

    const lockFrame = () => {
      lockRaf = null

      if (!stickyBottomRef.current) {
        return
      }

      pinToBottom()

      if (performance.now() < lockUntil) {
        lockRaf = requestAnimationFrame(lockFrame)
      }
    }

    pinToBottom()
    lockRaf = requestAnimationFrame(lockFrame)

    return () => {
      if (lockRaf !== null) {
        cancelAnimationFrame(lockRaf)
      }
    }
  }, [enabled, isRunning, pinToBottom, stickyBottomRef])

  useAuiEvent('thread.runStart', jumpToBottom)
}
