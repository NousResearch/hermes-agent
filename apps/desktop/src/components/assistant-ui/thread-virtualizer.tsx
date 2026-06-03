import { ThreadPrimitive, useAuiEvent, useAuiState } from '@assistant-ui/react'
import { useVirtualizer, type Virtualizer } from '@tanstack/react-virtual'
import { type ComponentProps, type FC, type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef } from 'react'

import { cn } from '@/lib/utils'
import { setThreadScrolledUp } from '@/store/thread-scroll'

const ESTIMATED_ITEM_HEIGHT = 220
const OVERSCAN = 4
const AT_BOTTOM_THRESHOLD = 4

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

export const VirtualizedThread: FC<VirtualizedThreadProps> = ({
  clampToComposer,
  components,
  emptyPlaceholder,
  loadingIndicator,
  sessionKey
}) => {
  const messageSignature = useAuiState(s =>
    s.thread.messages.map((message, index) => `${index}:${message.id}:${message.role}`).join('\n')
  )

  const groups = useMemo(() => buildGroups(messageSignature), [messageSignature])
  const renderEmpty = groups.length === 0 && Boolean(emptyPlaceholder)
  const scrollerRef = useRef<HTMLDivElement | null>(null)

  const virtualizer = useVirtualizer({
    count: groups.length,
    estimateSize: () => ESTIMATED_ITEM_HEIGHT,
    getItemKey: index => groups[index]?.id ?? index,
    getScrollElement: () => scrollerRef.current,
    // Seed the rect so the initial range mounts something before
    // `observeElementRect` reports the real layout (it overrides this).
    initialRect: { height: 600, width: 800 },
    overscan: OVERSCAN
  })

  useThreadScrollAnchor({
    enabled: !renderEmpty,
    groupCount: groups.length,
    scrollerRef,
    sessionKey: sessionKey ?? null,
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

interface ScrollAnchorOptions {
  enabled: boolean
  groupCount: number
  scrollerRef: React.RefObject<HTMLDivElement | null>
  sessionKey: string | null
  virtualizer: Virtualizer<HTMLDivElement, Element>
}

function useThreadScrollAnchor({ enabled, groupCount, scrollerRef, sessionKey, virtualizer }: ScrollAnchorOptions) {
  // `armed` = parked at bottom, content growth should follow. Cleared on
  // user-driven upward scroll; re-armed when they reach bottom again.
  const armedRef = useRef(true)
  const lastTopRef = useRef(0)
  const lastHeightRef = useRef(0)
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
  }, [scrollerRef])

  const jumpToBottom = useCallback(() => {
    armedRef.current = true

    if (groupCount > 0) {
      virtualizer.scrollToIndex(groupCount - 1, { align: 'end', behavior: 'auto' })
    }

    requestAnimationFrame(() => {
      if (armedRef.current) {
        pinToBottom()
      }
    })
  }, [groupCount, pinToBottom, virtualizer])

  useEffect(() => () => setThreadScrolledUp(false), [])

  // Track at-bottom state, dim composer when scrolled up, disarm on user
  // scroll/wheel/touch.
  useEffect(() => {
    const el = scrollerRef.current

    if (!el) {
      return undefined
    }

    const disarm = () => {
      armedRef.current = false
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
        // Always re-arm — sticky-bottom should hold through clamp races.
        armedRef.current = true
        const atBottom = el.scrollHeight - (top + el.clientHeight) <= AT_BOTTOM_THRESHOLD
        setThreadScrolledUp(!atBottom)

        return
      }

      // Disarm only when `scrollTop` decreases AND `scrollHeight` did NOT
      // grow this frame. A bare `top < lastTopRef.current` check is unsafe:
      // when content grows (virtualizer item measurement, streaming token,
      // code highlight re-tokenization, composer chip), the browser emits
      // an interim `scroll` event whose `scrollTop` is smaller than the
      // previous frame's because `scrollHeight` jumped — this fires before
      // the rAF-scheduled `pinToBottom` runs, so `programmaticScrollPendingRef`
      // is 0. Treating that as a user scroll permanently disarmed sticky-bottom
      // and produced the visible at-rest backward jump (#37997). Gating on a
      // stable `scrollHeight` keeps real user-driven upward intent — scrollbar
      // drag, keyboard PgUp, programmatic scrollIntoView — covered without
      // the false positive. Wheel-up and touchmove still disarm via their
      // own listeners below.
      const heightGrew = el.scrollHeight > lastHeightRef.current
      if (!heightGrew && top + 1 < lastTopRef.current) {
        armedRef.current = false
      }

      lastTopRef.current = top
      lastHeightRef.current = el.scrollHeight

      const atBottom = el.scrollHeight - (top + el.clientHeight) <= AT_BOTTOM_THRESHOLD

      if (atBottom) {
        armedRef.current = true
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
  }, [scrollerRef])

  // Follow content growth (streaming, item measurements, loading indicator)
  // while armed. During fast streaming the ResizeObserver can fire many
  // times per frame as Streamdown re-tokenizes; coalesce to one pin per
  // animation frame so we don't run the scroll-event/re-pin chain
  // (~20+ ms self in `Virtualizer.getMaxScrollOffset`) several times per
  // token.
  useEffect(() => {
    if (!enabled) {
      return undefined
    }

    const el = scrollerRef.current

    if (!el) {
      return undefined
    }

    let pinRafScheduled = false
    const schedulePin = () => {
      if (pinRafScheduled || !armedRef.current) {
        return
      }
      pinRafScheduled = true
      requestAnimationFrame(() => {
        pinRafScheduled = false
        if (armedRef.current) {
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
  }, [enabled, pinToBottom, scrollerRef])

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
    if (groupCount > prevGroupCountForLayoutRef.current && armedRef.current) {
      pinToBottom()
      requestAnimationFrame(() => {
        if (armedRef.current) {
          pinToBottom()
        }
      })
    }
    prevGroupCountForLayoutRef.current = groupCount
  }, [enabled, groupCount, pinToBottom])

  useAuiEvent('thread.runStart', jumpToBottom)
}
