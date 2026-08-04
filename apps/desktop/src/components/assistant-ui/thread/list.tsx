import { ThreadPrimitive, useAuiEvent, useAuiState } from '@assistant-ui/react'
import {
  type ComponentProps,
  type CSSProperties,
  type FC,
  memo,
  type ReactNode,
  startTransition,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { type GetTargetScrollTop, useStickToBottom } from 'use-stick-to-bottom'

import { useI18n } from '@/i18n'
import { messageRenderWeight } from '@/lib/render-weight'
import { cn } from '@/lib/utils'
import {
  onScrollToBottomRequest,
  onThreadEditClose,
  onThreadEditOpen,
  resetThreadScroll,
  setThreadAtBottom
} from '@/store/thread-scroll'
import { isSecondaryWindow } from '@/store/windows'

import { MessageRenderBoundary } from '../message-render-boundary'

import { resolveShowEarlierAction, useTranscriptWindow } from './transcript-window'

type ThreadMessageComponents = ComponentProps<typeof ThreadPrimitive.MessageByIndex>['components']

export type MessageGroup = { id: string; weight: number } & (
  { index: number; kind: 'standalone' } | { indices: number[]; kind: 'turn' }
)

// DOM is bounded by a render-cost budget, not a message/turn count. Every part
// costs one unit, and large strings add another unit per 512 characters. Parts
// approximate component/node count; characters approximate markdown parsing,
// text-node allocation, and tool-result formatting. Counting only parts badly
// underpriced a 51KB tool result as "1", so a handful of huge results let a
// 600KB transcript through the old 300-part cap and could drive Chromium's renderer
// into a GC crash.
//
// "Show earlier" prepends another page; whole turns stay intact so the sticky
// human bubble never loses its turn. This is the long-session perf lever WITHOUT
// a virtualizer — pure rendering, never touches scrollTop, so it can't fight
// use-stick-to-bottom (the single scroll owner).
const RENDER_BUDGET = 300
// On session switch, paint a small budget first (enough for the bottom turn(s)
// the user actually sees after scroll-to-bottom), then bump to the full budget
// in a requestAnimationFrame — defers the heavy markdown+syntax-highlight render
// past the initial commit, so the switch feels instant.
//
// 20, down from 60: the first-paint commit is synchronous and uninterruptible,
// and at 60 cost units it measured 627ms on a real session (LoAF: block=575ms, no
// attributed script — pure commit). A viewport after scroll-to-bottom shows
// 1-2 normal turns ≈ 10-20 units; the transition backfill below fills the rest
// interruptibly, so the only thing a smaller budget changes is how much work
// blocks the click-to-paint path.
const FIRST_PAINT_BUDGET = 20

// Browsers may quantize a requested scrollTop to a nearby device-pixel
// boundary. use-stick-to-bottom otherwise compares the lower actual value to
// the integer target forever, re-requesting the same instant scroll every
// frame. Treat a subpixel remainder as achieved; larger gaps still follow new
// streamed content normally.
const SCROLL_TARGET_EPSILON_PX = 0.5

export const resolveThreadScrollTarget: GetTargetScrollTop = (targetScrollTop, { scrollElement }) => {
  const currentScrollTop = scrollElement.scrollTop
  const remaining = targetScrollTop - currentScrollTop

  return remaining >= 0 && remaining <= SCROLL_TARGET_EPSILON_PX ? currentScrollTop : targetScrollTop
}

interface ThreadMessageListProps {
  clampToComposer: boolean
  components: ThreadMessageComponents
  emptyPlaceholder?: ReactNode
  loadingIndicator?: ReactNode
  sessionKey?: string | null
}

// Group each user message with the assistant turn(s) that follow it so the
// human bubble can `position: sticky` against the scroller across its whole
// turn (see StickyHumanMessageContainer in thread.tsx).
export function buildGroups(signature: string): MessageGroup[] {
  if (!signature) {
    return []
  }

  const messages = signature.split('\n').map(row => {
    const [index, id, role, weight] = row.split(':')

    return { id, index: Number(index), role, weight: Number(weight) || 1 }
  })

  const groups: MessageGroup[] = []

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]

    if (message.role !== 'user') {
      groups.push({ id: message.id, index: message.index, kind: 'standalone', weight: message.weight })

      continue
    }

    const indices = [message.index]
    let weight = message.weight

    while (i + 1 < messages.length && messages[i + 1].role !== 'user') {
      weight += messages[++i].weight
      indices.push(messages[i].index)
    }

    groups.push({ id: message.id, indices, kind: 'turn', weight })
  }

  return groups
}

// Walk turns newest-first, summing their render weights until the budget is met;
// everything before the first kept turn is hidden. Returns the index of that
// first visible group.
export function firstVisibleGroupIndex(groups: readonly MessageGroup[], budget: number): number {
  let firstVisible = groups.length

  for (let i = groups.length - 1, weight = 0; i >= 0; i--) {
    weight += groups[i].weight
    firstVisible = i

    if (weight >= budget) {
      break
    }
  }

  return firstVisible
}

// content-visibility:auto skips off-screen turns for perf, but with
// contain-intrinsic-size:auto the browser only remembers a turn's size AFTER
// it has rendered. A turn that finishes streaming near the bottom may have had
// its (smaller) mid-stream size remembered; when it scrolls just off the top
// edge and gets skipped, it snaps back to that stale height, shifting content
// down. With overflow-anchor:none (the viewport can't self-correct) the
// stick-to-bottom lock drifts and the view creeps up over older turns — the
// "long session eventually shows old responses" glitch.
//
// Keep the newest turns always-rendered so a turn is only ever virtualized
// once its layout has settled at its final size (remembered == real → skipping
// it changes no height). Off-screen OLDER turns still skip, so the dialog/popover
// recalc win on long transcripts is preserved.
//
// The tail is budgeted in render-cost units, not turns, because that is what the
// cost actually scales with — the same currency as RENDER_BUDGET /
// FIRST_PAINT_BUDGET.
// A turn-count tail silently defeats itself on agent transcripts: one tool-heavy
// turn is 50-200 units, so a 6-TURN tail exempted the entire visible transcript
// and nothing virtualized at all. Measured on a 5-tile window (7/3/5/3/2 groups
// per tile): zero content-visibility containers were active, and every Radix
// overlay open paid the full ~610ms whole-document recalc that #66470 fixed.
//
// 40 units ≈ the 1-2 turns a viewport shows after scroll-to-bottom (the same
// reasoning as FIRST_PAINT_BUDGET=20, doubled so a turn that grows mid-stream
// doesn't fall out of the tail as it settles).
export const LIVE_TAIL_PARTS = 40
// Floor: always exempt at least this many turns regardless of weight, so a
// transcript of very heavy turns still keeps the streaming one unvirtualized.
export const LIVE_TAIL_MIN_GROUPS = 2
// Ceiling: never exempt more than this many turns, however light they are. On a
// long transcript of tiny turns a weight-only budget would walk back further
// than the old turn-count tail did and virtualize LESS — this keeps the new
// policy a strict improvement on every shape.
export const LIVE_TAIL_MAX_GROUPS = 6

/**
 * Index of the newest group that still virtualizes — everything at or after it
 * is the live tail and stays rendered. Walks newest-first accumulating weight,
 * so the tail covers a viewport's worth of content rather than a fixed number
 * of turns, clamped to [MIN, MAX] turns. Computed once per render, not per row.
 */
export function liveTailStart(
  groups: readonly MessageGroup[],
  tailWeight = LIVE_TAIL_PARTS,
  minGroups = LIVE_TAIL_MIN_GROUPS,
  maxGroups = LIVE_TAIL_MAX_GROUPS
): number {
  let weight = 0
  let start = groups.length

  for (let i = groups.length - 1; i >= 0; i--) {
    weight += groups[i]?.weight ?? 1
    start = i

    if (weight > tailWeight) {
      break
    }
  }

  // Clamp the tail to [minGroups, maxGroups] turns: the floor keeps the live
  // turn rendered when turns are huge, the ceiling stops a tail of tiny turns
  // from sprawling past what the old turn-count policy rendered.
  const floor = Math.max(0, groups.length - minGroups)
  const ceiling = Math.max(0, groups.length - maxGroups)

  return Math.min(floor, Math.max(ceiling, start))
}

const ThreadMessageListInner: FC<ThreadMessageListProps> = ({
  clampToComposer,
  components,
  emptyPlaceholder,
  loadingIndicator,
  sessionKey
}) => {
  // TWO signatures, deliberately split. The STRUCTURAL one (ids/roles/count)
  // changes only when messages are added/removed/swapped — it keys the error
  // boundaries and the row identity. The WEIGHT one (parts + character cost)
  // ticks while a streaming turn appends content — it feeds only the render
  // budget. Folding weights into the structural key handed every boundary a
  // new resetKey per appended part, which reconciled every turn's subtree on
  // every tick (measured: 540 wasted Block renders per explain() sample with
  // two threads streaming).
  const structuralSignature = useAuiState(s =>
    s.thread.messages.map((message, index) => `${index}:${message.id}:${message.role}`).join('\n')
  )

  const weightSignature = useAuiState(s =>
    s.thread.messages.map(message => messageRenderWeight(message.content)).join(',')
  )

  const { t } = useI18n()
  // Row structure is memoized on the STRUCTURAL signature only, so streaming
  // part-appends can't churn group identity (that would defeat the rows memo
  // below on every tick). Weights are folded in separately for the budget.
  const groups = useMemo(() => buildGroups(structuralSignature), [structuralSignature])
  const renderEmpty = groups.length === 0 && Boolean(emptyPlaceholder)

  // use-stick-to-bottom owns scrollTop (single writer): follow while locked,
  // escape on user scroll-up, re-lock at bottom. Snap instantly, not spring — a
  // spring can't tell live-token growth from a session-switch bulk relayout, and
  // chasing the latter reads as the view scrolling to random spots before
  // settling. Its refs hang off our own DOM so the sticky human bubbles survive.
  const { scrollRef, contentRef, isAtBottom, scrollToBottom, stopScroll } = useStickToBottom({
    initial: 'instant',
    resize: 'instant',
    targetScrollTop: resolveThreadScrollTarget
  })

  const { olderAvailable, expandWindow } = useTranscriptWindow()

  const [renderBudget, setRenderBudget] = useState(FIRST_PAINT_BUDGET)

  // Cut the budget during RENDER, not in the post-commit layout effect. An
  // effect-time cut is too late: React would first build the whole tree with
  // the full budget (up to 300 cost units of markdown + syntax highlighting),
  // commit it, and only then re-render at the small budget. The render-phase
  // state adjustment restarts this component immediately — before any child
  // renders — so the heavy commit never happens.
  //
  // Two triggers, because the transcript swap arrives differently per path:
  // a WARM switch publishes sessionKey + messages in one commit (the key
  // branch), while a COLD switch changes sessionKey with an empty transcript
  // and the prefetched messages land hundreds of ms later under the SAME key
  // (the empty→non-empty branch).
  const hasGroups = groups.length > 0
  const [budgetSessionKey, setBudgetSessionKey] = useState(sessionKey)
  const [hadGroups, setHadGroups] = useState(hasGroups)

  if (budgetSessionKey !== sessionKey) {
    setBudgetSessionKey(sessionKey)
    setHadGroups(hasGroups)
    setRenderBudget(FIRST_PAINT_BUDGET)
  } else if (hadGroups !== hasGroups) {
    setHadGroups(hasGroups)

    if (hasGroups) {
      setRenderBudget(FIRST_PAINT_BUDGET)
    }
  }

  // Where to land after a prepend, in distance-from-bottom (survives the
  // height change). Shared by "Show earlier" and the budget backfill below.
  const restoreFromBottomRef = useRef<number | null>(null)
  // False from a session switch until the settle loop below parks the
  // transcript at its true bottom. While false, scrollTop is a way-point of a
  // load in progress, not a reading position anyone chose — never anchor to it.
  const loadSettledRef = useRef(false)
  // Session the settle loop last armed for, so a re-arm within the same load
  // is distinguishable from a switch to a different transcript.
  const settleKeyRef = useRef(sessionKey)

  // Record where the view should land once a prepend has grown the content,
  // measured from the BOTTOM so the added height doesn't invalidate it. Only a
  // settled load has an offset the user chose; mid-load the answer is simply
  // the bottom.
  const anchorBeforePrepend = useCallback(() => {
    const el = scrollRef.current

    restoreFromBottomRef.current = el && loadSettledRef.current ? el.scrollHeight - el.scrollTop : 0
  }, [scrollRef])

  // Backfill from FIRST_PAINT_BUDGET to the full budget after the small
  // commit painted — as a TRANSITION, so the heavy markdown + syntax
  // highlight render of the older turns is interruptible instead of one long
  // synchronous commit that freezes input right after the switch. Route
  // changes stay urgent (main.tsx disables router transitions); it's exactly
  // this backfill that belongs at background priority. "Show earlier" pages
  // (budget > RENDER_BUDGET) never re-enter here.
  useEffect(() => {
    if (renderBudget >= RENDER_BUDGET) {
      return
    }

    const rafId = requestAnimationFrame(() => {
      // The backfill PREPENDS older turns, so everything on screen slides down
      // by their height. Anchor first and let the restore effect below re-apply
      // it in the same commit the taller tree lands in — otherwise the view is
      // stranded near the TOP until use-stick-to-bottom's ResizeObserver
      // catches up a frame or two later (measured: an 11.5k px jump showing
      // ~160ms of unrelated old turns, on every session load).
      anchorBeforePrepend()

      // Functional max, not a plain set: an urgent "Show earlier" click can
      // land between scheduling and committing this transition, and a plain
      // set would rebase over it and shrink the budget back down.
      startTransition(() => setRenderBudget(budget => Math.max(budget, RENDER_BUDGET)))
    })

    return () => cancelAnimationFrame(rafId)
  }, [anchorBeforePrepend, renderBudget])

  // Weights (part count + visible character cost) fold into the BUDGET only.
  // Group identity stays structural, so a streaming append re-runs this cheap
  // sum — not the row JSX. Settled content hits messageRenderWeight's WeakMap.
  const weightedGroups = useMemo(() => {
    const weights = weightSignature.split(',').map(w => Number(w) || 1)

    return groups.map(group => ({
      ...group,
      weight:
        group.kind === 'turn'
          ? group.indices.reduce((sum, index) => sum + (weights[index] ?? 1), 0)
          : (weights[group.index] ?? 1)
    }))
  }, [groups, weightSignature])

  const hiddenCount = firstVisibleGroupIndex(weightedGroups, renderBudget)
  const visibleGroups = hiddenCount > 0 ? groups.slice(hiddenCount) : groups

  // Where the always-rendered live tail begins. Derived from the WEIGHTED
  // groups (render cost, not turns) so the tail is a viewport's worth of content —
  // see liveTailStart. Computed once here rather than per row.
  const tailStart = useMemo(
    () => liveTailStart(hiddenCount > 0 ? weightedGroups.slice(hiddenCount) : weightedGroups),
    [weightedGroups, hiddenCount]
  )

  // Secondary windows (new-session scratch, subagent watch, cmd-click pop-out)
  // hide the titlebar tool cluster + session header, but the OS traffic lights
  // still sit in the top-left, so reserve the titlebar gap above the transcript.
  const secondaryWindow = isSecondaryWindow()
  // NB: CSS calc() requires whitespace around the +/- operator. This string is
  // assigned verbatim to the --sticky-human-top inline style below (it does not
  // go through Tailwind, which would auto-space it), so the spaces are load-
  // bearing — without them the declaration is invalid, gets dropped, and the
  // sticky user bubble falls back to its ~4px default and slides under the OS
  // traffic lights.
  const secondaryTitlebarGap = 'calc(var(--titlebar-height) + 0.75rem)'

  const threadContentTopPad = secondaryWindow
    ? 'pt-[calc(var(--titlebar-height)+0.75rem)]'
    : 'pt-[calc(var(--titlebar-height)-0.5rem)]'

  useEffect(() => setThreadAtBottom(isAtBottom), [isAtBottom])
  useEffect(() => () => resetThreadScroll(), [])

  // Floating jump button (outside this subtree) → return to the bottom.
  useEffect(() => onScrollToBottomRequest(() => void scrollToBottom()), [scrollToBottom])

  const endEditHold = useCallback(() => {
    scrollRef.current?.removeAttribute('data-editing')
  }, [scrollRef])

  // Inline edit grows a sticky bubble. Escape before focus/layout so the
  // resize-follow can't snap scrollTop; native anchoring holds the viewport.
  const beginEditHold = useCallback(() => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    endEditHold()
    stopScroll()
    el.setAttribute('data-editing', 'true')
  }, [endEditHold, scrollRef, stopScroll])

  useEffect(() => onThreadEditOpen(beginEditHold), [beginEditHold])
  useEffect(() => onThreadEditClose(endEditHold), [endEditHold])
  useEffect(() => () => endEditHold(), [endEditHold])
  // New run → snap to the latest turn.
  useAuiEvent('thread.runStart', () => void scrollToBottom())

  // Reset the cap and pin to bottom on mount + every session switch (messages
  // swap in place on a long-lived runtime, so sessionKey is the only signal).
  // The swap is multi-step and lays out over many frames; letting the library
  // follow re-pins every frame to a moving target — visible as ~10 scroll jumps.
  // Instead: quiet it, glue to the true bottom until the height holds steady,
  // then hand back locked. Live streaming afterward uses the normal resize follow.
  //
  // `hasGroups` joins sessionKey as a dep because a COLD load changes the key
  // while the transcript is still empty and publishes messages hundreds of ms
  // later. Keyed on the switch alone the loop measured an EMPTY viewport, saw
  // a stable height in two frames, and handed back "settled" before the
  // transcript existed — so the turns painted at scrollTop 0 and only snapped
  // down once use-stick-to-bottom's ResizeObserver noticed, a full-viewport
  // lurch on every cold load. The empty→non-empty flip re-arms for the
  // transcript that actually arrived; being a boolean, it cannot re-fire on a
  // streaming append.
  useLayoutEffect(() => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    stopScroll()
    el.scrollTop = el.scrollHeight
    loadSettledRef.current = false

    // An anchor captured for the OUTGOING transcript must not be applied to
    // this one — a switch owns the position outright. The empty→non-empty
    // re-arm is the SAME load, whose in-flight anchor is still correct.
    if (settleKeyRef.current !== sessionKey) {
      settleKeyRef.current = sessionKey
      restoreFromBottomRef.current = null
    }

    let frame = 0
    let stableFrames = 0
    let lastHeight = el.scrollHeight

    const settle = () => {
      const node = scrollRef.current

      if (!node) {
        return
      }

      const height = node.scrollHeight

      stableFrames = height === lastHeight ? stableFrames + 1 : 0
      lastHeight = height
      node.scrollTop = height

      // Most session switches are synchronous and stabilize within 2 frames;
      // the old 90-frame ceiling was for slow async image loads. Cap at 15
      // frames to minimize the settle-loop racing markdown paint on every switch.
      if (stableFrames >= 2 || ++frame > 15) {
        void scrollToBottom('instant')
        loadSettledRef.current = true

        return
      }

      rafId = requestAnimationFrame(settle)
    }

    let rafId = requestAnimationFrame(settle)

    return () => cancelAnimationFrame(rafId)
  }, [hasGroups, scrollRef, scrollToBottom, sessionKey, stopScroll])

  // Prepend an older page while preserving the on-screen position. The user is
  // scrolled up (reading history) so the stick-to-bottom lock is escaped and
  // won't fight this manual restore. Spend the already-materialized DOM page
  // first; only when that is exhausted pull more messages out of the session
  // store (#55191).
  const showEarlier = useCallback(() => {
    const action = resolveShowEarlierAction(hiddenCount, olderAvailable)

    if (!action) {
      return
    }

    anchorBeforePrepend()

    if (action === 'dom') {
      setRenderBudget(budget => budget + RENDER_BUDGET)

      return
    }

    expandWindow()
  }, [anchorBeforePrepend, expandWindow, hiddenCount, olderAvailable])

  useLayoutEffect(() => {
    const el = scrollRef.current

    if (el && restoreFromBottomRef.current != null) {
      el.scrollTop = el.scrollHeight - restoreFromBottomRef.current
      restoreFromBottomRef.current = null
    }
    // renderBudget covers DOM pages; groups.length covers store-window expands.
  }, [scrollRef, renderBudget, groups.length])

  // The row array is memoized on the inputs the rows actually read. This
  // component re-renders on every isAtBottom flip — and use-stick-to-bottom
  // flips it from a ResizeObserver, so a sidebar DRAG re-renders this list per
  // frame. Without the memo, the inline .map() rebuilt every row's JSX each
  // time, and rebuilt children re-render their whole subtree even when nothing
  // changed (measured live: 865 wasted Block renders in one drag, walked to
  // "MessageRenderBoundary (children only)" by explain()). With it, React
  // bails out on element identity and a scroll flip re-renders nothing below.
  const rows = useMemo(
    () =>
      visibleGroups.map((group, indexInVisible) => (
        // content-visibility:auto — off-screen turns skip style recalc,
        // layout, and paint. On a long transcript this is what keeps
        // UNRELATED UI fast: any dialog/popover mount (Radix Presence
        // reads getComputedStyle) forces a whole-document style recalc,
        // measured ~650-730ms per open on a 1300-message session and
        // ~100-200ms with this on. contain-intrinsic-size keeps a
        // placeholder height for never-rendered turns (auto: remembered
        // real size once rendered), so scrollbar/anchoring stay stable.
        // Sticky human bubbles are unaffected — their turn is rendered
        // whenever any part of it intersects the viewport.
        //
        // The live tail (newest turns) is exempt: virtualizing a turn
        // whose final size hasn't been remembered yet snaps it to a stale
        // height when it scrolls off, drifting stick-to-bottom up over old
        // turns. See liveTailStart.
        <div
          className={cn(
            'flex min-w-0 flex-col gap-(--conversation-turn-gap) pb-(--conversation-turn-gap)',
            indexInVisible < tailStart && '[contain-intrinsic-size:auto_37.5rem] [content-visibility:auto]'
          )}
          key={group.id}
        >
          <MessageRenderBoundary resetKey={structuralSignature}>
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
          </MessageRenderBoundary>
        </div>
      )),
    [visibleGroups, components, structuralSignature, tailStart]
  )

  return (
    <div
      className="relative min-h-0 max-w-full overflow-hidden contain-[layout_paint]"
      style={
        {
          height: clampToComposer ? 'var(--thread-viewport-height)' : '100%',
          ...(secondaryWindow ? { '--sticky-human-top': secondaryTitlebarGap } : {})
        } as CSSProperties
      }
    >
      {secondaryWindow && (
        // Secondary windows hide the titlebar chrome, so the scroller runs to
        // the window's top edge and streamed text slides up under the OS
        // traffic lights. Content padding alone scrolls away with the text — a
        // fixed opaque strip (the titlebar's drag region) masks anything behind
        // it and keeps the window draggable, matching the main window's header.
        <div
          aria-hidden="true"
          className="absolute inset-x-0 top-0 z-10 h-(--titlebar-height) bg-background [-webkit-app-region:drag]"
        />
      )}
      <div
        className="size-full overflow-x-hidden overflow-y-auto overscroll-contain"
        data-following={isAtBottom ? 'true' : 'false'}
        data-slot="aui_thread-viewport"
        ref={scrollRef as React.RefCallback<HTMLDivElement>}
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
            className={cn('mx-auto flex w-full max-w-(--composer-width) min-w-0 flex-col px-6', threadContentTopPad)}
            data-slot="aui_thread-content"
            ref={contentRef as React.RefCallback<HTMLDivElement>}
          >
            {(hiddenCount > 0 || olderAvailable) && (
              <button
                className="mx-auto mb-(--conversation-turn-gap) rounded-full border border-border/65 bg-(--composer-fill) px-3 py-1 text-xs text-muted-foreground hover:text-foreground"
                onClick={showEarlier}
                type="button"
              >
                {t.assistant.thread.showEarlier}
              </button>
            )}
            {rows}
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

export const ThreadMessageList = memo(ThreadMessageListInner)
