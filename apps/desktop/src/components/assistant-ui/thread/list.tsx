import { ThreadPrimitive, type ThreadMessage, useAuiEvent, useAuiState } from '@assistant-ui/react'
import {
  type ComponentProps,
  createContext,
  type CSSProperties,
  type FC,
  memo,
  type ReactNode,
  startTransition,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { type GetTargetScrollTop, useStickToBottom } from 'use-stick-to-bottom'

import { useI18n } from '@/i18n'
import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { messagePaintWeight } from '@/lib/render-weight'
import { cn } from '@/lib/utils'
import {
  $threadScrolledUp,
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

/**
 * Revision-bound acknowledgement of a committed Thread message-list render.
 * The semantic signature covers the complete non-optimistic chain while the
 * leaf reports only after every required rendered message part has committed.
 */
export interface ThreadCommitReceipt {
  revision: number
  chainSignature: string
  headMessage: ThreadMessage | null
  contentSignature: string
  complete: boolean
}

type ThreadMessageCommitReport = {
  message: ThreadMessage
  part?: object
}

type ThreadMessageCommitReporter = (report: ThreadMessageCommitReport) => void

const ThreadMessageCommitContext = createContext<ThreadMessageCommitReporter | null>(null)

/** Report the commit boundary for a message row itself. */
export function useReportThreadMessageCommit(waitForParts = false): void {
  const report = useContext(ThreadMessageCommitContext)
  const message = useAuiState(s => s.message as unknown as ThreadMessage)
  const partCount = useAuiState(s => s.message.parts.length)

  useLayoutEffect(() => {
    if (!waitForParts || partCount === 0) {
      report?.({ message })
    }
  }, [message, partCount, report, waitForParts])
}

/** Report from a part component that owns its actual DOM commit boundary. */
export function useReportThreadMessagePartCommit(): void {
  const report = useContext(ThreadMessageCommitContext)
  const message = useAuiState(s => s.message as unknown as ThreadMessage)
  const part = useAuiState(s => s.part as unknown as object)

  useLayoutEffect(() => {
    report?.({ message, part })
  }, [message, part, report])
}

/** Bind a nested markdown renderer's receipt to the current part projection. */
export function useThreadMessagePartCommitCallback(
  expectedText?: string
): ((committedText: string) => void) | undefined {
  const report = useContext(ThreadMessageCommitContext)
  const message = useAuiState(s => s.message as unknown as ThreadMessage)
  const part = useAuiState(s => s.part as unknown as object)
  const renderedText = expectedText ?? (part as { text?: unknown }).text

  return useMemo(
    () =>
      report
        ? (committedText: string) => {
            if (renderedText === committedText) {
              report({ message, part })
            }
          }
        : undefined,
    [message, part, renderedText, report]
  )
}

/** A group renderer acknowledges parts that deliberately have no mounted leaf. */
export function useThreadMessagePartRangeCommitCallback(
  startIndex: number,
  endIndex: number
): (() => void) | undefined {
  const report = useContext(ThreadMessageCommitContext)
  const message = useAuiState(s => s.message as unknown as ThreadMessage)
  const parts = useAuiState(s => s.message.parts as unknown as readonly object[])

  return useMemo(
    () =>
      report
        ? () => {
            for (let index = Math.max(0, startIndex); index <= endIndex; index += 1) {
              const part = parts[index]

              if (part) {
                report({ message, part })
              }
            }
          }
        : undefined,
    [endIndex, message, parts, report, startIndex]
  )
}

export function commitReceiptContentSignature(messages: readonly ThreadMessage[]): string {
  return JSON.stringify(
    messages.map(message => [message.id, message.role, message.content, message.status ?? null] as const)
  )
}

export function commitReceiptChain(messages: readonly ThreadMessage[]): {
  chainSignature: string
  headMessage: ThreadMessage | null
  contentSignature: string
} {
  const committed: ThreadMessage[] = []

  for (const message of messages) {
    if (message.metadata?.isOptimistic !== true) {
      committed.push(message)
    }
  }

  const headMessage = committed.at(-1) ?? null

  return {
    chainSignature: committed.map(message => message.id).join('\n'),
    headMessage,
    contentSignature: commitReceiptContentSignature(committed)
  }
}

export function partRequiresCommit(part: object): boolean {
  const type = (part as { type?: string }).type

  return type === 'text' || type === 'reasoning'
}

export function pruneCommitMap<T>(map: Map<string, T>, messages: readonly { id: string }[]): void {
  const currentIds = new Set(messages.map(message => message.id))

  for (const id of map.keys()) {
    if (!currentIds.has(id)) {
      map.delete(id)
    }
  }
}

/** The first-paint render is complete once no groups are hidden, or after the
 * full DOM budget's backfill has committed. */
export function isThreadRenderComplete(hiddenCount: number, renderBudget: number): boolean {
  return hiddenCount === 0 || renderBudget >= RENDER_BUDGET
}

export type MessageGroup = { id: string; weight: number } & (
  { index: number; kind: 'standalone' } | { indices: number[]; kind: 'turn' }
)

// DOM is bounded by a render-cost budget, not a message/turn count. The
// currency is `messagePaintWeight`: what a turn actually MOUNTS, which is what
// the grouping decides rather than what the payload weighs. A settled run of
// twelve reads is one grey summary line, a thought is one collapsed
// disclosure, a hoisted `todo` is nothing — while a diff, an image card or a
// wall of markdown really does build DOM and is charged for it.
//
// Pricing by payload instead had the budget counting work that never mounts:
// one tool-heavy turn measured 84-281 units of tool JSON that painted as a
// dozen one-line summaries, so a session spent the whole page in two or three
// turns and offered "Show earlier" over a screen and a half of transcript.
//
// "Show earlier" prepends another page; whole turns stay intact so the sticky
// human bubble never loses its turn. This is the long-session perf lever WITHOUT
// a virtualizer — pure rendering, never touches scrollTop, so it can't fight
// use-stick-to-bottom (the single scroll owner).
//
// 600 units ≈ 10-20 agentic turns on measured real sessions (a tool-heavy turn
// prices at 30-90, a plain exchange at 5-10), and a whole session of ordinary
// work now fits one page instead of paging three times to reach its start.
// What the DOM can hold is bounded above by the store window regardless
// (TRANSCRIPT_WINDOW_BUDGET), so this cannot admit more than one window's
// content.
export const RENDER_BUDGET = 600
// Never offer "Show earlier" over fewer turns than this, however heavy they
// are. A weight-only cut on a session of enormous turns put the button two
// turns from the bottom, where it reads as broken rather than as paging — the
// user has not been given enough transcript to have gone looking for more. The
// store window caps what the DOM can reach at all, so a floor here stays
// bounded.
const MIN_VISIBLE_GROUPS = 8
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
  onCommitReceipt?: (receipt: ThreadCommitReceipt) => void
  resumePublicationRevision?: number
  sessionKey?: string | null
}

export function shouldRestoreResumeAnchor(
  previousRevision: number,
  nextRevision: number,
  followingBottom: boolean
): boolean {
  return nextRevision > previousRevision && followingBottom
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
// everything before the first kept turn is hidden. `minVisible` turns are kept
// regardless of weight. Returns the index of that first visible group.
export function firstVisibleGroupIndex(groups: readonly MessageGroup[], budget: number, minVisible = 0): number {
  let firstVisible = groups.length

  for (let i = groups.length - 1, weight = 0; i >= 0; i--) {
    weight += groups[i].weight
    firstVisible = i

    if (weight >= budget) {
      break
    }
  }

  return Math.min(firstVisible, Math.max(0, groups.length - minVisible))
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
  onCommitReceipt,
  resumePublicationRevision = 0,
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
    s.thread.messages.map(message => messagePaintWeight(message.content)).join(',')
  )

  // Identity of the non-optimistic semantic chain rendered by this leaf.
  // Skip its full serialization outside the short-lived reveal handshake.
  const threadMessages = useAuiState(s => s.thread.messages)
  const paneVisible = usePaneVisible()
  const {
    chainSignature: committedChainSignature,
    headMessage: committedHeadMessage,
    contentSignature: committedContentSignature
  } = useMemo(
    () =>
      onCommitReceipt
        ? commitReceiptChain(threadMessages)
        : { chainSignature: '', headMessage: null, contentSignature: '[]' },
    [onCommitReceipt, threadMessages]
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
  const lastResumePublicationRevisionRef = useRef(resumePublicationRevision)

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
  // sum — not the row JSX. Settled content hits messagePaintWeight's WeakMap.
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

  // The turn floor applies to a real page only. During the first-paint budget
  // the point is a small synchronous commit; forcing 8 turns into it would put
  // back exactly the freeze FIRST_PAINT_BUDGET exists to avoid, and the rAF
  // backfill a frame later fills them in anyway.
  const hiddenCount = firstVisibleGroupIndex(
    weightedGroups,
    renderBudget,
    renderBudget >= RENDER_BUDGET ? MIN_VISIBLE_GROUPS : 0
  )

  const visibleGroups = hiddenCount > 0 ? groups.slice(hiddenCount) : groups

  const renderedMessages = useMemo(
    () =>
      visibleGroups
        .flatMap(group =>
          group.kind === 'turn' ? group.indices.map(index => threadMessages[index]) : [threadMessages[group.index]]
        )
        .filter(
          (message): message is NonNullable<typeof message> =>
            Boolean(message) && message.metadata?.isOptimistic !== true
        ),
    [threadMessages, visibleGroups]
  )

  const onCommitReceiptRef = useRef(onCommitReceipt)
  onCommitReceiptRef.current = onCommitReceipt

  const expectedCommitRef = useRef({
    revision: resumePublicationRevision,
    chainSignature: committedChainSignature,
    headMessage: committedHeadMessage,
    contentSignature: committedContentSignature,
    complete: isThreadRenderComplete(hiddenCount, renderBudget),
    renderedMessages
  })
  expectedCommitRef.current = {
    revision: resumePublicationRevision,
    chainSignature: committedChainSignature,
    headMessage: committedHeadMessage,
    contentSignature: committedContentSignature,
    complete: isThreadRenderComplete(hiddenCount, renderBudget),
    renderedMessages
  }

  // An empty authoritative chain has no message or part leaf that can report a
  // commit. Acknowledge it from the real list layout boundary, but only while
  // the pane is visible: a receipt emitted while hidden is intentionally
  // ignored by ChatView and must be re-issued on the reveal commit.
  useLayoutEffect(() => {
    if (!paneVisible || !onCommitReceipt) {
      return
    }

    const expected = expectedCommitRef.current

    if (
      expected.complete &&
      expected.renderedMessages.length === 0 &&
      expected.chainSignature === '' &&
      expected.headMessage === null &&
      expected.contentSignature === '[]'
    ) {
      onCommitReceipt({
        revision: expected.revision,
        chainSignature: '',
        headMessage: null,
        contentSignature: '[]',
        complete: true
      })
    }
  }, [
    committedChainSignature,
    committedContentSignature,
    committedHeadMessage,
    hiddenCount,
    onCommitReceipt,
    paneVisible,
    renderBudget,
    renderedMessages,
    resumePublicationRevision
  ])

  const committedLeafObjectsRef = useRef(
    new Map<string, { message: ThreadMessage; messageCommitted: boolean; parts: Set<object> }>()
  )
  pruneCommitMap(committedLeafObjectsRef.current, renderedMessages)

  const reportMessageCommit = useCallback(({ message, part }: ThreadMessageCommitReport) => {
    let committed = committedLeafObjectsRef.current.get(message.id)

    if (!committed || committed.message !== message) {
      committed = { message, messageCommitted: false, parts: new Set<object>() }
      committedLeafObjectsRef.current.set(message.id, committed)
    }

    if (part) {
      committed.parts.add(part)
    } else {
      committed.messageCommitted = true
    }

    const expected = expectedCommitRef.current

    if (
      expected.complete &&
      expected.renderedMessages.every(renderedMessage => {
        const leaf = committedLeafObjectsRef.current.get(renderedMessage.id)

        if (!leaf || leaf.message !== renderedMessage) {
          return false
        }

        const parts = (renderedMessage as unknown as { parts?: readonly object[] }).parts ?? []
        const publicationSensitiveParts = renderedMessage.role === 'assistant' ? parts.filter(partRequiresCommit) : []

        return leaf.messageCommitted && publicationSensitiveParts.every(renderedPart => leaf.parts.has(renderedPart))
      })
    ) {
      onCommitReceiptRef.current?.({
        revision: expected.revision,
        chainSignature: expected.chainSignature,
        headMessage: expected.headMessage,
        contentSignature: expected.contentSignature,
        complete: true
      })
    }
  }, [])

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

  // Waking from display: hidden (HUD mode hides the main window; OS hide does
  // the same to any window): rAF and ResizeObserver were frozen the whole
  // time, so the virtualizer's measurements — and scrollTop itself — are
  // stale. If the user was following the bottom, re-anchor once visible;
  // leave a scrolled-up reader exactly where they were.
  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === 'visible' && !$threadScrolledUp.get()) {
        requestAnimationFrame(() => void scrollToBottom())
      }
    }

    document.addEventListener('visibilitychange', onVisible)

    return () => document.removeEventListener('visibilitychange', onVisible)
  }, [scrollToBottom])

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

  // A cached transcript can be followed by a genuinely different persisted-
  // authority publication under the same session key. The revision may commit
  // before assistant-ui publishes the corresponding message DOM. Cover both
  // schedules: restore immediately for a same-commit resize, and arm one
  // MutationObserver for a following DOM commit. Mutation callbacks run before
  // paint, unlike use-stick-to-bottom's ResizeObserver + rAF correction.
  useLayoutEffect(() => {
    const previousRevision = lastResumePublicationRevisionRef.current
    lastResumePublicationRevisionRef.current = resumePublicationRevision

    const el = scrollRef.current

    if (
      !el ||
      !shouldRestoreResumeAnchor(previousRevision, resumePublicationRevision, el.dataset.following === 'true')
    ) {
      return
    }

    stopScroll()
    const restoreIfFollowing = () => {
      if (el.dataset.following === 'true') {
        el.scrollTop = el.scrollHeight
      }
    }

    restoreIfFollowing()

    const observer = new MutationObserver(() => {
      restoreIfFollowing()
      observer.disconnect()
    })
    observer.observe(el, { childList: true, characterData: true, subtree: true })

    const timeout = window.setTimeout(() => observer.disconnect(), 250)

    return () => {
      window.clearTimeout(timeout)
      observer.disconnect()
    }
  }, [resumePublicationRevision, scrollRef, stopScroll])

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
    <ThreadMessageCommitContext.Provider value={onCommitReceipt ? reportMessageCommit : null}>
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
    </ThreadMessageCommitContext.Provider>
  )
}

export const ThreadMessageList = memo(ThreadMessageListInner)
