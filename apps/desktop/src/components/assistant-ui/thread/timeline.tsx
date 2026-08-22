import { useAui, useAuiState } from '@assistant-ui/react'
import { type FC, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { requestRevealMessage } from '@/store/thread-scroll'

import {
  activeTimelineIndex,
  deriveTimelineEntries,
  sameTimelineEntries,
  type TimelineEntry,
  type TimelineSourceMessage
} from './timeline-data'

const MIN_ENTRIES = 4
const VIEWPORT = '[data-slot="aui_thread-viewport"]'
const HOVER_CLOSE_MS = 140

const ROW_CLASS =
  'row-hover relative flex w-full min-w-0 max-w-full select-none overflow-hidden rounded-md px-2 py-1 text-left outline-hidden'

// Surface (border-color/bg/shadow/blur) comes from the shared
// `[data-slot='thread-timeline-popover']` rule in styles.css, so it's 1:1 with
// the dropdown/select/dialog menus. We only own layout + the border/radius here.
const POPOVER_SHELL =
  'absolute right-full top-1/2 z-50 max-h-[min(22rem,calc(100vh-8rem))] w-80 max-w-[min(20rem,calc(100vw-2rem))] -translate-y-1/2 overflow-x-hidden overflow-y-auto overscroll-contain rounded-lg border p-1 text-popover-foreground transition-[opacity,transform] duration-100 ease-out group-hover/timeline:transition-none'

function userPromptText(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }

  if (!Array.isArray(content)) {
    return ''
  }

  let out = ''

  for (const part of content) {
    if (typeof part === 'string') {
      out += part

      continue
    }

    if (!part || typeof part !== 'object') {
      continue
    }

    const row = part as { text?: unknown; type?: unknown }

    if ((!row.type || row.type === 'text') && typeof row.text === 'string') {
      out += row.text
    }
  }

  return out
}

/** Index-keyed ref-array setter — `ref={listRef(refs, i)}`. */
const listRef =
  <T,>(refs: React.RefObject<(T | null)[]>, index: number) =>
  (node: T | null) => {
    refs.current[index] = node
  }

/** Mouse enter/leave pair forwarding `on` to the shared paint(). */
const hoverProps = (index: number, paint: (index: number, on: boolean) => void) => ({
  onMouseEnter: () => paint(index, true),
  onMouseLeave: () => paint(index, false)
})

// Constant-duration jump (eased), NOT native `behavior:'smooth'` — Chromium's
// smooth scroll animates proportional to distance, so jumping across a long
// thread crawls for seconds. A fixed ~260ms feels instant near or far. A
// shared rAF handle cancels a prior jump so rapid tick clicks don't fight.
let jumpRaf = 0

function jumpScroll(viewport: HTMLElement, top: number, duration = 170): void {
  cancelAnimationFrame(jumpRaf)
  const start = viewport.scrollTop
  const delta = top - start

  if (Math.abs(delta) < 2) {
    viewport.scrollTop = top

    return
  }

  const t0 = performance.now()
  const ease = (t: number) => 1 - (1 - t) ** 3 // easeOutCubic

  const step = (now: number) => {
    const p = Math.min(1, (now - t0) / duration)
    viewport.scrollTop = start + delta * ease(p)

    if (p < 1) {
      jumpRaf = requestAnimationFrame(step)
    }
  }

  jumpRaf = requestAnimationFrame(step)
}

// A timeline belongs to ONE chat surface, and several are mounted at once — side
// by side in a split, and stacked (hidden but kept alive) as inactive tabs. Walk
// up to this timeline's own surface before looking for the viewport; a
// document-wide lookup scrolls somebody else's thread.
export const ownViewport = (root: HTMLElement | null): HTMLElement | null =>
  (root?.closest('[data-session-anchor]') ?? document).querySelector<HTMLElement>(VIEWPORT)

function scrollToPrompt(root: HTMLElement | null, id: string): boolean {
  const viewport = ownViewport(root)
  const node = viewport?.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(id)}"]`)

  if (!viewport || !node) {
    return false
  }

  const top = viewport.scrollTop + (node.getBoundingClientRect().top - viewport.getBoundingClientRect().top) - 8

  triggerHaptic('selection')
  jumpScroll(viewport, Math.max(0, top))

  return true
}

const REVEAL_WAIT_FRAMES = 45

function jumpToPrompt(root: HTMLElement | null, id: string): void {
  if (scrollToPrompt(root, id)) {
    return
  }

  requestRevealMessage(id)

  let frames = 0

  const wait = () => {
    if (scrollToPrompt(root, id) || ++frames > REVEAL_WAIT_FRAMES) {
      return
    }

    requestRevealMessage(id)
    requestAnimationFrame(wait)
  }

  requestAnimationFrame(wait)
}

/**
 * Right-edge prompt rail — hover previews, click to jump. ≥4 user turns only.
 *
 * Everything here is DEFERRED until it can actually be seen. A chat surface
 * stays mounted while its tab is in the background (keep-alive, see
 * pane-visibility.ts), and a background thread keeps streaming, so a naive
 * timeline would re-derive previews and re-measure prompt offsets all day for
 * a rail nobody is looking at. Four gates, cheapest first:
 *
 *  1. INACTIVE PANE → render null and subscribe to nothing. The transcript
 *     selector, the scroll listener, and the popover markup all stand down.
 *  2. ACTIVE BUT UNHOVERED → the ticks paint, but the popover's rows are not
 *     built at all; the previews only exist once the pointer opens it.
 *  3. BELOW THE THRESHOLD → the rail renders null, so the measure effect never
 *     touches layout for it.
 *  4. FOLLOWING THE BOTTOM → the active prompt is the last one by definition,
 *     answered from data instead of a rect walk (see compute() below).
 */
export const ThreadTimeline: FC = () => {
  // Cheapest possible gate, and it must come first: an inactive tab returns
  // before any of the work below is even declared.
  return usePaneVisible() ? <ActiveThreadTimeline /> : null
}

/** Derived prompt rail for a VISIBLE surface. Split out so the hook body — and
 *  the transcript subscription it opens — never runs for a background tab. */
const ActiveThreadTimeline: FC = () => {
  // Cheap in the selector, expensive only when it changes: the ids alone tell
  // us whether the RAIL changed. Prompt text is immutable once sent, and an
  // edit rewinds the transcript (dropping every id after it) and re-appends a
  // fresh message id — so a preview can never go stale behind a stable id.
  // Streaming an assistant reply churns that message's content on every token
  // and leaves this string untouched, which is the whole point.
  const promptIds = useAuiState(s => {
    let ids = ''

    for (const message of s.thread.messages) {
      if (message.role === 'user') {
        ids += `${message.id}\n`
      }
    }

    return ids
  })

  // `promptIds` is the change signal; the transcript is read imperatively when
  // it fires, so the selector above never pays for text extraction. The client
  // goes through a ref so the memo keys on the SIGNAL alone — an accessor whose
  // identity churned would otherwise re-derive every render, which is exactly
  // the streaming cost this is here to avoid.
  const aui = useAui()
  const auiRef = useRef(aui)
  auiRef.current = aui

  const previousRef = useRef<TimelineEntry[]>([])

  const entries = useMemo(() => {
    const rows: TimelineSourceMessage[] = []

    for (const message of auiRef.current.thread().getState().messages) {
      if (message.role === 'user') {
        rows.push({ id: message.id, role: 'user', text: userPromptText(message.content) })
      }
    }

    const next = deriveTimelineEntries(rows)

    // Hand back the PREVIOUS array when nothing user-visible moved. Blank and
    // background-notification prompts are filtered out, so a new id can leave
    // the rail identical — without this, that re-renders both subtrees and
    // restarts the measure effect for no visible change.
    if (sameTimelineEntries(previousRef.current, next)) {
      return previousRef.current
    }

    previousRef.current = next

    return next
    // promptIds is the intentional re-eval TRIGGER, not a value the derivation
    // reads (the transcript comes off the ref) — same shape as ChatRoutesSurface's
    // gatewayState memo in app/contrib/controller.tsx.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [promptIds])

  const [activeIndex, setActiveIndex] = useState(0)
  const [open, setOpen] = useState(false)
  const closeTimerRef = useRef<number | undefined>(undefined)
  const rootRef = useRef<HTMLDivElement | null>(null)
  const jump = useCallback((id: string) => jumpToPrompt(rootRef.current, id), [])

  // Hover sync lives on the DOM, not in React state — the tick and its popover
  // row are siblings in different subtrees, so a shared index-keyed paint() lights
  // both without a re-render (and without coupling them through a parent atom).
  const tickRefs = useRef<(HTMLSpanElement | null)[]>([])
  const rowRefs = useRef<(HTMLButtonElement | null)[]>([])

  // Hover sync: light the tick + its popover row, and scroll that row into view
  // when the list overflows so the hovered prompt is always visible.
  const paint = useCallback((index: number, on: boolean) => {
    const tick = tickRefs.current[index]

    if (tick) {
      tick.style.opacity = on ? '1' : ''
    }

    const row = rowRefs.current[index]
    row?.classList.toggle('bg-(--ui-row-hover-background)', on)

    if (on) {
      row?.scrollIntoView({ block: 'nearest' })
    }
  }, [])

  const keepOpen = useCallback(() => {
    window.clearTimeout(closeTimerRef.current)
    setOpen(true)
  }, [])

  const closeSoon = useCallback(() => {
    window.clearTimeout(closeTimerRef.current)
    closeTimerRef.current = window.setTimeout(() => setOpen(false), HOVER_CLOSE_MS)
  }, [])

  useEffect(() => () => window.clearTimeout(closeTimerRef.current), [])

  useEffect(() => {
    // Below the threshold the rail renders null, so measuring prompt offsets
    // buys nothing — bail before touching layout at all.
    if (entries.length < MIN_ENTRIES) {
      return
    }

    const viewport = ownViewport(rootRef.current)

    if (!viewport) {
      return
    }

    let raf = 0

    const compute = () => {
      raf = 0

      // Pinned to the bottom (the entire streaming steady-state): the active
      // prompt is simply the last one. Skipping the walk matters — it reads a
      // rect per user message per scroll frame, and interleaved with React's
      // streaming style writes each read forces a full reflow (the single
      // hottest frame in the multitab profile).
      if (viewport.dataset.following === 'true') {
        setActiveIndex(prev => (prev === entries.length - 1 ? prev : entries.length - 1))

        return
      }

      const top = viewport.getBoundingClientRect().top

      const offsets = entries.map(entry => {
        const node = viewport.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(entry.id)}"]`)

        return node ? node.getBoundingClientRect().top - top : null
      })

      const next = activeTimelineIndex(offsets)

      setActiveIndex(prev => (prev === next ? prev : next))
    }

    const onScroll = () => {
      if (!raf) {
        raf = requestAnimationFrame(compute)
      }
    }

    // Initial compute rides the same rAF batching as scroll. A sync call here
    // reads getBoundingClientRect for every user message while other commit
    // effects are still writing styles — on a session switch that interleaving
    // forces a full reflow per read on a large transcript. One rAF later the
    // reads batch into a single layout pass, and back-to-back entries updates
    // (prefetch paint, then resume reconcile) coalesce into one compute.
    onScroll()
    viewport.addEventListener('scroll', onScroll, { passive: true })

    return () => {
      viewport.removeEventListener('scroll', onScroll)

      if (raf) {
        cancelAnimationFrame(raf)
      }
    }
  }, [entries])

  if (entries.length < MIN_ENTRIES) {
    return null
  }

  return (
    <div
      aria-label="Conversation timeline"
      className="group/timeline pointer-events-auto absolute right-0 top-1/2 z-40 flex -translate-y-1/2 flex-col items-end"
      data-slot="thread-timeline"
      data-suppress-pane-reveal=""
      onMouseEnter={keepOpen}
      onMouseLeave={closeSoon}
      ref={rootRef}
      role="navigation"
    >
      <TimelineTicks activeIndex={activeIndex} entries={entries} onHover={paint} onJump={jump} tickRefs={tickRefs} />
      <TimelinePopover
        activeIndex={activeIndex}
        entries={entries}
        onHover={paint}
        onJump={jump}
        open={open}
        rowRefs={rowRefs}
      />
    </div>
  )
}

const TimelinePopover: FC<{
  activeIndex: number
  entries: TimelineEntry[]
  onHover: (index: number, on: boolean) => void
  onJump: (id: string) => void
  open: boolean
  rowRefs: React.RefObject<(HTMLButtonElement | null)[]>
}> = ({ activeIndex, entries, onHover, onJump, open, rowRefs }) => {
  // The rail is the always-visible part; this list is not built until the
  // pointer first opens it. The SHELL always renders so the opacity/translate
  // transition has a node to animate — only the N rows are deferred, and they
  // stay mounted afterwards so the close fade still has content.
  const [everOpened, setEverOpened] = useState(open)

  if (open && !everOpened) {
    setEverOpened(true)
  }

  return (
    <div
      className={cn(
        POPOVER_SHELL,
        open ? 'pointer-events-auto opacity-100 translate-x-0' : 'pointer-events-none translate-x-1 opacity-0'
      )}
      data-slot="thread-timeline-popover"
    >
      {everOpened &&
        entries.map((entry, index) => (
          <button
            aria-label={entry.preview}
            className={cn(ROW_CLASS, index === activeIndex && 'bg-(--ui-row-active-background) text-foreground')}
            key={entry.id}
            onClick={() => onJump(entry.id)}
            ref={listRef(rowRefs, index)}
            type="button"
            {...hoverProps(index, onHover)}
          >
            <span className="block w-full min-w-0 truncate font-medium leading-snug text-foreground">
              {entry.preview}
            </span>
          </button>
        ))}
    </div>
  )
}

const TimelineTicks: FC<{
  activeIndex: number
  entries: TimelineEntry[]
  onHover: (index: number, on: boolean) => void
  onJump: (id: string) => void
  tickRefs: React.RefObject<(HTMLSpanElement | null)[]>
}> = ({ activeIndex, entries, onHover, onJump, tickRefs }) => (
  <div className="flex flex-col items-end py-1" data-slot="thread-timeline-ticks">
    {entries.map((entry, index) => (
      <button
        aria-label={entry.preview}
        className="flex h-2 w-7 cursor-pointer items-center justify-end pr-1"
        key={entry.id}
        onClick={() => onJump(entry.id)}
        type="button"
        {...hoverProps(index, onHover)}
      >
        <span
          className={cn(
            'block h-px w-3 transition-opacity duration-100 ease-out',
            index === activeIndex ? 'bg-(--theme-primary)' : 'dither text-(--ui-text-quaternary) opacity-70'
          )}
          ref={listRef(tickRefs, index)}
        />
      </button>
    ))}
  </div>
)
