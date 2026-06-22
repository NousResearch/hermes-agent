import { useAuiState } from '@assistant-ui/react'
import { type FC, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { composerPanelCard } from '@/components/chat/composer-dock'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { setPaneHoverRevealSuppressed } from '@/store/panes'

import {
  activeTimelineIndex,
  deriveTimelineEntries,
  type TimelineEntry,
  type TimelineSourceMessage
} from './thread-timeline-data'

const MIN_ENTRIES = 4
const VIEWPORT = '[data-slot="aui_thread-viewport"]'
const HOVER_CLOSE_MS = 140

const ROW_CLASS =
  'relative flex w-full min-w-0 max-w-full cursor-pointer select-none overflow-hidden rounded-md px-2 py-1 text-left outline-hidden transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:transition-none'

const POPOVER_SHELL = cn(
  'absolute right-full top-1/2 z-50 mr-1.5 max-h-[min(22rem,calc(100vh-8rem))] w-80 max-w-[min(20rem,calc(100vw-2rem))] -translate-y-1/2 overflow-x-hidden overflow-y-auto overscroll-contain p-1 text-popover-foreground transition-[opacity,transform] duration-100 ease-out group-hover/timeline:transition-none',
  composerPanelCard,
  // Solid fill — composerPanelCard is deliberately translucent; without this,
  // directive chips in the transcript bleed through and look like popover overflow.
  'bg-(--composer-fill)'
)

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

function scrollToPrompt(id: string) {
  const viewport = document.querySelector<HTMLElement>(VIEWPORT)
  const node = viewport?.querySelector<HTMLElement>(`[data-message-id="${CSS.escape(id)}"]`)

  if (!viewport || !node) {
    return
  }

  const top = viewport.scrollTop + (node.getBoundingClientRect().top - viewport.getBoundingClientRect().top) - 8

  triggerHaptic('selection')
  viewport.scrollTo({ behavior: 'smooth', top: Math.max(0, top) })
}

/** Right-edge prompt rail — hover previews, click to jump. ≥4 user turns only. */
export const ThreadTimeline: FC = () => {
  const sourceSignature = useAuiState(s => {
    const rows: TimelineSourceMessage[] = []

    for (const message of s.thread.messages) {
      if (message.role !== 'user') {
        continue
      }

      rows.push({ id: message.id, role: 'user', text: userPromptText(message.content) })
    }

    return JSON.stringify(rows)
  })

  const entries = useMemo(
    () => deriveTimelineEntries(JSON.parse(sourceSignature) as TimelineSourceMessage[]),
    [sourceSignature]
  )

  const [activeIndex, setActiveIndex] = useState(0)
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const [open, setOpen] = useState(false)
  const closeTimerRef = useRef<number | undefined>(undefined)

  const keepOpen = useCallback(() => {
    window.clearTimeout(closeTimerRef.current)
    setPaneHoverRevealSuppressed(true)
    setOpen(true)
  }, [])

  const closeSoon = useCallback(() => {
    window.clearTimeout(closeTimerRef.current)
    setHoverIndex(null)
    setPaneHoverRevealSuppressed(false)
    closeTimerRef.current = window.setTimeout(() => setOpen(false), HOVER_CLOSE_MS)
  }, [])

  useEffect(
    () => () => {
      window.clearTimeout(closeTimerRef.current)
      setPaneHoverRevealSuppressed(false)
    },
    []
  )

  useEffect(() => {
    if (entries.length < MIN_ENTRIES) {
      setPaneHoverRevealSuppressed(false)
    }
  }, [entries.length])

  useEffect(() => {
    const viewport = document.querySelector<HTMLElement>(VIEWPORT)

    if (!viewport || entries.length === 0) {
      return
    }

    let raf = 0

    const compute = () => {
      raf = 0

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

    compute()
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
      onMouseEnter={keepOpen}
      onMouseLeave={closeSoon}
      role="navigation"
    >
      <TimelineTicks
        activeIndex={activeIndex}
        entries={entries}
        onHover={setHoverIndex}
        onJump={scrollToPrompt}
      />
      <TimelinePopover
        activeIndex={activeIndex}
        entries={entries}
        hoverIndex={hoverIndex}
        onHover={setHoverIndex}
        onJump={scrollToPrompt}
        open={open}
      />
    </div>
  )
}

const TimelinePopover: FC<{
  activeIndex: number
  entries: TimelineEntry[]
  hoverIndex: number | null
  onHover: (index: number) => void
  onJump: (id: string) => void
  open: boolean
}> = ({ activeIndex, entries, hoverIndex, onHover, onJump, open }) => (
  <div
    className={cn(
      POPOVER_SHELL,
      open ? 'pointer-events-auto opacity-100 translate-x-0' : 'pointer-events-none translate-x-1 opacity-0'
    )}
    data-slot="thread-timeline-popover"
  >
    {entries.map((entry, index) => {
      const hovered = index === hoverIndex
      const active = index === activeIndex

      return (
        <button
          aria-label={entry.preview}
          className={cn(
            ROW_CLASS,
            active && 'bg-(--ui-row-active-background) text-foreground',
            hovered && 'bg-(--ui-row-hover-background) text-foreground transition-none'
          )}
          key={entry.id}
          onClick={() => onJump(entry.id)}
          onMouseEnter={() => onHover(index)}
          type="button"
        >
          <span className="block w-full min-w-0 truncate font-medium leading-snug text-foreground">
            {entry.preview}
          </span>
        </button>
      )
    })}
  </div>
)

const TimelineTicks: FC<{
  activeIndex: number
  entries: TimelineEntry[]
  onHover: (index: number) => void
  onJump: (id: string) => void
}> = ({ activeIndex, entries, onHover, onJump }) => (
  <div className="flex flex-col items-end py-1" data-slot="thread-timeline-ticks">
    {entries.map((entry, index) => (
      <button
        aria-label={entry.preview}
        className="group/tick flex h-2 w-7 cursor-pointer items-center justify-end pr-1"
        key={entry.id}
        onClick={() => onJump(entry.id)}
        onMouseEnter={() => onHover(index)}
        type="button"
      >
        <span
          className={cn(
            'block h-px w-3 transition-opacity duration-100 ease-out',
            index === activeIndex
              ? 'bg-(--theme-primary)'
              : 'dither text-(--ui-text-quaternary) opacity-70 group-hover/tick:opacity-100 group-hover/tick:transition-none'
          )}
        />
      </button>
    ))}
  </div>
)
