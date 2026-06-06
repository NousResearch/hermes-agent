import { useAuiState } from '@assistant-ui/react'
import type { Virtualizer } from '@tanstack/react-virtual'
import { useCallback, useMemo, useRef, useState } from 'react'

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

interface MessageGroupLike {
  id: string
  indices?: number[]
  kind: 'standalone' | 'turn'
}

interface ConversationMinimapProps {
  groups: MessageGroupLike[]
  virtualizer: Virtualizer<HTMLDivElement, Element>
}

function extractMessageText(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }

  if (Array.isArray(content)) {
    return content
      .map(part => {
        if (typeof part === 'string') {
          return part
        }

        if (
          part != null &&
          typeof part === 'object' &&
          'text' in part &&
          typeof (part as { text: unknown }).text === 'string'
        ) {
          return (part as { text: string }).text
        }

        return ''
      })
      .join('')
  }

  return ''
}

const MAX_TRIGGER_MARKERS = 10

function sampleTriggerMarkers<T>(items: T[]): T[] {
  if (items.length <= MAX_TRIGGER_MARKERS) {
    return items
  }

  const half = Math.floor(MAX_TRIGGER_MARKERS / 2)

  return [...items.slice(0, half), ...items.slice(-half)]
}

export function ConversationMinimap({ groups, virtualizer }: ConversationMinimapProps) {
  const messages = useAuiState(s => s.thread.messages)
  const [open, setOpen] = useState(false)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const closeTimer = useRef<ReturnType<typeof setTimeout>>(null)

  const markers = useMemo(() => {
    return groups.reduce<
      { id: string; groupIndex: number; preview: string }[]
    >((acc, group, idx) => {
      if (group.kind !== 'turn') {
        return acc
      }

      const firstMsg = messages.find(m => m.id === group.id)
      const preview = firstMsg
        ? extractMessageText(firstMsg.content).trim().slice(0, 120)
        : ''

      acc.push({ id: group.id, groupIndex: idx, preview })

      return acc
    }, [])
  }, [groups, messages])

  const triggerMarkers = useMemo(() => sampleTriggerMarkers(markers), [markers])

  const popupHeight = useMemo(
    () => Math.min(288, Math.max(120, markers.length * 40 + 8)),
    [markers.length]
  )

  const cancelClose = useCallback(() => {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current)

      closeTimer.current = null
    }
  }, [])

  const scheduleClose = useCallback(() => {
    cancelClose()
    closeTimer.current = setTimeout(() => {
      setOpen(false)
      setHoveredId(null)
    }, 200)
  }, [cancelClose])

  const openPopover = useCallback(() => {
    cancelClose()
    setOpen(true)
  }, [cancelClose])

  const scrollTo = useCallback(
    (groupIndex: number, id: string) => {
      const el = virtualizer.scrollElement

      if (!el) {
        return
      }

      setSelectedId(id)

      const result = virtualizer.getOffsetForIndex(groupIndex, 'start')

      if (!result) {
        return
      }

      el.scrollTo({ top: result[0], behavior: 'smooth' })
    },
    [virtualizer]
  )

  if (markers.length <= 1) {
    return null
  }

  return (
    <div
      className="fixed right-4 top-1/2 z-50 -translate-y-1/2"
      onMouseEnter={openPopover}
      onMouseLeave={scheduleClose}
    >
      <Popover open={open}>
        <PopoverTrigger asChild>
          <div className="flex cursor-pointer flex-col items-center gap-1.5 rounded-lg px-1 py-2 transition-colors hover:bg-muted-foreground/5">
            {triggerMarkers.map(m => (
              <span
                key={m.id}
                className={cn(
                  'block h-[3px] w-4 rounded-full transition-all',
                  selectedId === m.id
                    ? 'scale-x-110 bg-foreground'
                    : hoveredId === m.id
                      ? 'bg-muted-foreground/60'
                      : 'bg-muted-foreground/25 hover:bg-muted-foreground/40'
                )}
              />
            ))}
          </div>
        </PopoverTrigger>
        <PopoverContent
          align="center"
          className="w-80 p-0"
          onMouseEnter={cancelClose}
          onMouseLeave={scheduleClose}
          side="left"
          sideOffset={-12}
        >
          <div className="minimap-scroll-list" style={{ height: `${popupHeight}px` }}>
            {markers.map(m => (
              <button
                key={m.id}
                className={cn(
                  'mb-0.5 block w-full flex-shrink-0 overflow-hidden text-ellipsis whitespace-nowrap rounded-md px-3 py-2 text-left text-sm leading-snug transition-colors last:mb-0',
                  'hover:bg-accent hover:text-accent-foreground',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  selectedId === m.id && 'bg-accent text-accent-foreground'
                )}
                onClick={() => scrollTo(m.groupIndex, m.id)}
                onMouseEnter={() => setHoveredId(m.id)}
                type="button"
              >
                {m.preview || (
                  <span className="italic text-muted-foreground">Empty message</span>
                )}
              </button>
            ))}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
