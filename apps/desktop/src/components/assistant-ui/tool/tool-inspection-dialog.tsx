import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { LogView } from '@/components/ui/log-view'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import type { ToolPart } from './fallback-model/types'
import {
  INSPECTION_PAGE_CHARS,
  type InspectionSection,
  type InspectionSectionId,
  inspectionSections,
  inspectionText,
  inspectionWindow
} from './tool-inspection-model'

const MAX_SEARCH_MATCHES = 1_000

function searchMatches(text: string | undefined, query: string): { offsets: number[]; truncated: boolean } {
  if (text === undefined || !query) {
    return { offsets: [], truncated: false }
  }

  const offsets: number[] = []
  let from = 0

  while (offsets.length < MAX_SEARCH_MATCHES) {
    const index = text.indexOf(query, from)

    if (index < 0) {
      return { offsets, truncated: false }
    }

    offsets.push(index)
    from = index + Math.max(query.length, 1)
  }

  return { offsets, truncated: text.indexOf(query, from) >= 0 }
}

function InspectionText({ section }: { section: InspectionSection }) {
  const { t } = useI18n()
  const copy = t.assistant.tool.inspector
  const text = useMemo(() => inspectionText(section.value), [section.value])
  const [offset, setOffset] = useState(0)
  const [query, setQuery] = useState('')
  const [activeMatch, setActiveMatch] = useState(0)
  const [wrap, setWrap] = useState(true)
  const matches = useMemo(() => searchMatches(text, query), [query, text])
  const normalizedMatch = matches.offsets.length ? Math.min(activeMatch, matches.offsets.length - 1) : 0
  const match = matches.offsets[normalizedMatch] ?? -1
  const shown = inspectionWindow(text ?? '', offset)

  function moveToMatch(index: number) {
    const next = matches.offsets[index]

    if (next === undefined) {
      return
    }

    setActiveMatch(index)
    setOffset(Math.max(0, next - 200))
  }

  function changeQuery(value: string) {
    setQuery(value)
    setActiveMatch(0)

    const first = !value || text === undefined ? -1 : text.indexOf(value)
    setOffset(first >= 0 ? Math.max(0, first - 200) : 0)
  }

  const unavailable = text === undefined
  const empty = text === ''
  const highlightStart = Math.max(0, match - shown.start)
  const highlightEnd = Math.min(shown.text.length, match + query.length - shown.start)
  const highlighted = query && match >= shown.start && match < shown.end
  const matchCount = query
    ? matches.offsets.length
      ? `${normalizedMatch + 1}/${matches.offsets.length}${matches.truncated ? '+' : ''}`
      : '0/0'
    : ''

  return (
    <div className="grid min-h-0 min-w-0 gap-2">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <SearchField aria-label={copy.search} onChange={changeQuery} placeholder={copy.search} value={query} />
        {query && (
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground" role="status">
            {matchCount}
          </span>
        )}
        <Button
          disabled={!matches.offsets.length}
          onClick={() => moveToMatch((normalizedMatch - 1 + matches.offsets.length) % matches.offsets.length)}
          size="sm"
          variant="ghost"
        >
          {copy.previousMatch}
        </Button>
        <Button
          disabled={!matches.offsets.length}
          onClick={() => moveToMatch((normalizedMatch + 1) % matches.offsets.length)}
          size="sm"
          variant="ghost"
        >
          {copy.nextMatch}
        </Button>
        <Button aria-pressed={wrap} onClick={() => setWrap(value => !value)} size="sm" variant="ghost">
          {copy.wrap}
        </Button>
        <CopyButton disabled={unavailable || empty} label={copy.copySection} text={text ?? ''} />
      </div>
      {query && match < 0 && <p role="status">{copy.noMatch}</p>}
      <LogView
        aria-label={copy.sections[section.id]}
        className={cn(
          'h-[45vh] min-h-0 text-(--ui-text-secondary)',
          wrap ? 'whitespace-pre-wrap wrap-anywhere' : 'whitespace-pre break-normal'
        )}
        role="region"
        tabIndex={0}
      >
        {unavailable ? (
          section.value === undefined ? (
            copy.unavailable
          ) : (
            copy.serializationFailed
          )
        ) : empty ? (
          copy.empty
        ) : highlighted ? (
          <>
            {shown.text.slice(0, highlightStart)}
            <mark>{shown.text.slice(highlightStart, highlightEnd)}</mark>
            {shown.text.slice(highlightEnd)}
          </>
        ) : (
          shown.text
        )}
      </LogView>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          disabled={shown.start === 0}
          onClick={() => setOffset(Math.max(0, shown.start - INSPECTION_PAGE_CHARS))}
          size="sm"
          variant="ghost"
        >
          {copy.previousPage}
        </Button>
        <span className="text-xs text-muted-foreground" role="status">
          {copy.range(empty || unavailable ? 0 : shown.start + 1, shown.end, text?.length ?? 0)}
        </span>
        <Button
          disabled={shown.end >= (text?.length ?? 0)}
          onClick={() => setOffset(shown.end)}
          size="sm"
          variant="ghost"
        >
          {copy.nextPage}
        </Button>
      </div>
    </div>
  )
}

export default function ToolInspectionDialog({
  part,
  inlineDiff,
  onClose,
  trigger
}: {
  part: ToolPart
  inlineDiff: string
  onClose: () => void
  trigger: HTMLButtonElement
}) {
  const { t } = useI18n()
  const copy = t.assistant.tool.inspector
  const sections = useMemo(() => inspectionSections(part, inlineDiff), [part, inlineDiff])
  const [selected, setSelected] = useState<InspectionSectionId>(part.result === undefined ? 'args' : 'result')
  const section = sections.find(value => value.id === selected) ?? sections[0]

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {
          onClose()
        }
      }}
      open
    >
      <DialogContent
        className="w-[90vw] max-w-4xl"
        onCloseAutoFocus={event => {
          event.preventDefault()

          if (trigger.isConnected) {
            trigger.focus()
          }
        }}
      >
        <DialogHeader>
          <DialogTitle>
            {copy.title} · {part.toolName}
          </DialogTitle>
          <DialogDescription>{copy.notice}</DialogDescription>
        </DialogHeader>
        <div className="max-w-full overflow-x-auto">
          <SegmentedControl
            onChange={setSelected}
            options={sections.map(value => ({ id: value.id, label: copy.sections[value.id] }))}
            value={selected}
          />
        </div>
        <InspectionText key={section.id} section={section} />
      </DialogContent>
    </Dialog>
  )
}
