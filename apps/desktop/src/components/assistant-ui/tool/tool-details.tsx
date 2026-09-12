'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { LogView } from '@/components/ui/log-view'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { ChevronLeft, ChevronRight } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { parseMaybeObject, type ToolPart } from './fallback-model'

const DETAIL_WINDOW_CHARS = 20_000

export type ToolDetailSectionId = 'arguments' | 'command' | 'diff' | 'metadata' | 'result' | 'stderr' | 'stdout'

export interface ToolDetailSection {
  copyText: string
  id: ToolDetailSectionId
  state: 'empty' | 'present' | 'unavailable'
}

function serializeAvailableValue(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }

  const serialized = JSON.stringify(value, null, 2)

  return serialized === undefined ? String(value) : serialized
}

function section(id: ToolDetailSectionId, supplied: boolean, value: unknown): ToolDetailSection {
  if (!supplied) {
    return { copyText: '', id, state: 'unavailable' }
  }

  const copyText = serializeAvailableValue(value)

  return { copyText, id, state: copyText.length === 0 ? 'empty' : 'present' }
}

function own(record: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

/** Build full-fidelity sections only when the inspector is mounted. */
export function buildToolDetailSections(part: ToolPart, inlineDiff: string): ToolDetailSection[] {
  const args = parseMaybeObject(part.args)
  const result = parseMaybeObject(part.result)
  const commandKey = own(args, 'command') ? 'command' : own(args, 'code') ? 'code' : null
  const stdoutKey = own(result, 'stdout') ? 'stdout' : own(result, 'output') ? 'output' : null
  const diffKey = own(result, 'inline_diff') ? 'inline_diff' : own(result, 'diff') ? 'diff' : null

  const metadata = {
    toolName: part.toolName,
    ...(part.toolCallId === undefined ? {} : { toolCallId: part.toolCallId }),
    ...(part.isError === undefined ? {} : { isError: part.isError }),
    ...(part.timestamp === undefined ? {} : { timestamp: part.timestamp }),
    ...(part.completedAt === undefined ? {} : { completedAt: part.completedAt })
  }

  const sections = [
    section('arguments', part.args !== undefined, part.args),
    ...(commandKey ? [section('command', true, args[commandKey])] : []),
    section('result', part.result !== undefined, part.result),
    ...(stdoutKey ? [section('stdout', true, result[stdoutKey])] : []),
    ...(own(result, 'stderr') ? [section('stderr', true, result.stderr)] : []),
    ...(inlineDiff || diffKey ? [section('diff', true, inlineDiff || (diffKey ? result[diffKey] : ''))] : []),
    section('metadata', true, metadata)
  ]

  return sections
}

export interface TextMatch {
  end: number
  start: number
}

export function findTextMatches(text: string, query: string): TextMatch[] {
  if (!query) {
    return []
  }

  const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  const matches = Array.from(text.matchAll(new RegExp(escapedQuery, 'giu')), match => ({
    end: match.index + match[0].length,
    start: match.index
  }))

  return matches
}

interface ToolDetailsDialogProps {
  inlineDiff: string
  onOpenChange: (open: boolean) => void
  open: boolean
  part: ToolPart
}

export function ToolDetailsDialog({ inlineDiff, onOpenChange, open, part }: ToolDetailsDialogProps) {
  const { t } = useI18n()
  const copy = t.assistant.tool.details
  const [sections] = useState(() => buildToolDetailSections(part, inlineDiff))
  const [selectedId, setSelectedId] = useState<ToolDetailSectionId>('arguments')
  const [query, setQuery] = useState('')
  const [activeMatch, setActiveMatch] = useState(0)
  const [wrap, setWrap] = useState(true)
  const [windowStart, setWindowStart] = useState(0)
  const activeMatchRef = useRef<HTMLElement>(null)
  const selected = sections.find(item => item.id === selectedId) ?? sections[0]!
  const matches = useMemo(() => findTextMatches(selected.copyText, query), [query, selected.copyText])
  const normalizedMatch = matches.length ? Math.min(activeMatch, matches.length - 1) : 0
  const match = matches[normalizedMatch]
  const matchOffset = match?.start
  const maxWindowStart = Math.max(0, selected.copyText.length - DETAIL_WINDOW_CHARS)

  const searchStart =
    matchOffset === undefined
      ? windowStart
      : Math.min(maxWindowStart, Math.max(0, matchOffset - Math.floor(DETAIL_WINDOW_CHARS / 2)))

  const visibleStart = query ? searchStart : windowStart
  const visibleEnd = Math.min(selected.copyText.length, visibleStart + DETAIL_WINDOW_CHARS)

  const visibleText = selected.copyText.slice(visibleStart, visibleEnd)
  const visibleMatchOffset = matchOffset === undefined ? -1 : matchOffset - visibleStart
  const visibleMatchLength = Math.min(match ? match.end - match.start : 0, visibleText.length - visibleMatchOffset)
  const hasVisibleMatch = Boolean(query && visibleMatchOffset >= 0 && visibleMatchLength > 0)
  const hasMore = !query && visibleEnd < selected.copyText.length

  const sectionLabel = (id: ToolDetailSectionId) => copy.sections[id]

  useEffect(() => {
    activeMatchRef.current?.scrollIntoView?.({ block: 'center', inline: 'center' })
  }, [normalizedMatch, query, selected.id])

  const chooseSection = (id: ToolDetailSectionId) => {
    setSelectedId(id)
    setQuery('')
    setActiveMatch(0)
    setWindowStart(0)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent bodyClassName="overflow-hidden" className="h-[min(42rem,85vh)] max-w-4xl">
        <DialogHeader>
          <DialogTitle>{copy.title(part.toolName)}</DialogTitle>
          <DialogDescription>{copy.description}</DialogDescription>
        </DialogHeader>
        <div className="flex min-h-0 flex-1 flex-col gap-2">
          <div aria-label={copy.sectionsLabel} className="flex shrink-0 gap-1 overflow-x-auto" role="tablist">
            {sections.map(item => (
              <Button
                aria-selected={selected.id === item.id}
                key={item.id}
                onClick={() => chooseSection(item.id)}
                role="tab"
                size="xs"
                type="button"
                variant={selected.id === item.id ? 'secondary' : 'ghost'}
              >
                {sectionLabel(item.id)}
              </Button>
            ))}
          </div>
          <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
            <SearchField
              aria-label={copy.searchPlaceholder}
              containerClassName="min-w-48 flex-1"
              onChange={value => {
                setQuery(value)
                setActiveMatch(0)
              }}
              placeholder={copy.searchPlaceholder}
              trailingAction={
                query ? (
                  <span className="shrink-0 text-[0.65rem] tabular-nums text-(--ui-text-tertiary)">
                    {matches.length ? `${normalizedMatch + 1}/${matches.length}` : '0/0'}
                  </span>
                ) : null
              }
              value={query}
            />
            <div className="flex items-center gap-1">
              <Button
                aria-label={copy.previousMatch}
                disabled={!matches.length}
                onClick={() => setActiveMatch((normalizedMatch - 1 + matches.length) % matches.length)}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <ChevronLeft />
              </Button>
              <Button
                aria-label={copy.nextMatch}
                disabled={!matches.length}
                onClick={() => setActiveMatch((normalizedMatch + 1) % matches.length)}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <ChevronRight />
              </Button>
              <Button onClick={() => setWrap(value => !value)} size="xs" type="button" variant="ghost">
                {wrap ? copy.unwrap : copy.wrap}
              </Button>
              <CopyButton
                disabled={selected.state !== 'present'}
                label={copy.copySection(sectionLabel(selected.id))}
                text={selected.copyText}
              />
            </div>
          </div>
          <div className="min-h-0 flex-1" role="tabpanel">
            {selected.state === 'unavailable' ? (
              <div className="grid h-full place-items-center text-sm text-(--ui-text-tertiary)">{copy.unavailable}</div>
            ) : selected.state === 'empty' ? (
              <div className="grid h-full place-items-center text-sm text-(--ui-text-tertiary)">{copy.empty}</div>
            ) : (
              <LogView
                className={cn(
                  'h-full max-h-none overflow-auto text-(--ui-text-secondary)',
                  wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre break-normal'
                )}
              >
                {hasVisibleMatch ? (
                  <>
                    {visibleText.slice(0, visibleMatchOffset)}
                    <mark
                      className="bg-(--ui-row-active-background) text-inherit"
                      data-match-offset={matchOffset}
                      ref={activeMatchRef}
                    >
                      {visibleText.slice(visibleMatchOffset, visibleMatchOffset + visibleMatchLength)}
                    </mark>
                    {visibleText.slice(visibleMatchOffset + visibleMatchLength)}
                  </>
                ) : (
                  visibleText
                )}
              </LogView>
            )}
          </div>
          <div className="flex shrink-0 items-center justify-between gap-2 text-[0.65rem] text-(--ui-text-tertiary)">
            <span>{copy.receivedPayload}</span>
            {hasMore ? (
              <Button
                onClick={() => setWindowStart(value => value + DETAIL_WINDOW_CHARS)}
                size="xs"
                type="button"
                variant="ghost"
              >
                {copy.nextChunk}
              </Button>
            ) : null}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
