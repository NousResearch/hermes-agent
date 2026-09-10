'use client'

import { useMemo, useState } from 'react'

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
const SEARCH_CONTEXT_CHARS = 2_000
const MAX_SEARCH_MATCHES = 1_000

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

export function findTextMatches(text: string, query: string): number[] {
  const needle = query.toLocaleLowerCase()

  if (!needle) {
    return []
  }

  const haystack = text.toLocaleLowerCase()
  const matches: number[] = []
  let offset = 0

  while (matches.length < MAX_SEARCH_MATCHES) {
    const index = haystack.indexOf(needle, offset)

    if (index < 0) {
      break
    }

    matches.push(index)
    offset = index + Math.max(needle.length, 1)
  }

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
  const sections = useMemo(() => buildToolDetailSections(part, inlineDiff), [inlineDiff, part])
  const [selectedId, setSelectedId] = useState<ToolDetailSectionId>('arguments')
  const [query, setQuery] = useState('')
  const [activeMatch, setActiveMatch] = useState(0)
  const [wrap, setWrap] = useState(true)
  const [visibleChars, setVisibleChars] = useState(DETAIL_WINDOW_CHARS)
  const selected = sections.find(item => item.id === selectedId) ?? sections[0]!
  const matches = useMemo(() => findTextMatches(selected.copyText, query), [query, selected.copyText])
  const normalizedMatch = matches.length ? Math.min(activeMatch, matches.length - 1) : 0
  const matchOffset = matches[normalizedMatch]
  const searchStart = matchOffset === undefined ? 0 : Math.max(0, matchOffset - SEARCH_CONTEXT_CHARS)
  const searchEnd = Math.min(selected.copyText.length, searchStart + DETAIL_WINDOW_CHARS)
  const visibleText = query ? selected.copyText.slice(searchStart, searchEnd) : selected.copyText.slice(0, visibleChars)
  const hasMore = !query && visibleChars < selected.copyText.length
  const sectionLabel = (id: ToolDetailSectionId) => copy.sections[id]

  const chooseSection = (id: ToolDetailSectionId) => {
    setSelectedId(id)
    setQuery('')
    setActiveMatch(0)
    setVisibleChars(DETAIL_WINDOW_CHARS)
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
                <ChevronLeft className="size-3.5" />
              </Button>
              <Button
                aria-label={copy.nextMatch}
                disabled={!matches.length}
                onClick={() => setActiveMatch((normalizedMatch + 1) % matches.length)}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <ChevronRight className="size-3.5" />
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
                {visibleText}
              </LogView>
            )}
          </div>
          <div className="flex shrink-0 items-center justify-between gap-2 text-[0.65rem] text-(--ui-text-tertiary)">
            <span>{copy.receivedPayload}</span>
            {hasMore ? (
              <Button
                onClick={() => setVisibleChars(value => value + DETAIL_WINDOW_CHARS)}
                size="xs"
                type="button"
                variant="ghost"
              >
                {copy.loadMore}
              </Button>
            ) : null}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
