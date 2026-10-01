import type { ReactNode } from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Lock } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { describesNothingNew } from './derive-tools'
import { hintTags, tagCopy, vocabularyTag, type VocabularyTone } from './hint-vocabulary'
import type { ToolRowModel } from './types'

export const TOOL_ROW_HEIGHT = 30

const MAX_ROW_HINTS = 2

const TONE_CLASS = {
  danger: 'text-(--ui-red)',
  neutral: 'text-(--ui-text-secondary)',
  notice: 'text-(--ui-yellow)',
  unknown: 'text-(--ui-purple)'
} satisfies Record<VocabularyTone, string>

export interface ToolRowProps {
  expanded: boolean
  on: boolean
  onExpand: () => void
  onToggle: () => void
  preview?: boolean
  readOnly?: boolean
  tool: ToolRowModel
}

export function ToolRow({ expanded, on, onExpand, onToggle, preview = false, readOnly = false, tool }: ToolRowProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools
  const locked = tool.lockedBy !== null
  const struck = locked || tool.deprecated
  const facet = vocabularyTag(tool.facet)
  const facetCopy = tagCopy(facet, t.connectorsPage.vocabulary)
  const hints = hintTags(tool.hints, tool.facet)
  const expandable = !describesNothingNew(tool)

  const line = (
    <>
      <span
        className={cn(
          'min-w-0 truncate text-xs font-medium',
          struck ? 'text-(--ui-text-quaternary) line-through' : 'text-(--ui-text-primary)'
        )}
      >
        {tool.name}
      </span>

      <span className="flex shrink-0 items-center gap-1.5 text-[0.65rem] text-(--ui-text-quaternary)">
        {tool.facet === 'unclassified' ? null : (
          <Tip label={facetCopy.long}>
            <span className={cn('whitespace-nowrap', locked ? 'text-(--ui-text-quaternary)' : TONE_CLASS[facet.tone])}>
              {facetCopy.label}
            </span>
          </Tip>
        )}

        {locked ? (
          <span className="whitespace-nowrap">{copy.lockedHint}</span>
        ) : (
          <HintTags hints={hints.slice(0, MAX_ROW_HINTS)} more={Math.max(0, hints.length - MAX_ROW_HINTS)} />
        )}

        {expandable ? <DisclosureCaret className="text-(--ui-text-quaternary)" open={expanded} /> : null}
      </span>
    </>
  )

  return (
    <div
      className={cn('grid', locked && 'bg-muted/40', expanded && 'bg-(--ui-row-open-background)')}
      data-slot="tool-row"
      data-tool={tool.slug}
    >
      <div className="flex items-center gap-2.5 px-3.5" style={{ height: TOOL_ROW_HEIGHT }}>
        <span className="flex w-7 shrink-0 items-center">
          {preview ? null : locked ? (
            <Lock aria-hidden className="size-3 text-(--ui-text-quaternary)" />
          ) : (
            <Switch
              aria-label={on ? copy.turnToolOff(tool.name) : copy.turnToolOn(tool.name)}
              checked={on}
              disabled={readOnly}
              onCheckedChange={onToggle}
              size="xs"
            />
          )}
        </span>

        {expandable ? (
          <button
            aria-expanded={expanded}
            className="grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-center gap-2.5 text-left outline-none focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50"
            onClick={onExpand}
            type="button"
          >
            {line}
            <span className="sr-only">{expanded ? copy.hideDetails(tool.name) : copy.showDetails(tool.name)}</span>
          </button>
        ) : (
          <div className="grid min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-center gap-2.5">{line}</div>
        )}
      </div>

      {expanded ? <ToolDetail hints={hints} tool={tool} /> : null}
    </div>
  )
}

function HintTags({ hints, more }: { hints: ReturnType<typeof hintTags>; more: number }) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  return (
    <>
      {hints.map(hint => (
        <span className="whitespace-nowrap" key={hint.raw}>
          {tagCopy(hint, copy.vocabulary).label}
        </span>
      ))}
      {more > 0 ? <span className="whitespace-nowrap tabular-nums">{copy.tools.moreHints(more)}</span> : null}
    </>
  )
}

function ToolDetail({ hints, tool }: { hints: ReturnType<typeof hintTags>; tool: ToolRowModel }): ReactNode {
  const { t } = useI18n()

  return (
    <div className="grid gap-1 pb-2.5 pl-[3.625rem] pr-3.5">
      <p className="max-w-[60ch] text-[0.7rem] leading-relaxed text-(--ui-text-secondary)">{tool.description}</p>
      {hints.length > MAX_ROW_HINTS ? (
        <p className="text-[0.65rem] text-(--ui-text-quaternary)">
          {hints.map(hint => tagCopy(hint, t.connectorsPage.vocabulary).label).join(' · ')}
        </p>
      ) : null}
    </div>
  )
}
