// One tool, one line. The name is the line; the slug and the description live
// behind the disclosure, because a list of 896 rows is scanned, not read.
//
// Every lane is a fixed width with `shrink-0`, so the facet words and the hint
// tags line up down the whole list instead of drifting with the name.

import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { Switch } from '@/components/ui/switch'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Lock } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { facetTag, hintTags, tagCopy, type VocabularyTone } from './hint-vocabulary'
import type { ToolRowModel } from './types'

/** The windowing arithmetic depends on every closed row being exactly this tall. */
export const TOOL_ROW_HEIGHT = 30

/** Read stays in the text ramp on purpose: colouring the commonest facet would
 *  make a mostly-harmless list look busy, and the loud ones would stop reading
 *  as loud. */
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
  tool: ToolRowModel
}

export function ToolRow({ expanded, on, onExpand, onToggle, tool }: ToolRowProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools
  const locked = tool.lockedBy !== null
  const struck = locked || tool.deprecated
  const facet = facetTag(tool.facet)
  const facetCopy = tagCopy(facet, t.connectorsPage.vocabulary)
  const hints = hintTags(tool.hints)

  return (
    <div
      className={cn('grid', locked && 'bg-muted/40', expanded && 'bg-(--ui-row-open-background)')}
      data-slot="tool-row"
      data-tool={tool.slug}
    >
      <div className="flex items-center gap-2.5 px-3.5" style={{ height: TOOL_ROW_HEIGHT }}>
        <span className="flex w-7 shrink-0 items-center">
          {locked ? (
            <Lock aria-hidden className="size-3 text-(--ui-text-quaternary)" />
          ) : (
            <Switch
              aria-label={on ? copy.turnToolOff(tool.name) : copy.turnToolOn(tool.name)}
              checked={on}
              onCheckedChange={onToggle}
              size="xs"
            />
          )}
        </span>

        <button
          aria-expanded={expanded}
          className="flex min-w-0 flex-1 items-center gap-2.5 text-left outline-none focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50"
          onClick={onExpand}
          type="button"
        >
          <span
            className={cn(
              'min-w-0 flex-1 truncate text-xs font-medium',
              struck ? 'text-(--ui-text-quaternary) line-through' : 'text-(--ui-text-primary)'
            )}
          >
            {tool.name}
          </span>

          <Tip label={facetCopy.long}>
            <span
              className={cn(
                'w-[4.375rem] shrink-0 truncate text-[0.65rem]',
                locked ? 'text-(--ui-text-quaternary)' : TONE_CLASS[facet.tone]
              )}
            >
              {facetCopy.label}
            </span>
          </Tip>

          <span className="flex w-[8.25rem] shrink-0 items-center gap-1 overflow-hidden text-[0.65rem] text-(--ui-text-quaternary)">
            {locked ? copy.lockedHint : <HintTags hints={hints} />}
          </span>

          <DisclosureCaret className="text-(--ui-text-quaternary)" open={expanded} />
          <span className="sr-only">{expanded ? copy.hideDetails(tool.name) : copy.showDetails(tool.name)}</span>
        </button>
      </div>

      {expanded ? <ToolDetail tool={tool} /> : null}
    </div>
  )
}

function HintTags({ hints }: { hints: ReturnType<typeof hintTags> }) {
  const { t } = useI18n()

  return (
    <>
      {hints.map((hint, index) => (
        <span className="truncate" key={hint.raw}>
          {index > 0 ? '· ' : ''}
          {tagCopy(hint, t.connectorsPage.vocabulary).label}
        </span>
      ))}
    </>
  )
}

/** The open row. Indented to the name lane so the disclosure reads as belonging
 *  to the line above it, not as a new row. */
function ToolDetail({ tool }: { tool: ToolRowModel }) {
  return (
    <div className="grid gap-1 pb-2.5 pl-[3.625rem] pr-3.5">
      <code className="truncate font-mono text-[0.65rem] text-(--ui-text-quaternary)">{tool.slug}</code>
      <p className="max-w-[60ch] text-[0.7rem] leading-relaxed text-(--ui-text-secondary)">{tool.description}</p>
    </div>
  )
}
