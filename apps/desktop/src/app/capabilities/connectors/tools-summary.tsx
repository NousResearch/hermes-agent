import { useCallback, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'

import { tagCopy, vocabularyTag } from './hint-vocabulary'
import type { FacetSummaryRow } from './types'

const OPENED = new Set<string>()

export function openToolsList(listKey: string): void {
  OPENED.add(listKey)
}

export function resetOpenedTools(): void {
  OPENED.clear()
}

export function useShowAllTools(key: string, forced: boolean) {
  const [choice, setChoice] = useState<null | { key: string; open: boolean }>(null)

  const show = useCallback(() => {
    OPENED.add(key)
    setChoice({ key, open: true })
  }, [key])

  const hide = useCallback(() => {
    OPENED.delete(key)
    setChoice({ key, open: false })
  }, [key])

  const chosen = choice?.key === key ? choice.open : null

  return { hide, open: forced || (chosen ?? OPENED.has(key)), show }
}

function onLabel(copy: Translations['connectorsPage']['tools'], row: FacetSummaryRow): string {
  if (row.switchState === 'on') {
    return copy.summaryAllOn
  }

  return row.switchState === 'off' ? copy.summaryOff : copy.summarySomeOn(row.on, row.total)
}

export interface ToolsSummaryProps {
  connectorName: string
  onShowAll: () => void
  onToggleFacet: (facet: string, on: boolean) => void
  preview: boolean
  readOnly: boolean
  rows: FacetSummaryRow[]
  title?: string
  total: number
}

export function ToolsSummary({
  connectorName,
  onShowAll,
  onToggleFacet,
  preview,
  readOnly,
  rows,
  title,
  total
}: ToolsSummaryProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools

  return (
    <div className="grid min-h-0 flex-1 content-start gap-3 overflow-y-auto px-3.5 py-3" data-slot="tools-summary">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="min-w-0 truncate text-xs font-medium text-(--ui-text-primary)">
          {title ?? (preview ? copy.summaryPreviewTitle(connectorName) : copy.summaryTitle(connectorName))}
        </h3>
        <span className="shrink-0 tabular-nums text-[0.7rem] text-(--ui-text-tertiary)">
          {copy.summaryCount(total)}
        </span>
      </div>

      <ul className="grid">
        {rows.map(row => {
          const unclassed = row.facet === 'unclassified'
          const alone = unclassed && rows.length === 1

          const label = unclassed
            ? copy[alone ? 'summaryAllTools' : 'summaryOther']
            : tagCopy(vocabularyTag(row.facet), t.connectorsPage.vocabulary).label

          return (
            <li
              className="flex items-center gap-3 border-t border-(--ui-stroke-tertiary) py-2 first:border-t-0"
              key={row.facet}
            >
              <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium text-(--ui-text-primary)">
                {label}
              </span>
              <span className="shrink-0 tabular-nums text-xs text-(--ui-text-secondary)">{row.total}</span>

              {preview ? null : (
                <>
                  <span className="w-24 shrink-0 text-end text-[0.7rem] text-(--ui-text-tertiary)">
                    {onLabel(copy, row)}
                  </span>
                  <Switch
                    aria-label={alone ? copy.allToolsSwitch : copy.facetSwitch(label)}
                    checked={row.switchState !== 'off'}
                    disabled={readOnly || row.locked}
                    onCheckedChange={next => onToggleFacet(row.facet, next)}
                    size="xs"
                  />
                </>
              )}
            </li>
          )
        })}
      </ul>

      <div className="flex items-center gap-3">
        <Button onClick={onShowAll} size="inline" variant="textStrong">
          {copy.showAllTools(total)}
        </Button>
      </div>
    </div>
  )
}
