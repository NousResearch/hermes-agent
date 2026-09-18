// The one row of chrome above the tool list. Every control in it is derived from
// the list: a facet chip exists because that facet is present twice, the category
// picker exists because the connector has categories, the deprecated toggle
// exists because something is deprecated. Nothing is switched on per connector,
// and a connector of eight tools or fewer gets none of it.

import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { Separator } from '@/components/ui/separator'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { cn } from '@/lib/utils'

import { CategoryPicker } from './category-picker'
import type { CountedValue } from './derive-tools'
import { facetTag, hintTag, tagCopy } from './hint-vocabulary'
import type { QuickAction, QuickActionId, ToolsFilter, ToolsFreshness } from './types'

const HOUR_MS = 3_600_000

/** The quiet freshness cue. A 24 h cache means the list is usually already there,
 *  so the only honest thing to say is how old it is. */
export function freshnessLabel(copy: Translations['connectorsPage']['tools'], freshness: ToolsFreshness): string {
  const hours = Math.floor(Math.max(0, Date.now() - freshness.fetchedAt) / HOUR_MS)

  if (hours < 1) {
    return copy.freshnessJustNow
  }

  return hours < 24 ? copy.freshnessHours(hours) : copy.freshnessDays(Math.floor(hours / 24))
}

function quickActionLabel(copy: Translations['connectorsPage']['tools'], id: QuickActionId): string {
  return {
    'everything-on': copy.quickEverythingOn,
    'no-destructive': copy.quickNoDestructive,
    'read-only': copy.quickReadOnly
  }[id]
}

export interface ToolsFilterBarProps {
  categories: CountedValue[]
  currentAction: null | QuickAction
  deprecated: number
  facets: CountedValue[]
  filter: ToolsFilter
  freshness?: ToolsFreshness
  hints: CountedValue[]
  onApplyQuickAction: (id: QuickActionId) => void
  onFilterChange: (next: ToolsFilter) => void
  onRefresh: () => void
  quickActions: QuickAction[]
  /** Under nine tools the whole bar is one search field's worth of noise. */
  tiny: boolean
  total: number
}

export function ToolsFilterBar({
  categories,
  currentAction,
  deprecated,
  facets,
  filter,
  freshness,
  hints,
  onApplyQuickAction,
  onFilterChange,
  onRefresh,
  quickActions,
  tiny,
  total
}: ToolsFilterBarProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools
  const set = (patch: Partial<ToolsFilter>) => onFilterChange({ ...filter, ...patch })

  return (
    <div className="grid gap-2 border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome) px-3.5 py-2">
      <div className="flex items-center gap-3">
        <span className="shrink-0 text-xs font-medium text-(--ui-text-primary)">{copy.title}</span>
        <span className="shrink-0 tabular-nums text-[0.7rem] text-(--ui-text-tertiary)">{total}</span>

        {tiny ? null : (
          <SearchField
            containerClassName="min-w-0 flex-1"
            onChange={query => set({ query })}
            placeholder={copy.searchCountPlaceholder(total)}
            value={filter.query}
          />
        )}

        {freshness ? (
          <span className="ml-auto shrink-0 text-[0.65rem] text-(--ui-text-quaternary)">
            {freshnessLabel(copy, freshness)}
          </span>
        ) : null}

        <Button className="shrink-0" onClick={onRefresh} size="xs" variant="text">
          {copy.refresh}
        </Button>
      </div>

      {tiny ? null : (
        <div className="flex flex-wrap items-center gap-1.5">
          {facets.map(entry => (
            <FilterChip
              count={entry.count}
              key={entry.value}
              label={tagCopy(facetTag(entry.value), t.connectorsPage.vocabulary).label}
              onClick={() => set({ facet: filter.facet === entry.value ? null : entry.value })}
              selected={filter.facet === entry.value}
            />
          ))}

          {facets.length > 0 && hints.length > 0 ? <Separator className="mx-1 h-4" orientation="vertical" /> : null}

          {hints.map(entry => (
            <FilterChip
              key={entry.value}
              label={tagCopy(hintTag(entry.value), t.connectorsPage.vocabulary).label}
              onClick={() => set({ hint: filter.hint === entry.value ? null : entry.value })}
              selected={filter.hint === entry.value}
            />
          ))}

          {categories.length > 0 ? (
            <CategoryPicker categories={categories} onChange={category => set({ category })} value={filter.category} />
          ) : null}

          {deprecated > 0 ? (
            <Button
              aria-pressed={filter.showDeprecated}
              onClick={() => set({ showDeprecated: !filter.showDeprecated })}
              size="xs"
              variant="text"
            >
              {filter.showDeprecated ? copy.hideDeprecated(deprecated) : copy.showDeprecated(deprecated)}
            </Button>
          ) : null}

          {quickActions.length > 0 ? (
            <div className="ml-auto flex items-center gap-1.5">
              {quickActions.map(action => (
                <Button
                  aria-pressed={currentAction?.id === action.id}
                  key={action.id}
                  onClick={() => onApplyQuickAction(action.id)}
                  size="xs"
                  variant={currentAction?.id === action.id ? 'secondary' : 'outline'}
                >
                  {quickActionLabel(copy, action.id)}
                </Button>
              ))}
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}

/** A chip is a toggle, so it is a button — the app has one button primitive and
 *  a badge is not interactive. The count rides inside the label. */
function FilterChip({
  count,
  label,
  onClick,
  selected
}: {
  count?: number
  label: string
  onClick: () => void
  selected: boolean
}) {
  return (
    <Button aria-pressed={selected} onClick={onClick} size="xs" variant={selected ? 'secondary' : 'ghost'}>
      {label}
      {count === undefined ? null : (
        <span className={cn('tabular-nums', selected ? 'opacity-70' : 'text-(--ui-text-quaternary)')}>{count}</span>
      )}
    </Button>
  )
}
