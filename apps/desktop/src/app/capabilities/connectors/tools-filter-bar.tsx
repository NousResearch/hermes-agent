import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { Separator } from '@/components/ui/separator'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { cn } from '@/lib/utils'

import { CategoryPicker } from './category-picker'
import type { CountedValue } from './derive-tools'
import { tagCopy, vocabularyTag } from './hint-vocabulary'
import type { QuickAction, QuickActionId, ToolsFilter } from './types'

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
  facets: CountedValue[]
  filter: ToolsFilter
  hints: CountedValue[]
  onApplyQuickAction: (id: QuickActionId) => void
  onFilterChange: (next: ToolsFilter) => void
  onShowSummary: () => void
  quickActions: QuickAction[]
  tiny: boolean
  total: number
}

export function ToolsFilterBar({
  categories,
  currentAction,
  facets,
  filter,
  hints,
  onApplyQuickAction,
  onFilterChange,
  onShowSummary,
  quickActions,
  tiny,
  total
}: ToolsFilterBarProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools
  const set = (patch: Partial<ToolsFilter>) => onFilterChange({ ...filter, ...patch })
  const chips = facets.length + hints.length + categories.length + quickActions.length

  return (
    <div className="grid shrink-0 gap-2 border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome) px-3.5 py-2">
      <div className="flex items-center gap-3">
        <span className="shrink-0 text-xs font-medium text-(--ui-text-primary)">{copy.title}</span>
        <span className="shrink-0 tabular-nums text-[0.7rem] text-(--ui-text-tertiary)">{total}</span>

        <Button className="shrink-0" onClick={onShowSummary} size="xs" variant="text">
          {copy.showSummary}
        </Button>

        {tiny ? null : (
          <SearchField
            containerClassName="min-w-0 flex-1"
            onChange={query => set({ query })}
            placeholder={copy.searchCountPlaceholder(total)}
            value={filter.query}
          />
        )}
      </div>

      {tiny || chips === 0 ? null : (
        <div className="flex flex-wrap items-center gap-1.5">
          {facets.map(entry => (
            <FilterChip
              count={entry.count}
              key={entry.value}
              label={tagCopy(vocabularyTag(entry.value), t.connectorsPage.vocabulary).label}
              onClick={() => set({ facet: filter.facet === entry.value ? null : entry.value })}
              selected={filter.facet === entry.value}
            />
          ))}

          {facets.length > 0 && hints.length > 0 ? <Separator className="mx-1 data-[orientation=vertical]:h-4" orientation="vertical" /> : null}

          {hints.map(entry => (
            <FilterChip
              key={entry.value}
              label={tagCopy(vocabularyTag(entry.value), t.connectorsPage.vocabulary).label}
              onClick={() => set({ hint: filter.hint === entry.value ? null : entry.value })}
              selected={filter.hint === entry.value}
            />
          ))}

          {categories.length > 0 ? (
            <CategoryPicker categories={categories} onChange={category => set({ category })} value={filter.category} />
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
