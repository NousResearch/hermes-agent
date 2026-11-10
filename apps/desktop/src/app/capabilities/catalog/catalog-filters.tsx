import { compactNumber } from '@hermes/shared'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { catalogLabel } from './catalog-data'
import type { CatalogFacets } from './catalog-query'

export interface FacetRow {
  value: string
  label: string
  count?: number
}

interface CatalogFiltersProps {
  sources: ReadonlyArray<{ value: string; count: number }>
  categories: ReadonlyArray<readonly [string, { label: string; count: number }]>
  tags: FacetRow[]
  facets: CatalogFacets
  /** `null` clears the facet. */
  onSource: (value: string | null) => void
  onCategory: (value: string | null) => void
  onTag: (value: string | null) => void
  onInstalled: () => void
  onClear: () => void
  resultCount: number
  sidebarActions?: ReactNode
  sortControl?: ReactNode
  className?: string
}

/** Figma SkillsFilterRow — node 293:3000 / 293:3008 */
function FacetItem({ row, active, onClick }: { row: FacetRow; active: boolean; onClick: () => void }) {
  return (
    <button
      aria-pressed={active}
      className={cn(
        'flex w-full shrink-0 items-center justify-between rounded-[10px] p-2 text-left transition-colors',
        active ? 'bg-[#e3ecfd]' : 'bg-transparent hover:bg-[#f1f5f9]'
      )}
      onClick={onClick}
      type="button"
    >
      <span className="flex min-w-0 items-center gap-2.5 overflow-hidden">
        <span
          aria-hidden
          className={cn(
            'grid size-4 shrink-0 place-items-center rounded-[4px] border-solid',
            active ? 'border border-[#2f5bea] bg-[#2f5bea]' : 'border-[1.5px] border-[#cbd5e1] bg-white'
          )}
        >
          {active ? (
            <svg className="size-2.5 text-white" fill="none" viewBox="0 0 10 10" aria-hidden>
              <path d="M1.5 5.2 3.8 7.5 8.5 2.5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.6" />
            </svg>
          ) : null}
        </span>
        <span
          className={cn(
            'min-w-0 truncate text-[13px] font-medium leading-none',
            active ? 'text-[#0f172a]' : 'text-[#334155]'
          )}
        >
          {row.label}
        </span>
      </span>
      {row.count !== undefined ? (
        <span className="shrink-0 pl-2 text-[12px] font-semibold tabular-nums leading-none text-[#64748b]">
          {compactNumber(row.count)}
        </span>
      ) : null}
    </button>
  )
}

function Facet({
  label,
  all,
  allCount,
  beforeRows,
  rows,
  active,
  onSelect
}: {
  label: string
  all?: string
  allCount?: number
  beforeRows?: ReactNode
  rows: FacetRow[]
  active: string[]
  onSelect: (value: string | null) => void
}) {
  if (!all && !rows.length) {
    return null
  }

  return (
    <section className="space-y-0.5 pt-4">
      <h3 className="px-2 pb-1 text-[11px] font-semibold uppercase tracking-[0.6px] text-[#64748b]">{label}</h3>
      <div className="space-y-0">
        {all ? (
          <FacetItem
            active={!active.length}
            onClick={() => onSelect(null)}
            row={{ value: 'all', label: all, count: allCount }}
          />
        ) : null}
        {beforeRows}
        {rows.map(row => (
          <FacetItem
            active={active.includes(row.value)}
            key={row.value}
            onClick={() => onSelect(row.value)}
            row={row}
          />
        ))}
      </div>
    </section>
  )
}

export function CatalogFilters({
  sources,
  categories,
  tags,
  facets,
  onSource,
  onCategory,
  onTag,
  onInstalled,
  onClear,
  resultCount,
  sidebarActions,
  sortControl,
  className
}: CatalogFiltersProps) {
  const { t } = useI18n()
  const c = t.catalog
  const filtered = facets.sources.length + facets.categories.length + facets.tags.length > 0 || facets.installedOnly
  const categoryTotal = categories.reduce((sum, [, { count }]) => sum + count, 0)

  return (
    <aside className={cn('flex min-h-0 w-[248px] shrink-0 flex-col overflow-y-auto', className)} data-catalog-filters>
      <header className="shrink-0 space-y-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="min-w-0 truncate text-[12px] font-bold uppercase tracking-[0.8px] text-[#0f172a]">
            {c.discover}
          </h2>
          <span className="ml-auto shrink-0 text-[12px] font-semibold tabular-nums text-[#2f5bea]">
            {compactNumber(resultCount)}
          </span>
          {filtered ? (
            <Button aria-label={c.clearFilters} className="h-auto shrink-0 p-0 text-[11px]" onClick={onClear} size="inline" variant="text">
              {t.common.clear}
            </Button>
          ) : null}
        </div>
        {sortControl ? <div data-catalog-sort>{sortControl}</div> : null}
      </header>

      <Facet
        active={facets.sources}
        all={c.allSources}
        beforeRows={
          <FacetItem
            active={facets.installedOnly}
            onClick={onInstalled}
            row={{ value: 'installed', label: c.installed }}
          />
        }
        label={c.source}
        onSelect={onSource}
        rows={sources.map(({ value, count }) => ({ value, label: catalogLabel(value), count }))}
      />
      <Facet
        active={facets.categories}
        all={c.allCategories}
        allCount={categoryTotal || undefined}
        label={c.category}
        onSelect={onCategory}
        rows={categories.map(([value, { label, count }]) => ({ value, label: catalogLabel(label), count }))}
      />
      <Facet active={facets.tags} label={c.tags} onSelect={onTag} rows={tags} />
      {sidebarActions}
    </aside>
  )
}
