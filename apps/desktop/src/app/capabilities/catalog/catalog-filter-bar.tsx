import { IconCategory, IconDatabase, IconFilter2 } from '@tabler/icons-react'
import { type ReactNode, useId, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'

import { catalogLabel } from './catalog-data'
import type { FacetRow } from './catalog-filters'
import type { CatalogFacets } from './catalog-query'

interface CatalogFilterBarProps {
  sources: string[]
  categories: ReadonlyArray<readonly [string, { label: string; count: number }]>
  tags: FacetRow[]
  facets: CatalogFacets
  onSource: (value: string | null) => void
  onCategory: (value: string | null) => void
  onTag: (value: string | null) => void
  onInstalled: () => void
  onClear: () => void
  sortControl: ReactNode
  actions: ReactNode
  resultCount: number
}

export function CatalogFilterBar(props: CatalogFilterBarProps) {
  const { t } = useI18n()
  const c = t.catalog
  const panelId = useId()
  const [mode, setMode] = useState<'categories' | 'sources'>('categories')
  const [expanded, setExpanded] = useState(false)
  const [more, setMore] = useState(false)

  const rows = mode === 'categories'
    ? props.categories.map(([value, meta]) => ({ value, label: catalogLabel(meta.label) }))
    : props.sources.map(value => ({ value, label: catalogLabel(value) }))

  const selected = props.facets[mode]
  const onSelect = mode === 'categories' ? props.onCategory : props.onSource
  const visible = rows.filter((row, index) => index < 3 || selected.includes(row.value))
  const overflow = rows.filter(row => !visible.includes(row))
  const activeCount = props.facets.categories.length + props.facets.sources.length + props.facets.tags.length

  return (
    <div className="catalog-filter-bar" data-catalog-filters>
      <div className="catalog-filter-row">
        <SegmentedControl onChange={setMode} options={[{ id: 'categories', label: c.category, icon: IconCategory }, { id: 'sources', label: c.source, icon: IconDatabase }]} sizing="content" value={mode} />
        <div className="catalog-filter-chips">
          <Button aria-pressed={!selected.length} onClick={() => onSelect(null)} size="xs" variant={!selected.length ? 'default' : 'outline'}>{t.skills.all}</Button>
          <span aria-hidden className="px-1 text-(--ui-text-quaternary)">/</span>
          {visible.map(row => <Button aria-pressed={selected.includes(row.value)} key={row.value} onClick={() => onSelect(row.value)} size="xs" variant={selected.includes(row.value) ? 'default' : 'outline'}>{row.label}</Button>)}
          {overflow.length > 0 && <Popover onOpenChange={setMore} open={more}>
            <PopoverTrigger asChild><Button aria-label={c.more} size="xs" variant="secondary">{`+${overflow.length}`}</Button></PopoverTrigger>
            <PopoverContent align="start" className="w-72 max-h-80 overflow-y-auto p-2" variant="menu">
              <div className="flex flex-wrap gap-1.5">{overflow.map(row => <Button aria-pressed={selected.includes(row.value)} key={row.value} onClick={() => onSelect(row.value)} size="xs" variant={selected.includes(row.value) ? 'default' : 'outline'}>{row.label}</Button>)}</div>
            </PopoverContent>
          </Popover>}
        </div>
        <div className="catalog-filter-trigger">
          <span className="text-[0.625rem] font-medium uppercase tracking-wider text-(--ui-text-tertiary)" id={`${panelId}-label`}>{c.filters}{activeCount > 0 && ` · ${activeCount}`}</span>
          <Popover onOpenChange={setExpanded} open={expanded}>
            <PopoverTrigger asChild>
              <Button aria-labelledby={`${panelId}-label`} size="icon-xs" variant={expanded ? 'default' : 'secondary'}>
                <IconFilter2 />
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" aria-label={c.filters} className="w-80 max-w-[calc(100vw-2rem)] p-4" side="bottom" variant="menu">
              <div className="flex max-h-[min(60vh,28rem)] flex-col gap-4 overflow-y-auto" data-catalog-expanded-filters>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-medium">{c.filters}</span>
                  <Button onClick={props.onClear} size="inline" variant="text">{c.clearFilters}</Button>
                </div>
                <div className="flex flex-col gap-2">
                  <span className="text-xs text-(--ui-text-tertiary)">{c.sortBy}</span>
                  {props.sortControl}
                </div>
                <div className="flex items-center justify-between gap-3">
                  <Button aria-pressed={props.facets.installedOnly} onClick={props.onInstalled} size="xs" variant={props.facets.installedOnly ? 'default' : 'outline'}>{c.installed}</Button>
                  <span className="text-xs text-(--ui-text-tertiary)">{c.results(props.resultCount)}</span>
                </div>
                {props.tags.length > 0 && <div className="flex flex-col gap-2">
                  <span className="text-xs text-(--ui-text-tertiary)">{c.tags}</span>
                  <div className="flex flex-wrap gap-1.5">
                    {props.tags.map(row => <Button aria-pressed={props.facets.tags.includes(row.value)} key={row.value} onClick={() => props.onTag(row.value)} size="xs" variant={props.facets.tags.includes(row.value) ? 'default' : 'outline'}>{row.label}</Button>)}
                  </div>
                </div>}
                {props.actions && <div className="flex flex-wrap items-center gap-2">{props.actions}</div>}
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>
    </div>
  )
}
