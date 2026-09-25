import type { ReactNode } from 'react'

import { SidebarDateDivider, SidebarSectionMeta } from '@/app/chat/sidebar/chrome'
import { OverlayNavItem, OverlaySidebar } from '@/app/overlays/overlay-split-layout'
import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Button } from '@/components/ui/button'
import { CheckboxMark } from '@/components/ui/checkbox'
import { useI18n } from '@/i18n'

import { catalogLabel } from './catalog-data'
import type { CatalogFacets } from './catalog-query'

export interface FacetRow {
  value: string
  label: string
  count?: number
}

interface CatalogFiltersProps {
  sources: string[]
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

function FacetItem({ row, active, onClick }: { row: FacetRow; active: boolean; onClick: () => void }) {
  return (
    <OverlayNavItem
      active={active}
      current={false}
      label={row.label}
      leading={<CheckboxMark checked={active} />}
      nested
      onClick={onClick}
      pressed={active}
      trailing={
        row.count !== undefined ? <SidebarSectionMeta>{row.count.toLocaleString()}</SidebarSectionMeta> : undefined
      }
    />
  )
}

/** Multi-select group: rows toggle, the optional "All" row clears. */
function Facet({
  label,
  all,
  beforeRows,
  rows,
  active,
  onSelect
}: {
  label: string
  all?: string
  beforeRows?: ReactNode
  rows: FacetRow[]
  active: string[]
  onSelect: (value: string | null) => void
}) {
  if (!all && !rows.length) {
    return null
  }

  return (
    <>
      <SidebarDateDivider label={label} />
      {all && <FacetItem active={!active.length} onClick={() => onSelect(null)} row={{ value: 'all', label: all }} />}
      {beforeRows}
      {rows.map(row => (
        <FacetItem active={active.includes(row.value)} key={row.value} onClick={() => onSelect(row.value)} row={row} />
      ))}
    </>
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

  return (
    <OverlaySidebar className={className}>
      <div className="flex h-7 shrink-0 items-center gap-2 pb-1">
        <SidebarPanelLabel meta={resultCount.toLocaleString()}>{c.discover}</SidebarPanelLabel>
        {filtered && (
          <Button aria-label={c.clearFilters} className="ml-auto" onClick={onClear} size="inline" variant="text">
            {t.common.clear}
          </Button>
        )}
      </div>

      <div className="pb-2">{sortControl}</div>
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
        rows={sources.map(value => ({ value, label: catalogLabel(value) }))}
      />
      <Facet
        active={facets.categories}
        all={c.allCategories}
        label={c.category}
        onSelect={onCategory}
        rows={categories.map(([value, { label, count }]) => ({ value, label: catalogLabel(label), count }))}
      />
      <Facet active={facets.tags} label={c.tags} onSelect={onTag} rows={tags} />
      {sidebarActions}
    </OverlaySidebar>
  )
}
