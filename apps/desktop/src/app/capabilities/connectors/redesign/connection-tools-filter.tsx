import './connections.css'

import { type RefObject, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Kbd } from '@/components/ui/kbd'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'

import { categoryLabel, UNCATEGORISED } from '../derive-tools'
import type { ToolRowModel } from '../types'

import { type ConnectionsCopy, useConnectionsCopy } from './connections-copy'
import { onFilterGroupKeyDown } from './connections-filter'
import {
  activeToolsChipCount,
  CONNECTION_TOOL_FACETS,
  type ConnectionToolsFilterState,
  filterConnectionTools,
  groupToolsByArea,
  toggleToolAreaFilter,
  toggleToolFacetFilter
} from './filter-logic'

/**
 * Search, the Tool Filter popover (Figma `1565:31077`) and the active-chip row
 * (`1565:30694`) for the per-app Tools page. All state lives with the page: this
 * module only reads a `ConnectionToolsFilterState` and patches it.
 */

type FilterPatch = Partial<ConnectionToolsFilterState>

interface AreaOption {
  area: string
  count: number
  label: string
}

interface FacetCounts {
  all: number
  destructive: number
  read: number
  write: number
}

interface StatusCounts {
  off: number
  on: number
}

export interface ConnectionToolsToolbarProps {
  appName: string
  filter: ConnectionToolsFilterState
  onFilterChange: (patch: FilterPatch) => void
  tools: readonly ToolRowModel[]
}

export function ConnectionToolsToolbar({
  appName,
  filter,
  onFilterChange,
  tools
}: ConnectionToolsToolbarProps) {
  const copy = useConnectionsCopy()
  const field = useRef<HTMLInputElement>(null)

  useSlashShortcut(field)

  const filteredCount = useMemo(() => filterConnectionTools(tools, filter).length, [filter, tools])
  const chipCount = activeToolsChipCount(filter)
  const areas = useAreaOptions(copy, tools)
  const clearChips = () => onFilterChange({ areas: [], facets: [], status: 'any' })

  return (
    <div className="conn-tools-toolbar" data-slot="connection-tools-toolbar">
      <div className="conn-tools-search">
        <Codicon aria-hidden name="search" size="1rem" />
        <input
          aria-label={copy.tools.searchAppTools(appName)}
          autoComplete="off"
          className="conn-tools-search-input"
          onChange={event => onFilterChange({ query: event.target.value })}
          onKeyDown={event => {
            if (event.key === 'Escape' && filter.query.length > 0) {
              event.preventDefault()
              event.stopPropagation()
              onFilterChange({ query: '' })
            }
          }}
          placeholder={copy.tools.searchAppTools(appName)}
          ref={field}
          spellCheck={false}
          type="text"
          value={filter.query}
        />
        {filter.query.length > 0 ? (
          <button
            aria-label={copy.search.clear}
            className="conn-search-clear"
            onClick={() => {
              onFilterChange({ query: '' })
              field.current?.focus()
            }}
            type="button"
          >
            <Codicon aria-hidden name="close" size="0.875rem" />
          </button>
        ) : (
          <Kbd aria-hidden size="sm">
            /
          </Kbd>
        )}
      </div>

      <ToolFilterPopover
        copy={copy}
        filter={filter}
        filteredCount={filteredCount}
        onFilterChange={onFilterChange}
        tools={tools}
      />

      {chipCount > 0 ? (
        <ActiveToolsChips
          areas={areas}
          copy={copy}
          filter={filter}
          onClear={clearChips}
          onFilterChange={onFilterChange}
        />
      ) : null}
    </div>
  )
}

/** `/` reaches the tools search from anywhere on the page, as the Figma shortcut promises. */
function useSlashShortcut(field: RefObject<HTMLInputElement | null>): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== '/' || event.defaultPrevented || event.repeat) {
        return
      }

      if (event.metaKey || event.ctrlKey || event.altKey) {
        return
      }

      const target = event.target instanceof HTMLElement ? event.target : null

      if (target?.isContentEditable) {
        return
      }

      if (target?.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])')) {
        return
      }

      if (
        document.querySelector(
          '[role="menu"], [data-radix-popper-content-wrapper], [aria-modal="true"]'
        ) !== null
      ) {
        return
      }

      event.preventDefault()
      field.current?.focus()
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [field])
}

function useAreaOptions(copy: ConnectionsCopy, tools: readonly ToolRowModel[]): AreaOption[] {
  return useMemo(
    () =>
      groupToolsByArea(tools).map(group => ({
        area: group.area,
        count: group.tools.length,
        label: areaName(group.area, copy)
      })),
    [copy, tools]
  )
}

function areaName(area: string, copy: ConnectionsCopy): string {
  return area === UNCATEGORISED ? copy.toolFilters.uncategorised : categoryLabel(area)
}

function areaSummary(filter: ConnectionToolsFilterState, areas: readonly AreaOption[], copy: ConnectionsCopy): string {
  if (filter.areas.length === 0) {
    return copy.tools.allAreas
  }

  return filter.areas
    .map(area => areas.find(option => option.area === area)?.label ?? areaName(area, copy))
    .join(', ')
}

function countFacets(tools: readonly ToolRowModel[]): FacetCounts {
  const counts: FacetCounts = { all: tools.length, destructive: 0, read: 0, write: 0 }

  for (const tool of tools) {
    if (tool.facet === 'read' || tool.facet === 'write' || tool.facet === 'destructive') {
      counts[tool.facet] += 1
    }
  }

  return counts
}

function countStatus(tools: readonly ToolRowModel[]): StatusCounts {
  const on = tools.filter(tool => tool.on).length

  return { off: tools.length - on, on }
}

interface ToolFilterPopoverProps {
  copy: ConnectionsCopy
  filter: ConnectionToolsFilterState
  filteredCount: number
  onFilterChange: (patch: FilterPatch) => void
  tools: readonly ToolRowModel[]
}

function ToolFilterPopover({ copy, filter, filteredCount, onFilterChange, tools }: ToolFilterPopoverProps) {
  const [open, setOpen] = useState(false)
  const [areaOpen, setAreaOpen] = useState(false)
  const [areaQuery, setAreaQuery] = useState('')

  const facetCounts = useMemo(() => countFacets(tools), [tools])
  const statusCounts = useMemo(() => countStatus(tools), [tools])
  const areas = useAreaOptions(copy, tools)
  const chipCount = activeToolsChipCount(filter)
  const clearAll = () => onFilterChange({ areas: [], facets: [], status: 'any' })

  const onOpenChange = (next: boolean) => {
    setOpen(next)

    if (!next) {
      setAreaOpen(false)
      setAreaQuery('')
    }
  }

  return (
    <Popover onOpenChange={onOpenChange} open={open}>
      <PopoverTrigger asChild>
        <Button
          className={chipCount > 0 ? 'conn-tools-filter-button is-active' : 'conn-tools-filter-button'}
          size="sm"
          variant="outline"
        >
          <Codicon aria-hidden name="filter" size="0.875rem" />
          {chipCount > 0 ? `${copy.tools.filter} (${chipCount})` : copy.tools.filter}
        </Button>
      </PopoverTrigger>

      <PopoverContent align="end" className="conn-filter-popover" variant="menu">
        <div className="conn-filter-form">
          <div className="conn-filter-head">
            <span className="conn-filter-title">{copy.tools.filter}</span>
            <Button className="conn-filter-clear" onClick={clearAll} size="xs" variant="text">
              {copy.toolFilters.clear}
            </Button>
          </div>

          <section className="conn-filter-section">
            <span className="conn-filter-label">{copy.tools.whatItDoes}</span>
            <p className="conn-filter-hint">{copy.tools.whatItDoesHint}</p>
            <div aria-label={copy.tools.whatItDoes} className="conn-filter-grid" role="group">
              <FacetOption
                checked={filter.facets.length === 0}
                count={facetCounts.all}
                label={copy.toolFilters.all}
                onToggle={() => onFilterChange({ facets: [] })}
              />
              {CONNECTION_TOOL_FACETS.map(facet => (
                <FacetOption
                  checked={filter.facets.includes(facet)}
                  count={facetCounts[facet]}
                  key={facet}
                  label={copy.badges[facet]}
                  onToggle={() =>
                    onFilterChange({
                      facets: toggleToolFacetFilter(filter.facets, facet, CONNECTION_TOOL_FACETS)
                    })
                  }
                />
              ))}
            </div>
          </section>

          <section className="conn-filter-section">
            <span className="conn-filter-label">{copy.toolFilters.status}</span>
            <p className="conn-filter-hint">{copy.filters.chooseOne}</p>
            <div
              aria-label={copy.toolFilters.status}
              className="conn-filter-status"
              onKeyDown={onFilterGroupKeyDown}
              role="radiogroup"
            >
              <StatusPill
                checked={filter.status === 'any'}
                label={copy.filters.statusAny}
                onSelect={() => onFilterChange({ status: 'any' })}
              />
              <StatusPill
                checked={filter.status === 'on'}
                count={statusCounts.on}
                label={copy.tools.on}
                onSelect={() => onFilterChange({ status: 'on' })}
              />
              <StatusPill
                checked={filter.status === 'off'}
                count={statusCounts.off}
                label={copy.tools.off}
                onSelect={() => onFilterChange({ status: 'off' })}
              />
            </div>
          </section>

          <section className="conn-filter-section">
            <span className="conn-filter-label">{copy.toolFilters.areas}</span>
            <button
              aria-expanded={areaOpen}
              className={areaOpen ? 'conn-filter-area-toggle is-open' : 'conn-filter-area-toggle'}
              onClick={() => {
                setAreaOpen(previous => !previous)
                setAreaQuery('')
              }}
              type="button"
            >
              <span className="conn-filter-area-summary">{areaSummary(filter, areas, copy)}</span>
              {filter.areas.length > 0 ? (
                <span className="conn-filter-area-meta">{copy.filters.selectedCount(filter.areas.length)}</span>
              ) : null}
              <Codicon
                aria-hidden
                className={areaOpen ? 'conn-filter-chevron is-open' : 'conn-filter-chevron'}
                name="chevron-down"
                size="0.875rem"
              />
            </button>

            {areaOpen ? (
              <AreaPicker
                areas={areas}
                copy={copy}
                filter={filter}
                onFilterChange={onFilterChange}
                onQueryChange={setAreaQuery}
                query={areaQuery}
              />
            ) : null}
          </section>

          <Button className="conn-filter-apply" onClick={() => onOpenChange(false)} size="sm" variant="default">
            {copy.tools.showTools(filteredCount)}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}

function FacetOption({
  checked,
  count,
  label,
  onToggle
}: {
  checked: boolean
  count: number
  label: string
  onToggle: () => void
}) {
  return (
    <button
      aria-checked={checked}
      className="conn-filter-option conn-check"
      onClick={onToggle}
      role="checkbox"
      type="button"
    >
      <span aria-hidden className="conn-check-mark">
        {checked ? <Codicon name="check" size="0.75rem" /> : null}
      </span>
      <span className="flex-1">{label}</span>
      <span className="conn-filter-count">{count}</span>
    </button>
  )
}

function StatusPill({
  checked,
  count,
  label,
  onSelect
}: {
  checked: boolean
  count?: number
  label: string
  onSelect: () => void
}) {
  return (
    <button
      aria-checked={checked}
      className="conn-filter-pill"
      data-selected={checked ? '' : undefined}
      onClick={onSelect}
      role="radio"
      type="button"
    >
      {label}
      {count === undefined ? null : <span className="conn-filter-count">{count}</span>}
    </button>
  )
}

interface AreaPickerProps {
  areas: readonly AreaOption[]
  copy: ConnectionsCopy
  filter: ConnectionToolsFilterState
  onFilterChange: (patch: FilterPatch) => void
  onQueryChange: (query: string) => void
  query: string
}

function AreaPicker({ areas, copy, filter, onFilterChange, onQueryChange, query }: AreaPickerProps) {
  const available = areas.map(option => option.area)
  const needle = query.trim().toLowerCase()
  const visible = needle === '' ? areas : areas.filter(option => option.label.toLowerCase().includes(needle))

  return (
    <div className="conn-filter-area">
      <div className="conn-filter-area-search">
        <Codicon aria-hidden name="search" size="0.875rem" />
        <input
          aria-label={copy.tools.findArea}
          autoFocus
          onChange={event => onQueryChange(event.target.value)}
          placeholder={copy.tools.findArea}
          spellCheck={false}
          type="text"
          value={query}
        />
      </div>

      <div
        aria-label={copy.toolFilters.areas}
        aria-multiselectable
        className="conn-filter-area-list"
        onKeyDown={onFilterGroupKeyDown}
        role="listbox"
      >
        <button
          aria-selected={filter.areas.length === 0}
          className="conn-filter-area-option"
          onClick={() => onFilterChange({ areas: [] })}
          role="option"
          type="button"
        >
          <span className="conn-filter-area-name">{copy.tools.allAreas}</span>
          {filter.areas.length === 0 ? <Codicon aria-hidden name="check" size="0.875rem" /> : null}
        </button>

        {visible.map(option => (
          <AreaRow
            available={available}
            filter={filter}
            key={option.area}
            onFilterChange={onFilterChange}
            option={option}
          />
        ))}

      </div>

      {visible.length === 0 ? (
        <p className="conn-filter-area-empty">{copy.filters.noResultsQuery(query)}</p>
      ) : null}
    </div>
  )
}

function AreaRow({
  available,
  filter,
  onFilterChange,
  option
}: {
  available: readonly string[]
  filter: ConnectionToolsFilterState
  onFilterChange: (patch: FilterPatch) => void
  option: AreaOption
}) {
  const on = filter.areas.includes(option.area)

  return (
    <button
      aria-selected={on}
      className={on ? 'conn-filter-area-option is-selected' : 'conn-filter-area-option'}
      onClick={() => onFilterChange({ areas: toggleToolAreaFilter(filter.areas, option.area, available) })}
      role="option"
      type="button"
    >
      <span className="conn-filter-area-name">{option.label}</span>
      <span className="conn-filter-area-count">{option.count}</span>
      {on ? <Codicon aria-hidden name="check" size="0.875rem" /> : null}
    </button>
  )
}

interface ActiveToolsChipsProps {
  areas: readonly AreaOption[]
  copy: ConnectionsCopy
  filter: ConnectionToolsFilterState
  onClear: () => void
  onFilterChange: (patch: FilterPatch) => void
}

function ActiveToolsChips({ areas, copy, filter, onClear, onFilterChange }: ActiveToolsChipsProps) {
  const showDivider = filter.facets.length > 0 && filter.areas.length > 0
  const statusLabel = filter.status === 'on' ? copy.tools.on : copy.tools.off

  return (
    <div className="conn-tools-chips basis-full" data-slot="connection-tools-chips">
      {filter.facets.length === 0 ? null : (
        <span className="conn-chips-label">{copy.tools.permissionChipLabel}</span>
      )}
      {filter.facets.map(facet => (
        <FilterChip
          copy={copy}
          key={facet}
          label={copy.badges[facet]}
          onRemove={() =>
            onFilterChange({ facets: toggleToolFacetFilter(filter.facets, facet, CONNECTION_TOOL_FACETS) })
          }
        />
      ))}

      {showDivider ? (
        <span aria-hidden className="conn-chips-divider">
          /
        </span>
      ) : null}

      {filter.areas.length === 0 ? null : (
        <span className="conn-chips-label">{copy.tools.areaChipLabel}</span>
      )}
      {filter.areas.map(area => (
        <FilterChip
          copy={copy}
          key={area}
          label={areas.find(option => option.area === area)?.label ?? areaName(area, copy)}
          onRemove={() =>
            onFilterChange({ areas: toggleToolAreaFilter(filter.areas, area, areas.map(option => option.area)) })
          }
        />
      ))}

      {filter.status === 'any' ? null : (
        <>
          {filter.facets.length > 0 || filter.areas.length > 0 ? (
            <span aria-hidden className="conn-chips-divider">
              /
            </span>
          ) : null}
          <span className="conn-chips-label">{copy.toolFilters.status}</span>
          <FilterChip copy={copy} label={statusLabel} onRemove={() => onFilterChange({ status: 'any' })} />
        </>
      )}

      <span aria-hidden className="conn-chips-divider">
        /
      </span>

      <button className="conn-chips-clear" onClick={onClear} type="button">
        {copy.toolFilters.clear}
      </button>
    </div>
  )
}

function FilterChip({ copy, label, onRemove }: { copy: ConnectionsCopy; label: string; onRemove: () => void }) {
  return (
    <button
      aria-label={`${copy.search.clear}: ${label}`}
      className="conn-chip"
      onClick={onRemove}
      type="button"
    >
      <span className="conn-chip-label">{label}</span>
      <Codicon aria-hidden className="conn-chip-remove" name="close" size="0.75rem" />
    </button>
  )
}
