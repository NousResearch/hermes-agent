import {
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState
} from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

import type { ConnectorCardModel } from '../types'

import { type ConnectionsCopy, useConnectionsCopy } from './connections-copy'
import {
  activeConnectionsChipCount,
  type ConnectionsFilterState,
  type ConnectionStatusFilter,
  countByCategory,
  countByStatus,
  filterConnectionCards,
  toggleCategoryFilter
} from './filter-logic'
import {
  categoryOrderForMode,
  type ConnectionCategoryId,
  type ConnectionInterfaceMode
} from './taxonomy'

export interface ConnectionsToolbarProps {
  cards: readonly ConnectorCardModel[]
  /** The Category See-All page already scopes to one category, so it hides the control. */
  hideCategoryFilter?: boolean
  filter: ConnectionsFilterState
  mode: ConnectionInterfaceMode
  onFilterChange: (next: ConnectionsFilterState) => void
  searchPlaceholder: string
  extraAction?: ReactNode
}

/** A field's keydown is the surface's own: `/` must not steal a keystroke from it. */
function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false
  }

  return (
    target.isContentEditable ||
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT'
  )
}

const GROUP_KEYS: Record<string, true> = {
  ArrowDown: true,
  ArrowLeft: true,
  ArrowRight: true,
  ArrowUp: true,
  End: true,
  Home: true
}

/** Arrow keys and Home/End walk a radiogroup or listbox option by option, as forms mode expects. */
export function onFilterGroupKeyDown(event: ReactKeyboardEvent<HTMLElement>): void {
  if (event.altKey || event.ctrlKey || event.metaKey || !Object.hasOwn(GROUP_KEYS, event.key)) {
    return
  }

  const options = [...event.currentTarget.querySelectorAll<HTMLElement>('[role="option"], [role="radio"]')]

  if (options.length === 0) {
    return
  }

  const current = options.findIndex(option => option === event.target || option.contains(event.target as Node))
  let next = current

  if (event.key === 'Home') {
    next = 0
  } else if (event.key === 'End') {
    next = options.length - 1
  } else {
    const forward = event.key === 'ArrowDown' || event.key === 'ArrowRight'
    const base = current === -1 ? (forward ? -1 : options.length) : current
    next = (base + (forward ? 1 : -1) + options.length) % options.length
  }

  event.preventDefault()
  options[next]?.focus()
}

function chipLabel(status: ConnectionStatusFilter, copy: ConnectionsCopy): string {
  if (status === 'connected') {
    return copy.filters.statusConnected
  }

  return status === 'not_connected' ? copy.filters.statusNotConnected : copy.filters.statusAny
}

/**
 * Search, the filter popover (Figma `1649:30098`) and the active-chip row
 * (`1649:29813`) shared by the directory, a category page and Review all tools.
 */
export function ConnectionsToolbar({
  cards,
  extraAction,
  filter,
  hideCategoryFilter = false,
  mode,
  onFilterChange,
  searchPlaceholder
}: ConnectionsToolbarProps) {
  const copy = useConnectionsCopy()
  const input = useRef<HTMLInputElement>(null)
  const titleId = useId()
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState<ConnectionsFilterState>(filter)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [pickerQuery, setPickerQuery] = useState('')

  const categories = categoryOrderForMode(mode)
  const statusCounts = useMemo(() => countByStatus(cards), [cards])
  const categoryCounts = useMemo(() => countByCategory(cards), [cards])
  const activeCount = activeConnectionsChipCount(filter)
  const draftFilter = useMemo(
    () => ({ ...draft, query: filter.query }),
    [draft, filter.query]
  )
  const draftResultCount = useMemo(
    () => filterConnectionCards(cards, draftFilter).length,
    [cards, draftFilter]
  )

  // `/` jumps to search from anywhere on the page; Escape empties the field
  // without blurring it, which is what a search field is for.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey || event.repeat) {
        return
      }

      if (event.key === 'Escape' && document.activeElement === input.current) {
        if (filter.query.length > 0) {
          event.preventDefault()
          onFilterChange({ ...filter, query: '' })
        }

        return
      }

      if (
        event.key !== '/' ||
        open ||
        isTypingTarget(event.target) ||
        document.querySelector('[aria-modal="true"]') !== null
      ) {
        return
      }

      event.preventDefault()
      input.current?.focus()
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [filter, onFilterChange, open])

  const changeOpen = (next: boolean) => {
    setOpen(next)

    if (next) {
      setDraft(filter)
      setPickerOpen(false)
      setPickerQuery('')
    }
  }

  const clearFilters = () => {
    onFilterChange({ ...filter, categories: [], status: 'any' })
  }

  const clearDraft = () => {
    setDraft({ ...draft, categories: [], status: 'any' })
    setPickerOpen(false)
    setPickerQuery('')
  }

  const applyDraft = () => {
    onFilterChange({ ...filter, categories: draft.categories, status: draft.status })
    setOpen(false)
  }

  const setStatus = (status: ConnectionStatusFilter) => setDraft({ ...draft, status })

  const toggleCategory = (category: ConnectionCategoryId) => {
    setDraft({
      ...draft,
      categories: toggleCategoryFilter(draft.categories, category, categories)
    })
  }

  const clearCategories = () => {
    setDraft({ ...draft, categories: [] })
    setPickerOpen(false)
    setPickerQuery('')
  }

  const needle = pickerQuery.trim().toLowerCase()
  const visibleCategories = categories.filter(category =>
    copy.categories[category].name.toLowerCase().includes(needle)
  )
  const selectedCategories = draft.categories.map(category => copy.categories[category].name)

  return (
    <div className="conn-toolbar-wrap">
      <div className="conn-toolbar">
        <div className="conn-search">
          <Codicon aria-hidden className="conn-search-icon" name="search" size="0.9375rem" />
          <input
            aria-label={searchPlaceholder}
            onChange={event => onFilterChange({ ...filter, query: event.target.value })}
            placeholder={searchPlaceholder}
            ref={input}
            spellCheck={false}
            type="text"
            value={filter.query}
          />
          {filter.query.length > 0 ? (
            <button
              aria-label={copy.search.clear}
              className="conn-search-clear"
              onClick={() => {
                onFilterChange({ ...filter, query: '' })
                input.current?.focus()
              }}
              type="button"
            >
              <Codicon aria-hidden name="close" size="0.875rem" />
            </button>
          ) : (
            <kbd aria-hidden className="conn-keycap">
              /
            </kbd>
          )}
        </div>

        <Popover onOpenChange={changeOpen} open={open}>
          <PopoverTrigger asChild>
            <Button
              aria-expanded={open}
              className={cn('conn-filter-btn', (activeCount > 0 || open) && 'is-active')}
              type="button"
              variant="secondary"
            >
              <Codicon aria-hidden name="filter" size="0.9375rem" />
              <span>{copy.filters.button}</span>
              {activeCount > 0 ? <span className="conn-filter-count">{activeCount}</span> : null}
            </Button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            aria-labelledby={titleId}
            className="conn-filter-popover"
            collisionPadding={12}
            sideOffset={8}
            variant="menu"
          >
            <div className="conn-filter-form">
              <div className="conn-filter-head">
                <h2 className="conn-filter-title" id={titleId}>
                  {copy.filters.title}
                </h2>
                <button className="conn-filter-clear" onClick={clearDraft} type="button">
                  {copy.filters.clear}
                </button>
              </div>

              <section className="conn-filter-section">
                <span className="conn-filter-label">{copy.filters.status}</span>
                <span className="conn-filter-hint">{copy.filters.chooseOne}</span>
                <div
                  aria-label={copy.filters.status}
                  className="conn-status-chips"
                  onKeyDown={onFilterGroupKeyDown}
                  role="radiogroup"
                >
                  {(['any', 'connected', 'not_connected'] as const).map(status => (
                    <button
                      aria-checked={draft.status === status}
                      className={cn('conn-filter-pill', draft.status === status && 'is-selected')}
                      key={status}
                      onClick={() => setStatus(status)}
                      role="radio"
                      type="button"
                    >
                      <span>{chipLabel(status, copy)}</span>
                      {status === 'any' ? null : (
                        <span className="conn-filter-count">
                          {status === 'connected'
                            ? statusCounts.connected
                            : statusCounts.not_connected}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
              </section>

              {hideCategoryFilter ? null : (
                <section className="conn-filter-section">
                  <span className="conn-filter-label">{copy.filters.categories}</span>
                  <button
                    aria-expanded={pickerOpen}
                    aria-haspopup="listbox"
                    className={cn(
                      'conn-filter-area-toggle',
                      'conn-picker-trigger',
                      pickerOpen && 'is-open'
                    )}
                    onClick={() => {
                      setPickerOpen(!pickerOpen)
                      setPickerQuery('')
                    }}
                    type="button"
                  >
                    <span className="conn-filter-area-value">
                      {selectedCategories.length > 0
                        ? selectedCategories.join(', ')
                        : copy.filters.allCategories}
                    </span>
                    {draft.categories.length > 0 ? (
                      <span className="conn-filter-area-meta">
                        {copy.filters.selectedCount(draft.categories.length)}
                      </span>
                    ) : null}
                    <Codicon
                      aria-hidden
                      className={cn('conn-filter-chevron', pickerOpen && 'is-open')}
                      name="chevron-down"
                      size="0.875rem"
                    />
                  </button>

                  {pickerOpen ? (
                    <div className="conn-filter-area">
                      <div className="conn-filter-area-search conn-picker-search">
                        <Codicon aria-hidden name="search" size="0.875rem" />
                        <input
                          aria-label={copy.filters.findCategory}
                          autoFocus
                          onChange={event => setPickerQuery(event.target.value)}
                          placeholder={copy.filters.findCategory}
                          type="text"
                          value={pickerQuery}
                        />
                      </div>
                      <div
                        aria-label={copy.filters.categories}
                        aria-multiselectable
                        className="conn-filter-area-list conn-picker-list"
                        onKeyDown={onFilterGroupKeyDown}
                        role="listbox"
                      >
                        <button
                          aria-selected={draft.categories.length === 0}
                          className={cn(
                            'conn-filter-area-option',
                            'conn-picker-row',
                            draft.categories.length === 0 && 'is-selected'
                          )}
                          onClick={clearCategories}
                          role="option"
                          type="button"
                        >
                          <span className="conn-filter-area-name conn-picker-row-name">
                            {copy.filters.allCategories}
                          </span>
                          {draft.categories.length === 0 ? (
                            <Codicon aria-hidden name="check" size="0.875rem" />
                          ) : null}
                        </button>
                        {visibleCategories.map(category => {
                          const selected = draft.categories.includes(category)

                          return (
                            <button
                              aria-selected={selected}
                              className={cn(
                                'conn-filter-area-option',
                                'conn-picker-row',
                                selected && 'is-selected'
                              )}
                              key={category}
                              onClick={() => toggleCategory(category)}
                              role="option"
                              type="button"
                            >
                              <span className="conn-filter-area-name conn-picker-row-name">
                                {copy.categories[category].name}
                              </span>
                              <span className="conn-filter-area-count conn-picker-row-count">
                                {categoryCounts[category]}
                              </span>
                              {selected ? <Codicon aria-hidden name="check" size="0.875rem" /> : null}
                            </button>
                          )
                        })}
                        {visibleCategories.length === 0 ? (
                          <p className="conn-filter-area-empty conn-picker-empty">
                            {copy.filters.noResultsTitle}
                          </p>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </section>
              )}

              <Button className="conn-filter-apply" onClick={applyDraft} type="button">
                {copy.filters.showConnections(draftResultCount)}
              </Button>
            </div>
          </PopoverContent>
        </Popover>

        {extraAction ? <div className="conn-toolbar-extra">{extraAction}</div> : null}
      </div>

      {activeCount > 0 ? (
        <div className="conn-chips-row">
          {filter.status === 'any' ? null : (
            <>
              <span className="conn-chips-label">{copy.filters.statusChipLabel}</span>
              <button
                aria-label={`${copy.filters.clear}: ${chipLabel(filter.status, copy)}`}
                className="conn-chip"
                onClick={() => onFilterChange({ ...filter, status: 'any' })}
                type="button"
              >
                <span>{chipLabel(filter.status, copy)}</span>
                <Codicon aria-hidden className="conn-chip-remove" name="close" size="0.75rem" />
              </button>
            </>
          )}

          {filter.categories.length > 0 ? (
            <>
              <span className="conn-chips-label">{copy.filters.categoryChipLabel}</span>
              {filter.categories.map(category => (
                <button
                  aria-label={`${copy.filters.clear}: ${copy.categories[category].name}`}
                  className="conn-chip"
                  key={category}
                  onClick={() =>
                    onFilterChange({
                      ...filter,
                      categories: filter.categories.filter(entry => entry !== category)
                    })
                  }
                  type="button"
                >
                  <span>{copy.categories[category].name}</span>
                  <Codicon aria-hidden className="conn-chip-remove" name="close" size="0.75rem" />
                </button>
              ))}
            </>
          ) : null}

          <span aria-hidden className="conn-chips-divider">
            /
          </span>
          <button className="conn-chips-clear" onClick={clearFilters} type="button">
            {copy.filters.clear}
          </button>
        </div>
      ) : null}
    </div>
  )
}
