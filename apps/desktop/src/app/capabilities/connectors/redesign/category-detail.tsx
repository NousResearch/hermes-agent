import './connections.css'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'

import { cardKey } from '../derive'
import type { ConnectorCardModel } from '../types'

import { useConnectionsCopy } from './connections-copy'
import { ConnectorAppRow } from './connections-directory'
import { ConnectionsToolbar } from './connections-filter'
import {
  type ConnectionsFilterState,
  EMPTY_CONNECTIONS_FILTER,
  groupCardsBySubgroup
} from './filter-logic'
import { categoryOfCard, type ConnectionCategoryId, type ConnectionInterfaceMode } from './taxonomy'

export interface ConnectionsCategoryDetailProps {
  busyKey?: null | string
  cards: readonly ConnectorCardModel[]
  category: ConnectionCategoryId
  filter: ConnectionsFilterState
  mode: ConnectionInterfaceMode
  onBack: () => void
  onCtaClick: (card: ConnectorCardModel) => void
  onFilterChange: (next: ConnectionsFilterState) => void
  onOpenCard: (card: ConnectorCardModel) => void
  onPrefetch?: (card: ConnectorCardModel) => void
}

/**
 * Figma `1591:17731`: one category's every app, split into its taxonomy
 * subgroups in display order. The toolbar keeps search and status but hides
 * the category picker, which this page already is.
 */
export function ConnectionsCategoryDetail({
  busyKey = null,
  cards,
  category,
  filter,
  mode,
  onBack,
  onCtaClick,
  onFilterChange,
  onOpenCard,
  onPrefetch
}: ConnectionsCategoryDetailProps) {
  const copy = useConnectionsCopy()
  const scopedFilter: ConnectionsFilterState = { ...filter, categories: [] }
  const categoryCards = cards.filter(card => categoryOfCard(card) === category)
  const groups = groupCardsBySubgroup(cards, category, scopedFilter)

  return (
    <div className="conn-page" data-conn-mode={mode} data-owns-header="true">
      <button className="conn-back-link" onClick={onBack} type="button">
        <Codicon aria-hidden className="conn-back-glyph" name="arrow-left" size="0.875rem" />
        <span>{copy.filters.backToConnections}</span>
      </button>

      <header className="conn-header">
        <h1 className="conn-title">{copy.categories[category].name}</h1>
        <p className="conn-subtitle">
          {copy.categories[category].subtitle}
          <span aria-hidden className="conn-subtitle-divider">
            {' '}
            ·{' '}
          </span>
          {copy.filters.resultCount(categoryCards.length)}
        </p>
      </header>

      <ConnectionsToolbar
        cards={categoryCards}
        filter={scopedFilter}
        hideCategoryFilter
        mode={mode}
        onFilterChange={next => onFilterChange({ ...next, categories: [] })}
        searchPlaceholder={copy.filters.searchCategory(copy.categories[category].name)}
      />

      {groups.length === 0 ? (
        <div className="conn-empty">
          <p className="conn-empty-title">{copy.filters.noResultsTitle}</p>
          <p className="conn-empty-body">
            {scopedFilter.query.trim().length > 0
              ? copy.filters.noResultsQuery(scopedFilter.query.trim())
              : copy.filters.noResultsBody}
          </p>
          <Button
            className="conn-chips-clear"
            onClick={() => onFilterChange(EMPTY_CONNECTIONS_FILTER)}
            type="button"
            variant="text"
          >
            {copy.filters.clear}
          </Button>
        </div>
      ) : (
        <div className="conn-sections">
          {groups.map(group => (
            <section className="conn-section" key={group.id}>
              <div className="conn-section-head">
                <h2 className="conn-section-title">{copy.subgroups[group.id]}</h2>
                <span className="conn-section-count">{group.cards.length}</span>
              </div>
              <div className="conn-card-surface">
                <div className="conn-grid">
                  {group.cards.map(card => (
                    <ConnectorAppRow
                      busy={busyKey === card.slug || busyKey === cardKey(card)}
                      card={card}
                      key={cardKey(card)}
                      onCta={onCtaClick}
                      onOpen={onOpenCard}
                      onPrefetch={onPrefetch}
                    />
                  ))}
                </div>
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  )
}
