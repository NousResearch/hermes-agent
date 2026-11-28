import './connections.css'

import { type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConnectorLogo } from '@/components/ui/connector-logo'
import { cn } from '@/lib/utils'

import { cardKey } from '../derive'
import type { ConnectorCardModel } from '../types'

import { type ConnectionsCopy, useConnectionsCopy } from './connections-copy'
import { ConnectionsToolbar } from './connections-filter'
import {
  type ConnectionCardSection,
  type ConnectionsFilterState,
  EMPTY_CONNECTIONS_FILTER,
  groupCardsForMainPage
} from './filter-logic'
import {
  cardCtaKind,
  categoryOfCard,
  type ConnectionCategoryId,
  type ConnectionInterfaceMode
} from './taxonomy'

export interface ConnectorAppRowProps {
  busy: boolean
  card: ConnectorCardModel
  onCta: (card: ConnectorCardModel) => void
  onOpen: (card: ConnectorCardModel) => void
  onPrefetch?: (card: ConnectorCardModel) => void
}

/** One app line: the 50px mark, a name and one clipped description, then its action. */
export function ConnectorAppRow({ busy, card, onCta, onOpen, onPrefetch }: ConnectorAppRowProps) {
  const copy = useConnectionsCopy()
  const kind = cardCtaKind(card)
  const category = categoryOfCard(card)
  const description =
    card.description?.trim() ||
    (category === 'other' ? copy.sections.other : copy.categories[category].subtitle)
  const label = busy ? copy.card.waiting : copy.card[kind]

  return (
    <div className="conn-app-row" onMouseEnter={() => onPrefetch?.(card)}>
      <button
        className="conn-app-trigger"
        onClick={() => onOpen(card)}
        onFocus={() => onPrefetch?.(card)}
        type="button"
      >
        <span className="conn-logo-tile">
          <ConnectorLogo className="conn-logo-mark" connector={{ name: card.name }} />
        </span>
        <span className="conn-app-text">
          <span className="conn-app-name">{card.name}</span>
          <span className="conn-app-desc">{description}</span>
        </span>
      </button>
      <Button
        className={cn('conn-cta-btn', kind === 'manage' && 'is-manage')}
        disabled={busy}
        onClick={() => onCta(card)}
        size="sm"
        type="button"
        variant="secondary"
      >
        {label}
      </Button>
    </div>
  )
}

function sectionTitle(section: ConnectionCardSection, copy: ConnectionsCopy): string {
  if (section.id === 'connected') {
    return copy.sections.connected
  }

  if (section.id === 'suggested') {
    return copy.sections.suggested
  }

  return section.id === 'other' ? copy.sections.other : copy.categories[section.id].name
}

function DirectorySkeleton() {
  return (
    <div aria-busy="true" className="conn-sections">
      {[0, 1].map(section => (
        <section className="conn-section" key={section}>
          <div className="conn-section-head">
            <span className="conn-skeleton-heading" />
          </div>
          <div className="conn-card-surface">
            <div className="conn-grid">
              {[0, 1, 2, 3].map(row => (
                <div className="conn-app-row conn-skeleton-row" key={row}>
                  <span className="conn-skeleton-tile" />
                  <span className="conn-skeleton-lines">
                    <span className="conn-skeleton-line" />
                    <span className="conn-skeleton-line is-short" />
                  </span>
                  <span className="conn-skeleton-cta" />
                </div>
              ))}
            </div>
          </div>
        </section>
      ))}
    </div>
  )
}

function DirectoryEmpty({
  filter,
  onClear
}: {
  filter: ConnectionsFilterState
  onClear: () => void
}) {
  const copy = useConnectionsCopy()
  const query = filter.query.trim()

  return (
    <div className="conn-empty">
      <p className="conn-empty-title">{copy.filters.noResultsTitle}</p>
      <p className="conn-empty-body">
        {query.length > 0 ? copy.filters.noResultsQuery(query) : copy.filters.noResultsBody}
      </p>
      <button className="conn-chips-clear" onClick={onClear} type="button">
        {copy.filters.clear}
      </button>
    </div>
  )
}

export interface ConnectionsDirectoryRedesignProps {
  addDisabled?: boolean
  busyKey?: null | string
  cards: readonly ConnectorCardModel[]
  filter: ConnectionsFilterState
  hideHeader?: boolean
  hostedFailed?: boolean
  loading?: boolean
  mode: ConnectionInterfaceMode
  notices?: ReactNode
  onAddServer?: () => void
  onCtaClick: (card: ConnectorCardModel) => void
  onFilterChange: (next: ConnectionsFilterState) => void
  onOpenCard: (card: ConnectorCardModel) => void
  onPrefetch?: (card: ConnectorCardModel) => void
  onRetryHosted?: () => void
  onSelectCategory: (category: ConnectionCategoryId) => void
}

/**
 * Figma `1515:11577`: Connected, Suggested, then every category in the mode's
 * order, each a two-column surface of app rows. The page defers the drawer,
 * connect and install flows to its parent through the row callbacks.
 */
export function ConnectionsDirectoryRedesign({
  addDisabled = false,
  busyKey = null,
  cards,
  filter,
  hideHeader = false,
  hostedFailed = false,
  loading = false,
  mode,
  notices,
  onAddServer,
  onCtaClick,
  onFilterChange,
  onOpenCard,
  onPrefetch,
  onRetryHosted,
  onSelectCategory
}: ConnectionsDirectoryRedesignProps) {
  const copy = useConnectionsCopy()
  const sections = groupCardsForMainPage(cards, mode, filter)
  const showSkeleton = loading && cards.length === 0
  const addAction =
    onAddServer === undefined ? undefined : (
      <Button
        className="conn-add-btn"
        disabled={addDisabled}
        onClick={onAddServer}
        type="button"
        variant="secondary"
      >
        <Codicon aria-hidden name="add" size="0.875rem" />
        <span>{copy.filters.addMcpServer}</span>
      </Button>
    )

  return (
    <div className="conn-page" data-conn-mode={mode}>
      {hideHeader ? null : (
        <header className="conn-header">
          <h1 className="conn-title">{copy.page.title}</h1>
          <p className="conn-subtitle">{copy.page.subtitle}</p>
        </header>
      )}

      {notices ? <div className="conn-notices">{notices}</div> : null}

      {hostedFailed ? (
        <div className="conn-banner conn-banner-error" role="alert">
          <span className="conn-banner-text">
            <strong>{copy.errors.loadTitle}</strong>
            <span>{copy.errors.loadBody}</span>
          </span>
          {onRetryHosted ? (
            <button className="conn-retry" onClick={onRetryHosted} type="button">
              {copy.errors.tryAgain}
            </button>
          ) : null}
        </div>
      ) : null}

      <ConnectionsToolbar
        cards={cards}
        extraAction={addAction}
        filter={filter}
        mode={mode}
        onFilterChange={onFilterChange}
        searchPlaceholder={copy.search.placeholder}
      />

      {showSkeleton ? (
        <DirectorySkeleton />
      ) : sections.length === 0 ? (
        <DirectoryEmpty
          filter={filter}
          onClear={() => onFilterChange(EMPTY_CONNECTIONS_FILTER)}
        />
      ) : (
        <div className="conn-sections">
          {sections.map(section => {
            const category =
              section.id === 'connected' || section.id === 'suggested' || section.id === 'other'
                ? null
                : section.id

            return (
              <section className="conn-section" key={`${section.kind}:${section.id}`}>
                <div className="conn-section-head">
                  <h2 className="conn-section-title">{sectionTitle(section, copy)}</h2>
                  {category !== null ? (
                    <button
                      className="conn-section-link"
                      onClick={() => onSelectCategory(category)}
                      type="button"
                    >
                      {copy.sections.seeAll(section.totalCount)}
                    </button>
                  ) : section.kind === 'category' ? (
                    <span className="conn-section-count">{section.totalCount}</span>
                  ) : null}
                </div>

                {section.cards.length === 0 ? null : (
                  <div className="conn-card-surface">
                    <div className="conn-grid">
                      {section.cards.map(card => (
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
                )}
              </section>
            )
          })}
        </div>
      )}
    </div>
  )
}
