import { type ReactNode } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'

import { CapabilitySearch } from '../ui/capability-search'

import { ConnectorRowCard } from './connector-row-card'
import { cardKey, EMPTY_CONNECTORS_FILTER } from './derive'
import { derivePage, showsAttentionFirst } from './derive-page'
import { ToolsWash } from './tools-status'
import type { ConnectorCardModel, ConnectorGroupModel, ConnectorSegmentId, ConnectorsFilter } from './types'

export interface ConnectorsDirectoryProps {
  addYourOwn?: ReactNode
  busyKey?: null | string
  cards: ConnectorCardModel[]
  filter: ConnectorsFilter
  hostedFailed?: boolean
  loading?: boolean
  notices?: ReactNode
  onFilterChange: (next: ConnectorsFilter) => void
  onOpen: (card: ConnectorCardModel) => void
  onPrefetch?: (card: ConnectorCardModel) => void
  onRetryHosted?: () => void
  onServerToggle?: (card: ConnectorCardModel, next: boolean) => void
  onVerb?: (card: ConnectorCardModel) => void
  selectedKey?: null | string
}

export function ConnectorsDirectory({
  addYourOwn,
  busyKey = null,
  cards,
  filter,
  hostedFailed = false,
  loading = false,
  notices,
  onFilterChange,
  onOpen,
  onPrefetch,
  onRetryHosted,
  onServerToggle,
  onVerb,
  selectedKey = null
}: ConnectorsDirectoryProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const where = copy.residencyLocal
  const segmentLabel = (id: ConnectorSegmentId) => (id === 'local' ? where : copy.segment[id])
  const set = (patch: Partial<ConnectorsFilter>) => onFilterChange({ ...filter, ...patch })

  const { groups, hiddenMatches, segment, segments } = derivePage(cards, filter)

  const showSegments = segments.length > 2
  const segmentFellBack = segments.length > 0 && segment !== filter.segment

  return (
    <div className="flex min-h-0 flex-1 flex-col" data-slot="connectors-directory">
      {cards.length === 0 ? null : (
        <>
          <CapabilitySearch
            actions={addYourOwn}
            onChange={query => set({ query })}
            placeholder={copy.searchPlaceholder(cards.length)}
            scope={copy.title}
            value={filter.query}
          />

          {showSegments || segmentFellBack || hiddenMatches > 0 ? (
            <div className="mt-(--capabilities-gap-controls) flex shrink-0 flex-wrap items-center gap-2">
              {showSegments ? (
                <SegmentedControl
                  onChange={(next: ConnectorSegmentId) => set({ segment: next })}
                  options={segments.map(option => ({
                    id: option.id,
                    label: `${segmentLabel(option.id)} ${option.count}`
                  }))}
                  sizing="content"
                  value={segment}
                />
              ) : null}

              {segmentFellBack ? (
                <span className="text-[0.7rem] text-(--ui-text-tertiary)">
                  {copy.page.segmentNoMatch(segmentLabel(filter.segment))}
                </span>
              ) : null}

              {hiddenMatches > 0 ? (
                <span className="flex items-center gap-1 text-[0.7rem] text-(--ui-text-tertiary)">
                  {copy.page.matchesElsewhere(hiddenMatches)}
                  <Button onClick={() => set({ segment: 'all' })} size="xs" variant="text">
                    {copy.page.showAllMatches}
                  </Button>
                </span>
              ) : null}
            </div>
          ) : null}
        </>
      )}

      {notices ? <div className="mt-(--capabilities-gap-controls) shrink-0 empty:hidden">{notices}</div> : null}

      {hostedFailed && onRetryHosted ? (
        <ErrorBanner className="mt-(--capabilities-gap-controls) shrink-0 items-center">
          <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
            <span className="font-medium">{copy.page.hostedFailedTitle}</span>
            <span className="opacity-80">{copy.page.hostedFailedBody}</span>
            <Button className="text-destructive" onClick={onRetryHosted} size="xs" variant="text">
              {copy.page.retry}
            </Button>
          </span>
        </ErrorBanner>
      ) : null}

      {loading ? (
        <div className="mt-(--capabilities-gap-lead)"><ToolsWash label={copy.page.loading} rows={10} /></div>
      ) : groups.length > 0 ? (
        <div className="capabilities-sections min-h-0 flex-1 overflow-y-auto overscroll-contain pb-4">
          {groups.map(group => (
            <Group
              busyKey={busyKey}
              group={group}
              key={group.id}
              onOpen={onOpen}
              onPrefetch={onPrefetch}
              onServerToggle={onServerToggle}
              onVerb={onVerb}
              selectedKey={selectedKey}
              where={where}
            />
          ))}
        </div>
      ) : cards.length === 0 ? (
        hostedFailed ? null : (
          <PanelEmpty action={addYourOwn} icon="plug" title={copy.page.emptyTitle} />
        )
      ) : (
        <PanelEmpty
          action={
            <div className="flex items-center gap-2">
              <Button onClick={() => set(EMPTY_CONNECTORS_FILTER)} size="xs" variant="secondary">
                {copy.page.clearSearch}
              </Button>
              {addYourOwn}
            </div>
          }
          description={copy.page.noMatchBody}
          icon="search"
          title={copy.page.noMatchTitle}
        />
      )}
    </div>
  )
}

function Group({
  busyKey,
  group,
  onOpen,
  onPrefetch,
  onServerToggle,
  onVerb,
  selectedKey,
  where
}: {
  busyKey: null | string
  group: ConnectorGroupModel
  onOpen: (card: ConnectorCardModel) => void
  onPrefetch?: (card: ConnectorCardModel) => void
  onServerToggle?: (card: ConnectorCardModel, next: boolean) => void
  onVerb?: (card: ConnectorCardModel) => void
  selectedKey: null | string
  where: string
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.group

  return (
    <section>
      <header className="catalog-section-heading">
        <div>
          <h2>
            {group.id === 'local' ? where : copy[group.id]}
            <span className="ml-2 font-normal tabular-nums text-(--ui-text-tertiary)">{group.cards.length}</span>
          </h2>
          {group.id === 'connected' && showsAttentionFirst(group) ? <p>{copy.connectedNote}</p> : null}
          {group.id === 'off' ? <p>{copy.offNote}</p> : null}
        </div>
      </header>

      <div className="grid gap-3 sm:grid-cols-2">
        {group.cards.map(card => {
          const key = cardKey(card)

          return (
            <ConnectorRowCard
              busy={busyKey === key}
              card={card}
              key={key}
              onOpen={() => onOpen(card)}
              onPrefetch={onPrefetch ? () => onPrefetch(card) : undefined}
              onServerToggle={onServerToggle ? next => onServerToggle(card, next) : undefined}
              onVerb={onVerb ? () => onVerb(card) : undefined}
              selected={selectedKey === key}
            />
          )
        })}
      </div>
    </section>
  )
}
