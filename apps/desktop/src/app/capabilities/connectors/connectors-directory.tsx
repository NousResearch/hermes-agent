import { type ReactNode } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'

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
  loading?: boolean
  onFilterChange: (next: ConnectorsFilter) => void
  onOpen: (card: ConnectorCardModel) => void
  onServerToggle?: (card: ConnectorCardModel, next: boolean) => void
  onVerb?: (card: ConnectorCardModel) => void
  selectedKey?: null | string
}

export function ConnectorsDirectory({
  addYourOwn,
  busyKey = null,
  cards,
  filter,
  loading = false,
  onFilterChange,
  onOpen,
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
    <div className="flex min-h-0 flex-1 flex-col gap-3" data-slot="connectors-directory">
      <div className="flex shrink-0 items-center gap-3">
        <h2 className="flex-1 text-sm font-semibold text-(--ui-text-primary)">{copy.title}</h2>
        {addYourOwn}
      </div>

      {cards.length === 0 ? null : (
        <>
          <div className="flex shrink-0 items-center gap-3 border-b border-(--ui-stroke-tertiary) pb-1.5">
            <SearchField
              containerClassName="min-w-0 flex-1"
              inputClassName="flex-1"
              onChange={query => set({ query })}
              placeholder={copy.searchPlaceholder(cards.length)}
              value={filter.query}
            />
          </div>

          {showSegments || segmentFellBack || hiddenMatches > 0 ? (
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              {showSegments ? (
                <SegmentedControl
                  onChange={(next: ConnectorSegmentId) => set({ segment: next })}
                  options={segments.map(option => ({
                    id: option.id,
                    label: `${segmentLabel(option.id)} ${option.count}`
                  }))}
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

      {loading ? (
        <ToolsWash label={copy.page.loading} rows={10} />
      ) : groups.length > 0 ? (
        <div className="grid min-h-0 flex-1 content-start gap-6 overflow-y-auto overscroll-contain pb-4">
          {groups.map(group => (
            <Group
              busyKey={busyKey}
              group={group}
              key={group.id}
              onOpen={onOpen}
              onServerToggle={onServerToggle}
              onVerb={onVerb}
              selectedKey={selectedKey}
              where={where}
            />
          ))}
        </div>
      ) : cards.length === 0 ? (
        <PanelEmpty action={addYourOwn} icon="plug" title={copy.page.emptyTitle} />
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
  onServerToggle,
  onVerb,
  selectedKey,
  where
}: {
  busyKey: null | string
  group: ConnectorGroupModel
  onOpen: (card: ConnectorCardModel) => void
  onServerToggle?: (card: ConnectorCardModel, next: boolean) => void
  onVerb?: (card: ConnectorCardModel) => void
  selectedKey: null | string
  where: string
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.group

  return (
    <section className="grid gap-2">
      <header className="flex items-center gap-2">
        <h3 className="text-xs font-semibold text-(--ui-text-primary)">
          {group.id === 'local' ? where : copy[group.id]}
        </h3>
        <span className="tabular-nums text-xs text-(--ui-text-tertiary)">{group.cards.length}</span>

        {group.id === 'connected' && showsAttentionFirst(group) ? <Note>{copy.connectedNote}</Note> : null}
        {group.id === 'off' ? <Note>{copy.offNote}</Note> : null}
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

function Note({ children }: { children: ReactNode }) {
  return <span className="truncate text-[0.65rem] text-(--ui-text-tertiary)">{children}</span>
}
