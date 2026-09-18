// The page: a list you read through, not a wall of tiles.
//
// Search is the first control, then exactly one row of filters, then the groups.
// There is one way to filter and one way in to an app, and the page never moves a
// card under the person because something finished in the background — the
// grouping is derived from props, so a card only moves when its props do.

import { type ReactNode, useState } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'

import { ConnectorRowCard } from './connector-row-card'
import { cardCategoryOptions, filterCards, groupCards, inventoryCounts, pillCounts } from './derive'
import { ToolsWash } from './tools-status'
import type { ConnectorCardModel, ConnectorGroupModel, ConnectorPillId, ConnectorsFilter } from './types'

/** Four cards — two rows of two — before the group asks to be opened. Long enough
 *  to be a sample, short enough that the groups below it stay on screen. */
const AVAILABLE_PREVIEW = 4

const RESIDENCY_VALUES = { all: null, hosted: 'hosted', local: 'local' } as const

export interface ConnectorsDirectoryProps {
  /** The page action, top right. The wiring slice owns what it opens. */
  addYourOwn?: ReactNode
  /** The app whose write is in flight, so its verb can say so. */
  busySlug?: null | string
  cards: ConnectorCardModel[]
  filter: ConnectorsFilter
  /** Only the hosted half failed. The servers on this Mac still render. */
  hostedFailed?: boolean
  loading?: boolean
  onAddServer?: () => void
  onFilterChange: (next: ConnectorsFilter) => void
  onOpen: (card: ConnectorCardModel) => void
  onRetryHosted?: () => void
  onServerToggle?: (card: ConnectorCardModel, next: boolean) => void
  onVerb?: (card: ConnectorCardModel) => void
  /** `Configuring:` — which profile the servers on this Mac belong to. */
  profileSelector?: ReactNode
  selectedSlug?: null | string
}

export function ConnectorsDirectory({
  addYourOwn,
  busySlug = null,
  cards,
  filter,
  hostedFailed = false,
  loading = false,
  onAddServer,
  onFilterChange,
  onOpen,
  onRetryHosted,
  onServerToggle,
  onVerb,
  profileSelector,
  selectedSlug = null
}: ConnectorsDirectoryProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const set = (patch: Partial<ConnectorsFilter>) => onFilterChange({ ...filter, ...patch })

  const inventory = inventoryCounts(cards)
  // Counted with every filter except the pill itself, so the numbers describe
  // what pressing a pill would actually show.
  const pills = pillCounts(filterCards(cards, { ...filter, pill: 'all' }))
  const options = pills.some(pill => pill.id === filter.pill) ? pills : [...pills, { count: 0, id: filter.pill }]
  const groups = groupCards(filterCards(cards, filter))

  return (
    <div className="flex min-h-0 flex-col gap-3" data-slot="connectors-directory">
      <div className="flex items-center gap-3">
        <h2 className="flex-1 text-sm font-semibold text-(--ui-text-primary)">{copy.title}</h2>
        {addYourOwn}
      </div>

      {/* An empty list hides its search field, and with it the inventory line and
          the filter row: there is nothing to narrow, and a control that can only
          return the same nothing is noise in front of the way in. */}
      {cards.length === 0 ? null : (
        <>
          <div className="flex items-center gap-3 border-b border-(--ui-stroke-tertiary) pb-1.5">
            <SearchField
              containerClassName="min-w-0 flex-1"
              onChange={query => set({ query })}
              placeholder={copy.searchPlaceholder(cards.length)}
              value={filter.query}
            />
            <span className="shrink-0 text-[0.65rem] text-(--ui-text-quaternary)">
              {copy.inventory(inventory.hosted, inventory.local)}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <SegmentedControl
              onChange={(pill: ConnectorPillId) => set({ pill })}
              options={options.map(pill => ({ id: pill.id, label: `${copy.pill[pill.id]} ${pill.count}` }))}
              value={filter.pill}
            />

            <div className="ml-auto flex items-center gap-2">
              <Select
                onValueChange={value => set({ residency: RESIDENCY_VALUES[value as keyof typeof RESIDENCY_VALUES] })}
                value={filter.residency ?? 'all'}
              >
                <SelectTrigger aria-label={copy.filterWhere} className="w-auto" size="xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">{copy.whereAll}</SelectItem>
                  <SelectItem value="hosted">{copy.whereHosted}</SelectItem>
                  <SelectItem value="local">{copy.whereLocal}</SelectItem>
                </SelectContent>
              </Select>

              <CategorySelect
                categories={cardCategoryOptions(cards, filter)}
                onChange={category => set({ category })}
                value={filter.category}
              />
            </div>
          </div>
        </>
      )}

      <p className="text-[0.7rem] text-(--ui-text-tertiary)">{copy.accountNote}</p>

      {hostedFailed && onRetryHosted ? (
        <ErrorBanner className="items-center">
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
        <ToolsWash label={copy.page.loading} rows={10} />
      ) : groups.length > 0 ? (
        <div className="grid min-h-0 gap-6 overflow-y-auto pb-4">
          {groups.map(group => (
            <Group
              busySlug={busySlug}
              group={group}
              key={group.id}
              onAddServer={onAddServer}
              onOpen={onOpen}
              onServerToggle={onServerToggle}
              onVerb={onVerb}
              profileSelector={profileSelector}
              selectedSlug={selectedSlug}
            />
          ))}
        </div>
      ) : cards.length === 0 ? (
        <PanelEmpty action={addYourOwn} description={copy.page.emptyBody} icon="plug" title={copy.page.emptyTitle} />
      ) : (
        <PanelEmpty
          action={
            <div className="flex items-center gap-2">
              <Button
                onClick={() => set({ category: null, pill: 'all', query: '', residency: null })}
                size="xs"
                variant="secondary"
              >
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
  busySlug,
  group,
  onAddServer,
  onOpen,
  onServerToggle,
  onVerb,
  profileSelector,
  selectedSlug
}: {
  busySlug: null | string
  group: ConnectorGroupModel
  onAddServer?: () => void
  onOpen: (card: ConnectorCardModel) => void
  onServerToggle?: (card: ConnectorCardModel, next: boolean) => void
  onVerb?: (card: ConnectorCardModel) => void
  profileSelector?: ReactNode
  selectedSlug: null | string
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.group
  const [expanded, setExpanded] = useState(false)
  const truncates = group.id === 'available' && group.cards.length > AVAILABLE_PREVIEW
  const shown = truncates && !expanded ? group.cards.slice(0, AVAILABLE_PREVIEW) : group.cards

  // No tint and no frame around a group: the row-cards are boxes already, and a
  // rounded fill behind them is the card-in-card the design system rules out.
  // The heading and the gap do the grouping.
  return (
    <section className="grid gap-2">
      <header className="flex items-center gap-2">
        <h3 className="text-xs font-semibold text-(--ui-text-primary)">{copy[group.id]}</h3>
        <span className="tabular-nums text-xs text-(--ui-text-tertiary)">{group.cards.length}</span>

        {group.id === 'connected' ? <Note>{copy.connectedNote}</Note> : null}
        {group.id === 'off' ? <Note>{copy.offNote}</Note> : null}

        {group.id === 'local' ? (
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[0.65rem] text-(--ui-text-tertiary)">{copy.localConfiguring}</span>
            {profileSelector}
            {onAddServer ? (
              <Button onClick={onAddServer} size="xs" variant="outline">
                {copy.localAddServer}
              </Button>
            ) : null}
          </div>
        ) : null}

        {truncates ? (
          <Button className="ml-auto" onClick={() => setExpanded(!expanded)} size="xs" variant="text">
            {expanded ? copy.availableShowFewer : copy.availableShowAll(group.cards.length)}
          </Button>
        ) : null}
      </header>

      <div className="grid gap-3 sm:grid-cols-2">
        {shown.map(card => (
          <ConnectorRowCard
            busy={busySlug === card.slug}
            card={card}
            key={`${card.residency}:${card.slug}`}
            onOpen={() => onOpen(card)}
            onServerToggle={onServerToggle ? next => onServerToggle(card, next) : undefined}
            onVerb={onVerb ? () => onVerb(card) : undefined}
            selected={selectedSlug === card.slug}
          />
        ))}
      </div>
    </section>
  )
}

function Note({ children }: { children: ReactNode }) {
  return <span className="truncate text-[0.65rem] text-(--ui-text-tertiary)">{children}</span>
}

/** The page's category picker. Same control as the tool list's, different list. */
function CategorySelect({
  categories,
  onChange,
  value
}: {
  categories: { count: number; name: string }[]
  onChange: (next: null | string) => void
  value: null | string
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  if (categories.length === 0) {
    return null
  }

  return (
    <Select onValueChange={next => onChange(next === 'all' ? null : next)} value={value ?? 'all'}>
      <SelectTrigger aria-label={copy.filterCategory} className="w-auto" size="xs">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="all">{copy.categoryAll}</SelectItem>
        {categories.map(option => (
          <SelectItem key={option.name} value={option.name}>
            {`${option.name} ${option.count}`}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
