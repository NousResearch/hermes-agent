import { type ReactNode, useMemo } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorBanner } from '@/components/ui/error-state'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'

import { brandingLockedSlice } from '../free-tier'
import { LockFog, useLockFogTop } from '../lock-fog'

import { ConnectorRowCard } from './connector-row-card'
import { cardKey, EMPTY_CONNECTORS_FILTER } from './derive'
import { derivePage, showsAttentionFirst } from './derive-page'
import { ToolsWash } from './tools-status'
import type { ConnectorCardModel, ConnectorGroupModel, ConnectorsFilter } from './types'

export interface ConnectorsDirectoryProps {
  addConnector?: ReactNode
  addYourOwn?: ReactNode
  busyKey?: null | string
  cards: ConnectorCardModel[]
  filter: ConnectorsFilter
  hostedFailed?: boolean
  /** Free-plan gate: free cards stay clear; the rest sit under one page fog. */
  isLocked?: (card: ConnectorCardModel) => boolean
  lockedLabel?: string
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
  addConnector,
  addYourOwn,
  busyKey = null,
  cards,
  filter,
  hostedFailed = false,
  isLocked,
  lockedLabel,
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
  const set = (patch: Partial<ConnectorsFilter>) => onFilterChange({ ...filter, ...patch })

  const { groups } = derivePage(cards, { ...filter, segment: 'all' })
  const connectedGroups = useMemo(
    () => groups.filter(group => group.id === 'connected' || group.id === 'local' || group.id === 'off'),
    [groups]
  )
  const availableGroup = useMemo(() => groups.find(group => group.id === 'available') ?? null, [groups])
  const availableCards = availableGroup?.cards ?? []
  const connectedCards = useMemo(() => connectedGroups.flatMap(group => group.cards), [connectedGroups])
  const availableTotal = useMemo(
    () => cards.filter(card => card.state === 'available').length,
    [cards]
  )
  const hasConnected = connectedCards.length > 0

  const freeCards = useMemo(
    () => (isLocked ? availableCards.filter(card => !isLocked(card)) : availableCards),
    [availableCards, isLocked]
  )
  const lockedCards = useMemo(
    () => (isLocked ? availableCards.filter(card => isLocked(card)) : []),
    [availableCards, isLocked]
  )
  const lockedBehind = lockedCards.length > 0
  const { freeAnchor, fogTop } = useLockFogTop(lockedBehind, `${freeCards.length}:${filter.query}`)
  const lockTip = lockedLabel || 'Upgrade to unlock more'

  const renderCard = (card: ConnectorCardModel, interactive: boolean) => {
    const key = cardKey(card)

    return (
      <ConnectorRowCard
        busy={!interactive || busyKey === key}
        card={card}
        key={key}
        onOpen={interactive ? () => onOpen(card) : () => undefined}
        onPrefetch={interactive && onPrefetch ? () => onPrefetch(card) : undefined}
        onServerToggle={interactive && onServerToggle ? next => onServerToggle(card, next) : undefined}
        onVerb={interactive && onVerb ? () => onVerb(card) : undefined}
        selected={interactive && selectedKey === key}
      />
    )
  }

  if (loading) {
    return (
      <div className="flex flex-col gap-4" data-slot="connectors-directory">
        <ToolsWash label={copy.page.loading} rows={10} />
      </div>
    )
  }

  if (cards.length === 0) {
    return (
      <div className="flex flex-col gap-4" data-slot="connectors-directory">
        {hostedFailed ? null : (
          <PanelEmpty action={addYourOwn ?? addConnector} icon="plug" title={copy.page.emptyTitle} />
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4" data-slot="connectors-directory">
      {/* Connected strip — empty state or live cards */}
      {!hasConnected ? (
        <section
          className="flex flex-col items-center justify-center gap-3 px-6 py-10 text-center"
          data-connectors-empty
        >
          <div className="flex size-14 items-center justify-center rounded-2xl border border-[#e2e8f0] bg-[#f1f5f9] text-[#94a3b8]">
            <Codicon name="plug" size="1.75rem" />
          </div>
          <div className="space-y-1">
            <h2 className="text-[16px] font-semibold text-[#0f172a] dark:text-foreground">
              {copy.page.emptyConnectedTitle}
            </h2>
            <p className="text-[14px] text-[#64748b]">{copy.page.emptyConnectedBody}</p>
          </div>
          {addConnector}
        </section>
      ) : (
        <div className="grid gap-4">
          {connectedGroups.map(group => (
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
      )}

      {/* Available catalog — natural height; page scroll owns the scrollbar */}
      <section className="flex flex-col gap-3.5">
        <header className="flex items-center justify-between gap-3">
          <h2 className="text-[18px] font-bold text-[#0f172a] dark:text-foreground">{copy.group.available}</h2>
          {addYourOwn}
        </header>

        <SearchField
          containerClassName="w-full min-w-0"
          onChange={query => set({ query })}
          placeholder={copy.searchPlaceholder(Math.max(availableTotal, cards.length))}
          value={filter.query}
          variant="box"
        />

        {notices}

        {hostedFailed && onRetryHosted ? (
          <ErrorBanner className="shrink-0 items-center">
            <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
              <span className="font-medium">{copy.page.hostedFailedTitle}</span>
              <span className="opacity-80">{copy.page.hostedFailedBody}</span>
              <Button className="text-destructive" onClick={onRetryHosted} size="xs" variant="text">
                {copy.page.retry}
              </Button>
            </span>
          </ErrorBanner>
        ) : null}

        <div className="flex items-center justify-between gap-3 text-[#64748b]">
          <span className="text-[13px] font-semibold">{copy.page.availableCount(availableTotal)}</span>
          <span className="text-[12px] font-medium">{copy.page.showingCount(availableCards.length)}</span>
        </div>

        {availableCards.length > 0 && lockedBehind ? (
          <div className="relative">
            <div className="grid gap-4 pb-4">
              <div className="grid gap-4 sm:grid-cols-2" ref={freeAnchor}>
                {freeCards.map(card => renderCard(card, true))}
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                {brandingLockedSlice(lockedCards).map(card => renderCard(card, false))}
              </div>
            </div>
            {fogTop > 0 ? (
              <div className="absolute inset-x-0 bottom-0" style={{ top: fogTop }}>
                <LockFog label={lockTip} />
              </div>
            ) : null}
          </div>
        ) : availableCards.length > 0 ? (
          <div className="grid gap-4 pb-4 sm:grid-cols-2">
            {availableCards.map(card => renderCard(card, true))}
          </div>
        ) : filter.query.trim() ? (
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
        ) : null}
      </section>
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
    <section className="grid gap-3">
      <header className="flex items-center gap-2">
        <h3 className="text-[18px] font-bold text-[#0f172a] dark:text-foreground">
          {group.id === 'local' ? where : copy[group.id]}
        </h3>
        <span className="tabular-nums text-[13px] font-semibold text-[#64748b]">{group.cards.length}</span>

        {group.id === 'connected' && showsAttentionFirst(group) ? <Note>{copy.connectedNote}</Note> : null}
        {group.id === 'off' ? <Note>{copy.offNote}</Note> : null}
      </header>

      <div className="grid gap-4 sm:grid-cols-2">
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

function Note({ children }: { children: ReactNode }) {
  return <span className="truncate text-[12px] text-[#64748b]">{children}</span>
}
