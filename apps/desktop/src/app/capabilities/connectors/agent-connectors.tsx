import { type ReactNode, useState } from 'react'

import { PanelEmpty } from '@/app/overlays/panel'
import { Button } from '@/components/ui/button'
import { ErrorBanner } from '@/components/ui/error-state'
import { useI18n } from '@/i18n'

import { SettingsAddRow } from '../../settings/settings-card'
import { AgentCapabilityMore, AgentCapabilitySection } from '../ui/agent-capability'
import { useAgentCapabilityCopy } from '../ui/agent-capability-copy'
import { CapabilitySearch } from '../ui/capability-search'

import { ConnectorRowCard } from './connector-row-card'
import { cardKey, EMPTY_CONNECTORS_FILTER } from './derive'
import { cardMatchesQuery, groupIdOf } from './derive-page'
import { ToolsWash } from './tools-status'
import type { ConnectorCardModel, ConnectorsFilter } from './types'

/** Library apps the "Not added" card lists before "Show all". */
const AVAILABLE_PREVIEW = 8

interface AgentConnectorsProps {
  cards: ConnectorCardModel[]
  filter: ConnectorsFilter
  onFilterChange: (next: ConnectorsFilter) => void
  busyKey: null | string
  selectedKey: null | string
  loading: boolean
  hostedFailed: boolean
  notices: ReactNode
  addDisabled: boolean
  onAddServer: () => void
  onOpen: (card: ConnectorCardModel) => void
  onPrefetch: (card: ConnectorCardModel) => void
  onRetryHosted: () => void
  onServerToggle?: (card: ConnectorCardModel, next: boolean) => void
  onVerb: (card: ConnectorCardModel) => void
}

type SectionId = 'connected' | 'servers' | 'available'

// Hosted apps signed in (or switched off for this agent), this agent's MCP servers, then the library.
const sectionOf = (card: ConnectorCardModel): SectionId => {
  const group = groupIdOf(card)

  return group === 'local' ? 'servers' : group === 'available' ? 'available' : 'connected'
}

/** One agent's connectors (Figma 895:15892): Connected apps, MCP servers (+ Add MCP server), and
 *  what it has not added yet. Rows keep the full connector card: state, verb, switch and details. */
export function AgentConnectors({
  cards,
  filter,
  onFilterChange,
  busyKey,
  selectedKey,
  loading,
  hostedFailed,
  notices,
  addDisabled,
  onAddServer,
  onOpen,
  onPrefetch,
  onRetryHosted,
  onServerToggle,
  onVerb
}: AgentConnectorsProps) {
  const { t } = useI18n()
  const copy = useAgentCapabilityCopy()
  const page = t.connectorsPage
  const [showAll, setShowAll] = useState(false)
  const matches = cards.filter(card => cardMatchesQuery(card, filter.query))
  const pick = (id: SectionId) => matches.filter(card => sectionOf(card) === id)
  const connected = pick('connected')
  const servers = pick('servers')
  const available = pick('available')
  const availableShown = showAll || filter.query ? available : available.slice(0, AVAILABLE_PREVIEW)

  const rows = (items: ConnectorCardModel[]) =>
    items.map(card => (
      <ConnectorRowCard
        busy={busyKey === cardKey(card)}
        card={card}
        key={cardKey(card)}
        onOpen={() => onOpen(card)}
        onPrefetch={() => onPrefetch(card)}
        onServerToggle={onServerToggle ? next => onServerToggle(card, next) : undefined}
        onVerb={() => onVerb(card)}
        selected={selectedKey === cardKey(card)}
      />
    ))

  return (
    <>
      {cards.length > 0 && (
        <CapabilitySearch
          onChange={query => onFilterChange({ ...filter, query })}
          placeholder={page.searchPlaceholder(cards.length)}
          scope={page.title}
          value={filter.query}
        />
      )}
      {notices}
      {hostedFailed && (
        <ErrorBanner className="items-center">
          <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
            <span className="font-medium">{page.page.hostedFailedTitle}</span>
            <span className="opacity-80">{page.page.hostedFailedBody}</span>
            <Button className="text-destructive" onClick={onRetryHosted} size="xs" variant="text">
              {page.page.retry}
            </Button>
          </span>
        </ErrorBanner>
      )}
      {loading ? (
        <ToolsWash label={page.page.loading} rows={6} />
      ) : (
        <>
          {connected.length > 0 && (
            <AgentCapabilitySection description={copy.connectors.connectedDesc} title={copy.connectors.connected}>
              {rows(connected)}
            </AgentCapabilitySection>
          )}
          {/* MCP servers always show: the card is where this agent adds its own. */}
          <AgentCapabilitySection
            description={copy.connectors.serversDesc}
            footer={
              <SettingsAddRow disabled={addDisabled} onClick={onAddServer}>
                {copy.connectors.addServer}
              </SettingsAddRow>
            }
            title={copy.connectors.servers}
          >
            {rows(servers)}
          </AgentCapabilitySection>
          {available.length > 0 && (
            <AgentCapabilitySection
              description={copy.notAddedDesc}
              footer={
                availableShown.length < available.length && (
                  <AgentCapabilityMore label={copy.showAll(available.length)} onClick={() => setShowAll(true)} />
                )
              }
              title={copy.notAdded}
            >
              {rows(availableShown)}
            </AgentCapabilitySection>
          )}
          {matches.length === 0 && cards.length > 0 && (
            <PanelEmpty
              action={
                <Button onClick={() => onFilterChange(EMPTY_CONNECTORS_FILTER)} size="xs" variant="secondary">
                  {page.page.clearSearch}
                </Button>
              }
              description={page.page.noMatchBody}
              icon="search"
              title={page.page.noMatchTitle}
            />
          )}
        </>
      )}
    </>
  )
}
