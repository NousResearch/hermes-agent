import './agent-settings.css'

import { useStore } from '@nanostores/react'
import { type KeyboardEvent, useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

import { PageLoader } from '@/components/page-loader'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { RowButton } from '@/components/ui/row-button'
import { SearchField } from '@/components/ui/search-field'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $profileColors, $profiles } from '@/store/profile'

import { useFleetRoster } from '../chat/sidebar/use-fleet-roster'
import { type SimplifiedAgent, useSimplifiedAgents } from '../chat/sidebar/use-simplified-agents'
import { usePageFrameCopy } from '../overlays/page-frame-copy'
import { PageSidebarHead, usePageSidebarOpen } from '../overlays/page-sidebar'
import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'

import { $lastViewedAgent, agentPageHref, pickLandingAgent } from './landing'
import { NewAgentButton } from './new-agent-button'

/** Below this the Agents column folds away and the destination is the list itself (agent-settings.css). */
const NARROW_AGENTS = '(max-width: 800px)'
/** A short roster is scanned, not searched; the field appears once the list outgrows a glance. */
const SEARCH_FROM = 9

/** The roster every Agents surface lists, with which row is the gateway's default agent. */
function useAgentRoster() {
  const profiles = useStore($profiles)
  useFleetRoster(true)
  const { agents, owner } = useSimplifiedAgents([])
  const defaultProfile = profiles.find(item => item.is_default)?.name

  const isDefault = (agent: SimplifiedAgent) =>
    agent.profile === defaultProfile && (agent.connectionId ?? 'local') === (owner ?? 'local')

  return { agents, isDefault }
}

interface AgentRowsProps {
  agents: readonly SimplifiedAgent[]
  isDefault: (agent: SimplifiedAgent) => boolean
  selected?: (agent: SimplifiedAgent) => boolean
}

/** Up/Down move between rows, like the chat sidebar's list. */
function moveRowFocus(event: KeyboardEvent<HTMLDivElement>) {
  const step = { ArrowDown: 1, ArrowUp: -1 }[event.key]

  if (!step) {
    return
  }

  const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('.agent-management-nav-row')]
  const next = rows[rows.indexOf(document.activeElement as HTMLElement) + step]

  if (next) {
    event.preventDefault()
    next.focus()
  }
}

/** "N AGENTS" and one row per agent: tinted tile, name, DEFAULT chip (895:9473). */
function AgentRows({ agents, isDefault, selected }: AgentRowsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const colors = useStore($profileColors)

  return (
    <div className="agent-management-nav-list" onKeyDown={moveRowFocus}>
      <p className="agent-management-nav-count">{t.agents.agentsCount(agents.length)}</p>
      {agents.map(agent => (
        <RowButton
          aria-current={selected?.(agent) ? 'page' : undefined}
          aria-label={`${agent.label} · ${agent.gateway}`}
          className="agent-management-nav-row"
          key={agent.id}
          onClick={() => navigate(agentPageHref(agent))}
        >
          <FigmaAgentIcon color={resolveProfileColor(agent.profile, colors)} variant="subtle" />
          <span className="min-w-0 flex-1 truncate">{agent.label}</span>
          {isDefault(agent) && <span className="agent-management-default">{t.profiles.defaultBadge}</span>}
        </RowButton>
      ))}
    </div>
  )
}

interface AgentNavigationProps {
  profile: string
  connectionId: string
}

/** Keep the selected agent's neighbours reachable while inspecting its settings. */
export function AgentNavigation({ profile, connectionId }: AgentNavigationProps) {
  const { t } = useI18n()
  const { agents, isDefault } = useAgentRoster()
  const panelOpen = usePageSidebarOpen()

  if (agents.length === 0 || !panelOpen) {
    return null
  }

  return (
    <aside aria-label={t.simplifiedAgents.agents} className="agent-management-nav">
      <PageSidebarHead title={t.simplifiedAgents.agents} />
      <NewAgentButton owner={{ connectionId, profile }} row />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <AgentRows
          agents={agents}
          isDefault={isDefault}
          selected={agent => agent.profile === profile && (agent.connectionId ?? 'local') === connectionId}
        />
      </div>
    </aside>
  )
}

/** The Agents destination. A normal window opens an agent's page (its column lists the others);
 *  a narrow one, where that column folds away, lands on the list itself and a row opens the page. */
export function AgentDirectory({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const frame = usePageFrameCopy()
  const narrow = useMediaQuery(NARROW_AGENTS)
  const lastViewed = useStore($lastViewedAgent)
  const [query, setQuery] = useState('')
  const { agents, isDefault } = useAgentRoster()
  const landing = pickLandingAgent(agents, lastViewed, isDefault)
  const navigate = useNavigate()
  const target = landing && !narrow ? agentPageHref(landing) : null

  useEffect(() => {
    if (target) {
      navigate(target, { replace: true })
    }
  }, [navigate, target])

  // Every gateway has at least its default agent, so an empty roster is still loading.
  if (!landing || target) {
    return <PageLoader />
  }

  const needle = query.trim().toLocaleLowerCase()
  const visible = agents.filter(agent => `${agent.label} ${agent.gateway}`.toLocaleLowerCase().includes(needle))

  return (
    <Panel onClose={onClose}>
      <PanelHeader actions={<NewAgentButton />} subtitle={frame.agentsDescription} title={t.profiles.title} />
      {agents.length >= SEARCH_FROM && (
        <SearchField
          aria-label={t.profiles.title}
          containerClassName="mb-6"
          onChange={setQuery}
          placeholder={t.titlebar.search}
          value={query}
          variant="box"
        />
      )}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {visible.length === 0 ? (
          <PanelEmpty title={t.commandCenter.noResults} />
        ) : (
          <AgentRows agents={visible} isDefault={isDefault} />
        )}
      </div>
    </Panel>
  )
}
