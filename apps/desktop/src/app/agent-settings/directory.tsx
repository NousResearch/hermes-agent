import './agent-settings.css'

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

import { PageLoader } from '@/components/page-loader'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { SearchField } from '@/components/ui/search-field'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $profileColors, $profiles } from '@/store/profile'

import { useFleetRoster } from '../chat/sidebar/use-fleet-roster'
import { type SimplifiedAgent, useSimplifiedAgents } from '../chat/sidebar/use-simplified-agents'
import { usePageFrameCopy } from '../overlays/page-frame-copy'
import {
  PAGE_SIDEBAR_NARROW,
  PageSidebarCaption,
  PageSidebarList,
  PageSidebarPanel,
  PageSidebarRow
} from '../overlays/page-sidebar'
import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'

import { $lastViewedAgent, agentPageHref, pickLandingAgent } from './landing'
import { NewAgentButton } from './new-agent-button'

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

/** "N AGENTS" and one row per agent: tinted tile, name, DEFAULT chip (895:9473). */
function AgentRows({ agents, isDefault, selected }: AgentRowsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const colors = useStore($profileColors)

  return (
    <PageSidebarList>
      <PageSidebarCaption>{t.agents.agentsCount(agents.length)}</PageSidebarCaption>
      {agents.map(agent => (
        <PageSidebarRow
          aria-label={`${agent.label} · ${agent.gateway}`}
          current={selected?.(agent)}
          key={agent.id}
          onClick={() => navigate(agentPageHref(agent))}
        >
          <FigmaAgentIcon color={resolveProfileColor(agent.profile, colors)} variant="subtle" />
          <span className="min-w-0 flex-1 truncate">{agent.label}</span>
          {isDefault(agent) && <span className="agent-management-default">{t.profiles.defaultBadge}</span>}
        </PageSidebarRow>
      ))}
    </PageSidebarList>
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

  return (
    <PageSidebarPanel
      action={<NewAgentButton owner={{ connectionId, profile }} row />}
      title={t.simplifiedAgents.agents}
    >
      {/* Every gateway has at least its default agent, so an empty roster is still loading. */}
      {agents.length > 0 && (
        <AgentRows
          agents={agents}
          isDefault={isDefault}
          selected={agent => agent.profile === profile && (agent.connectionId ?? 'local') === connectionId}
        />
      )}
    </PageSidebarPanel>
  )
}

/** The Agents destination. A normal window opens an agent's page (its column lists the others);
 *  a narrow one, where that column folds away, lands on the list itself and a row opens the page. */
export function AgentDirectory({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const frame = usePageFrameCopy()
  const narrow = useMediaQuery(PAGE_SIDEBAR_NARROW)
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
