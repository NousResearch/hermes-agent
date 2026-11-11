import { useStore } from '@nanostores/react'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { Codicon } from '@/components/ui/codicon'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { RowButton } from '@/components/ui/row-button'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $profileColors, $profiles } from '@/store/profile'

import { useFleetRoster } from '../chat/sidebar/use-fleet-roster'
import { useSimplifiedAgents } from '../chat/sidebar/use-simplified-agents'
import { Panel, PanelEmpty, PanelHeader } from '../overlays/panel'
import { SETTINGS_ROUTE } from '../routes'

import { NewAgentButton } from './new-agent-button'

interface AgentNavigationProps {
  profile: string
  connectionId: string
}

/** Keep the selected agent's neighbours reachable while inspecting its settings. */
export function AgentNavigation({ profile, connectionId }: AgentNavigationProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const colors = useStore($profileColors)
  const profiles = useStore($profiles)
  useFleetRoster(true)
  const { agents, owner } = useSimplifiedAgents([])
  const defaultProfile = profiles.find(item => item.is_default)?.name

  if (agents.length < 2) {
    return null
  }

  return (
    <aside aria-label={t.simplifiedAgents.agents} className="agent-management-nav">
      <h2 className="page-nav-title">{t.simplifiedAgents.agents}</h2>
      <NewAgentButton owner={{ connectionId, profile }} />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <p className="agent-management-nav-count">{t.agents.agentsCount(agents.length)}</p>
        {agents.map(agent => (
          <RowButton
            aria-current={
              agent.profile === profile && (agent.connectionId ?? 'local') === connectionId ? 'page' : undefined
            }
            aria-label={`${agent.label} · ${agent.gateway}`}
            className="agent-management-nav-row"
            key={agent.id}
            onClick={() => {
              const params = new URLSearchParams({ agentProfile: agent.profile })

              if (agent.connectionId) {
                params.set('connection', agent.connectionId)
              }

              navigate(`${SETTINGS_ROUTE}?${params}`)
            }}
          >
            <FigmaAgentIcon color={resolveProfileColor(agent.profile, colors)} variant="subtle" />
            <span className="min-w-0 flex-1 truncate">{agent.label}</span>
            {agent.profile === defaultProfile && (agent.connectionId ?? 'local') === (owner ?? 'local') && (
              <span className="agent-management-default">{t.profiles.defaultBadge}</span>
            )}
          </RowButton>
        ))}
      </div>
    </aside>
  )
}

export function AgentDirectory({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const colors = useStore($profileColors)
  const [query, setQuery] = useState('')
  useFleetRoster(true)
  const { agents } = useSimplifiedAgents([])
  const needle = query.trim().toLocaleLowerCase()
  const visible = agents.filter(agent => `${agent.label} ${agent.gateway}`.toLocaleLowerCase().includes(needle))

  return (
    <Panel onClose={onClose}>
      <PanelHeader actions={<NewAgentButton />} title={t.profiles.title} />
      <SearchField
        aria-label={t.profiles.title}
        containerClassName="mb-6 max-w-md"
        onChange={setQuery}
        placeholder={t.titlebar.search}
        value={query}
      />
      <div className="min-h-0 flex-1 overflow-y-auto">
        {visible.length === 0 ? (
          <PanelEmpty icon="robot" title={needle ? t.commandCenter.noResults : t.profiles.noProfiles} />
        ) : (
          <div className="grid content-start gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {visible.map(agent => (
              <RowButton
                aria-label={`${t.simplifiedAgents.settings}: ${agent.label} · ${agent.gateway}`}
                className="group flex min-w-0 items-center gap-4 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-card) p-5 text-left hover:bg-(--chrome-action-hover)"
                key={agent.id}
                onClick={() => {
                  const params = new URLSearchParams({ agentProfile: agent.profile })

                  if (agent.connectionId) {
                    params.set('connection', agent.connectionId)
                  }

                  navigate(`${SETTINGS_ROUTE}?${params}`)
                }}
              >
                <FigmaAgentIcon color={resolveProfileColor(agent.profile, colors)} variant="subtle" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-base font-medium text-foreground">{agent.label}</span>
                  <span className="mt-1 block truncate text-xs text-muted-foreground">{agent.gateway}</span>
                </span>
                <Codicon className="text-muted-foreground group-hover:text-foreground" name="chevron-right" />
              </RowButton>
            ))}
          </div>
        )}
      </div>
    </Panel>
  )
}
