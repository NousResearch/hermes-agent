import { useStore } from '@nanostores/react'
import type { ReactNode, RefObject } from 'react'

import { Button } from '@/components/ui/button'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { $hiddenSidebarAgents, agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $activeConnectionId } from '@/store/connections'
import { recordIaAction } from '@/store/ia-metrics'
import { $activeGatewayProfile } from '@/store/profile'
import { knownSessionOwner } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'

import { tileOwnerRoute } from '../../session-tile-owner'
import { SimplifiedBotChatRow } from '../simplified-bot-chat-row'
import { SimplifiedGatewayHeader } from '../simplified-gateway-header'
import { useAgentSidebarAttention } from '../use-agent-sidebar-attention'
import type { SimplifiedAgent, useSimplifiedAgents } from '../use-simplified-agents'

import { PortalAgentMenu } from './portal-agent-menu'
import { PortalHiddenAgents } from './portal-hidden-agents'

interface PortalAgentGroupsProps {
  agents: ReturnType<typeof useSimplifiedAgents>
  allSessions: SessionInfo[]
  expanded: Record<string, boolean>
  onToggle: (id: string, open: boolean) => void
  renderSessions: (sessions: SessionInfo[], agent: SimplifiedAgent) => ReactNode
  query: string
  pending: string | null
  onOpenAgent: (agent: SimplifiedAgent) => void
  onResumeSession: (id: string, row?: SessionInfo) => void
  scrollRef: RefObject<HTMLDivElement | null>
}

interface AgentCardProps extends Pick<PortalAgentGroupsProps, 'onToggle' | 'onOpenAgent' | 'pending'> {
  agent: SimplifiedAgent
  selected: boolean
  open: boolean
  hidden: boolean
  children: ReactNode
}

const agentKey = (agent: SimplifiedAgent) =>
  agentSidebarKey({ connectionId: agent.connectionId, profile: agent.targetProfile || agent.profile })

function AgentCard({ agent, selected, open, hidden, onToggle, onOpenAgent, pending, children }: AgentCardProps) {
  const { t } = useI18n()
  const toggleLabel = `${open ? t.common.collapse : t.common.expand}: ${agent.label}`

  return (
    <div
      aria-current={selected ? 'true' : undefined}
      aria-label={`${agent.label} · ${agent.gateway}`}
      data-slot="agent-group"
      role="group"
    >
      <div data-active-agent={selected ? '' : undefined} data-slot="agent-heading">
        <SimplifiedBotChatRow
          agent={agent}
          avatarColor={selected ? 'var(--portal-sidebar-agent-selected-ink)' : undefined}
        />
        <Tip label={toggleLabel}>
          <Button
            aria-expanded={open}
            aria-label={toggleLabel}
            className="portal-agent-collapse"
            onClick={() => {
              recordIaAction(open ? 'ia.section.agent_collapsed' : 'ia.section.agent_expanded')
              onToggle(agentKey(agent), open)
            }}
            size="icon-xs"
            variant="ghost"
          >
            <ChevronDown aria-hidden />
          </Button>
        </Tip>
        <PortalAgentMenu agent={agent} hidden={hidden} onOpen={() => onOpenAgent(agent)} pending={pending !== null} />
      </div>
      {open && children}
    </div>
  )
}

export function PortalAgentGroups(props: PortalAgentGroupsProps) {
  const hidden = new Set(useStore($hiddenSidebarAgents))
  const activeConnection = useStore($activeConnectionId)
  const activeProfile = useStore($activeGatewayProfile)
  const focused = useStore($focusedStoredSessionId)
  const tiles = useStore($sessionTiles)
  const { agents, sessionsByAgent, owner } = props.agents
  const attention = useAgentSidebarAttention(agents, props.allSessions)
  const query = props.query.trim().toLowerCase()

  const viewedOwner = focused
    ? (tileOwnerRoute(tiles, props.allSessions, focused) ?? knownSessionOwner(props.allSessions, focused))
    : undefined

  const viewedProfile =
    typeof viewedOwner === 'string' ? viewedOwner : viewedOwner?.targetProfile || viewedOwner?.profile || activeProfile

  const viewedConnection =
    viewedOwner && typeof viewedOwner === 'object' ? viewedOwner.connectionId : (activeConnection ?? owner)

  const activeAgentKey = agentSidebarKey({ connectionId: viewedConnection, profile: viewedProfile })

  const isHidden = (agent: SimplifiedAgent) => hidden.has(agentKey(agent))

  const hiddenAgents = agents.filter(isHidden)
  const groups = new Map<string | null, SimplifiedAgent[]>()

  for (const agent of agents) {
    const group = groups.get(agent.connectionId) ?? []
    const matches = (sessionsByAgent.get(agent.id)?.length ?? 0) > 0 || agent.label.toLowerCase().includes(query)

    if (query ? matches : !isHidden(agent)) {
      group.push(agent)
    }

    groups.set(agent.connectionId, group)
  }

  return (
    <>
      <FadeScroll className="portal-section-scroll" maxHeight="none" ref={props.scrollRef}>
        {[...groups].map(([connectionId, group]) => {
          if (query && !group.length) {
            return null
          }

          const gatewayId = JSON.stringify(['gateway', connectionId])
          const open = Boolean(query) || (props.expanded[gatewayId] ?? true)
          const selected = connectionId === (activeConnection ?? owner)
          const label = agents.find(agent => agent.connectionId === connectionId)!.gateway

          return (
            <div data-open={open && group.length > 0} data-slot="gateway-group" key={gatewayId}>
              <SimplifiedGatewayHeader
                connectionId={connectionId}
                label={label}
                onToggle={() => props.onToggle(gatewayId, open)}
                open={open}
                selected={selected}
              />
              {open && group.length > 0 && (
                <div className="portal-machine-agents">
                  {group.map(agent => (
                    <AgentCard
                      agent={agent}
                      hidden={isHidden(agent)}
                      key={agent.id}
                      onOpenAgent={props.onOpenAgent}
                      onToggle={props.onToggle}
                      open={Boolean(query) || (props.expanded[agentKey(agent)] ?? true)}
                      pending={props.pending}
                      selected={agentKey(agent) === activeAgentKey}
                    >
                      {props.renderSessions(sessionsByAgent.get(agent.id) ?? [], agent)}
                    </AgentCard>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </FadeScroll>
      {!query && (
        <PortalHiddenAgents agents={hiddenAgents} attention={attention} onResumeSession={props.onResumeSession} />
      )}
    </>
  )
}
