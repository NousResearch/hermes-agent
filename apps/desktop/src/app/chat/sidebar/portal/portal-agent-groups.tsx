import { useStore } from '@nanostores/react'
import type { CSSProperties, ReactNode, RefObject } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $hiddenSidebarAgents, agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $activeConnectionId } from '@/store/connections'
import { recordIaAction } from '@/store/ia-metrics'
import { $activeGatewayProfile, $profileColors } from '@/store/profile'
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

function AgentCard({ agent, selected, open, hidden, onToggle, onOpenAgent, pending, children }: AgentCardProps) {
  const { t } = useI18n()
  const colors = useStore($profileColors)
  const toggleLabel = `${open ? t.common.collapse : t.common.expand}: ${agent.label}`

  return (
    <div
      aria-current={selected ? 'true' : undefined}
      aria-label={`${agent.label} · ${agent.gateway}`}
      data-slot="agent-group"
      role="group"
      style={{ '--agent-color': resolveProfileColor(agent.profile, colors) } as CSSProperties}
    >
      <div data-active-agent={selected ? '' : undefined} data-slot="agent-heading">
        <SimplifiedBotChatRow agent={agent} />
        <PortalAgentMenu agent={agent} hidden={hidden} onOpen={() => onOpenAgent(agent)} pending={pending !== null} />
        <Tip label={toggleLabel}>
          <Button
            aria-expanded={open}
            aria-label={toggleLabel}
            className="portal-agent-collapse"
            onClick={() => {
              recordIaAction(open ? 'ia.section.agent_collapsed' : 'ia.section.agent_expanded')
              onToggle(
                agentSidebarKey({ connectionId: agent.connectionId, profile: agent.targetProfile || agent.profile }),
                open
              )
            }}
            size="icon-xs"
            variant="grip"
          >
            <Codicon name={open ? 'chevron-down' : 'chevron-right'} size={12} />
          </Button>
        </Tip>
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

  const isHidden = (agent: SimplifiedAgent) =>
    hidden.has(agentSidebarKey({ connectionId: agent.connectionId, profile: agent.targetProfile || agent.profile }))

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
      <FadeScroll className="min-h-0 flex-1" fade="min(2.5rem, 25%)" maxHeight="none" ref={props.scrollRef}>
        {[...groups].map(([connectionId, group]) => {
          if (query && !group.length) {
            return null
          }

          const gatewayId = JSON.stringify(['gateway', connectionId])
          const open = Boolean(query) || (props.expanded[gatewayId] ?? true)
          const selected = connectionId === (activeConnection ?? owner)
          const fallbackLabel = agents.find(agent => agent.connectionId === connectionId)!.gateway

          return (
            <div data-slot="gateway-group" key={gatewayId}>
              <SimplifiedGatewayHeader
                connectionId={connectionId}
                fallbackLabel={fallbackLabel}
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
                      open={
                        Boolean(query) ||
                        (props.expanded[
                          agentSidebarKey({
                            connectionId: agent.connectionId,
                            profile: agent.targetProfile || agent.profile
                          })
                        ] ??
                          true)
                      }
                      pending={props.pending}
                      selected={
                        agentSidebarKey({
                          connectionId: agent.connectionId,
                          profile: agent.targetProfile || agent.profile
                        }) === activeAgentKey
                      }
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
