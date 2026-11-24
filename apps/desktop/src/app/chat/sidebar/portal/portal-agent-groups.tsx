import { useStore } from '@nanostores/react'
import type { ReactNode, RefObject } from 'react'

import type { AgentSettingsOwner } from '@/app/agent-settings/data'
import { SIDEBAR_PROFILE_GROUP_HEADER_AREA } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { Tip } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { ChevronDown, Plus } from '@/lib/icons'
import { $hiddenSidebarAgents, agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $activeConnectionId } from '@/store/connections'
import { recordAction } from '@/store/desktop-metrics'
import { recordIaAction } from '@/store/ia-metrics'
import { $activeGatewayProfile } from '@/store/profile'
import { $sessionProfilesTruncated, $sessionProfilesUsage, knownSessionOwner } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'

import { tileOwnerRoute } from '../../session-tile-owner'
import { ProfileGroupHeaderSlot } from '../gateway-groups'
import { useHiddenAgentCopy } from '../hidden-agent-copy'
import { SimplifiedBotChatRow, useBotModeLoaded } from '../simplified-bot-chat-row'
import { SimplifiedGatewayHeader } from '../simplified-gateway-header'
import { startSimplifiedAgentSession } from '../simplified-new-session-model'
import { agentKey, type SimplifiedAgent, type useSimplifiedAgents } from '../use-simplified-agents'

import { type AgentChatCount, agentChatCount } from './agent-chat-cap'
import type { NewAgentOffer } from './new-agent-offers'
import { PortalAgentMenu } from './portal-agent-menu'
import { AgentScreenRow } from './portal-agent-screen'
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
  /** New agent for each machine listed, and what starts one. */
  newAgent: NewAgentOffer[]
  onNewAgent: (owner: AgentSettingsOwner) => void
  onResumeSession: (id: string, row?: SessionInfo) => void
  scrollRef: RefObject<HTMLDivElement | null>
}

interface AgentCardProps extends Pick<PortalAgentGroupsProps, 'onToggle' | 'onOpenAgent' | 'pending'> {
  agent: SimplifiedAgent
  selected: boolean
  open: boolean
  /** A search holds the card open: its disclosure stands down and records nothing. */
  forcedOpen: boolean
  hidden: boolean
  /** How many chats the agent has, as far as the list knows: the card's second line when no description. */
  chats: AgentChatCount
  /** Whether any of them is in the list under the card: only then is there something to fold. */
  listed: boolean
  children: ReactNode
}

/** An agent block inside its machine (1250:37611): colour tile, name over a second line, then the
 *  menu and a new chat (hover/focus affordances) and the disclosure, with its chats underneath. */
function AgentCard({
  agent,
  selected,
  open,
  forcedOpen,
  hidden,
  chats,
  listed,
  onToggle,
  onOpenAgent,
  pending,
  children
}: AgentCardProps) {
  const { t } = useI18n()
  const c = useHiddenAgentCopy()
  const toggleLabel = `${open ? t.common.collapse : t.common.expand}: ${agent.label}`

  const summary = chats.more
    ? t.simplifiedAgents.chatCountAtLeast(chats.value)
    : chats.value
      ? t.simplifiedAgents.chatCount(chats.value)
      : t.simplifiedAgents.noChats

  const headerItems = useContributions(SIDEBAR_PROFILE_GROUP_HEADER_AREA)
  // The disclosure is offered only when folding would hide something: chats or plugin chrome.
  const foldable = listed || headerItems.length > 0

  return (
    <div
      aria-current={selected ? 'true' : undefined}
      aria-label={`${agent.label} · ${agent.gateway}`}
      data-slot="agent-group"
      role="group"
    >
      <div data-active-agent={selected ? '' : undefined} data-slot="agent-heading">
        <SimplifiedBotChatRow agent={agent} fallbackSummary={summary} />
        <PortalAgentMenu agent={agent} hidden={hidden} onOpen={() => onOpenAgent(agent)} pending={pending !== null} />
        <Tip label={c.newChat}>
          <Button
            aria-label={`${c.newChat}: ${agent.label}`}
            className="portal-agent-new"
            onClick={() => {
              recordAction('session.new', 'click')
              recordIaAction('ia.new_session.agent_row')
              startSimplifiedAgentSession(agent)
            }}
            size="icon-xs"
            variant="ghost"
          >
            <Plus aria-hidden />
          </Button>
        </Tip>
        {foldable && (
          <Tip label={toggleLabel}>
            <Button
              aria-expanded={open}
              aria-label={toggleLabel}
              className="portal-agent-collapse"
              disabled={forcedOpen}
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
        )}
      </div>
      {open && (
        <ProfileGroupHeaderSlot
          className="portal-agent-header-slot"
          connectionId={agent.connectionId}
          profile={agent.profile}
        />
      )}
      {open && children}
    </div>
  )
}

/** Whether a search keeps an agent listed: by its name, or by a chat of its that matched. */
export const agentMatches = (list: PortalAgentGroupsProps['agents'], agent: SimplifiedAgent, query: string) =>
  (list.sessionsByAgent.get(agentKey(agent))?.length ?? 0) > 0 || agent.label.toLowerCase().includes(query.trim().toLowerCase())

export function PortalAgentGroups(props: PortalAgentGroupsProps) {
  const botMode = useBotModeLoaded()
  const hidden = new Set(useStore($hiddenSidebarAgents))
  const activeConnection = useStore($activeConnectionId)
  const activeProfile = useStore($activeGatewayProfile)
  const focused = useStore($focusedStoredSessionId)
  const tiles = useStore($sessionTiles)
  const truncated = useStore($sessionProfilesTruncated)
  const usage = useStore($sessionProfilesUsage)
  const { agents, sessionsByAgent, owner } = props.agents
  const query = props.query.trim().toLowerCase()
  const chatsOf = (agent: SimplifiedAgent) => sessionsByAgent.get(agentKey(agent)) ?? []

  // The page flags and counts are keyed by profile alone and come from the machine the list was read
  // from: they say nothing about a same-named profile on another machine.
  const chatCountOf = (agent: SimplifiedAgent) => {
    const profile = agent.targetProfile || agent.profile
    const listed = agent.connectionId === owner

    return agentChatCount(
      chatsOf(agent).length,
      listed && Boolean(truncated[profile]),
      listed ? usage[profile]?.sessions : undefined
    )
  }

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

    if (query ? agentMatches(props.agents, agent, query) : !isHidden(agent)) {
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
          const machine = props.newAgent.find(machine => machine.connectionId === connectionId)!

          return (
            <div data-open={open && group.length > 0} data-slot="gateway-group" key={gatewayId}>
              <SimplifiedGatewayHeader
                machine={machine}
                onNewAgent={props.onNewAgent}
                onToggle={query ? undefined : () => props.onToggle(gatewayId, open)}
                open={open}
              />
              {open && group.length > 0 && (
                <div className="portal-machine-agents">
                  {group.map(agent => (
                    <AgentCard
                      agent={agent}
                      chats={chatCountOf(agent)}
                      forcedOpen={Boolean(query)}
                      hidden={isHidden(agent)}
                      key={agent.id}
                      listed={chatsOf(agent).length > 0}
                      onOpenAgent={props.onOpenAgent}
                      onToggle={props.onToggle}
                      open={Boolean(query) || (props.expanded[agentKey(agent)] ?? true)}
                      pending={props.pending}
                      selected={agentKey(agent) === activeAgentKey}
                    >
                      {/* Each row reads its agent's screen once: only for the machine the list was read from. */}
                      {botMode && agent.connectionId === owner && <AgentScreenRow agent={agent} />}
                      {props.renderSessions(chatsOf(agent), agent)}
                    </AgentCard>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </FadeScroll>
      {/* Attention is read only here, and only while some agent is hidden. */}
      {!query && hiddenAgents.length > 0 && (
        <PortalHiddenAgents
          agents={hiddenAgents}
          allAgents={agents}
          allSessions={props.allSessions}
          onResumeSession={props.onResumeSession}
        />
      )}
    </>
  )
}
