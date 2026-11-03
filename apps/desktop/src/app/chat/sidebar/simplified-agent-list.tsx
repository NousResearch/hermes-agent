import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

import { SETTINGS_ROUTE } from '@/app/routes'
import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { RowButton } from '@/components/ui/row-button'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { recordAction } from '@/store/desktop-metrics'
import { $fleetRoster } from '@/store/fleet-roster'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profileColors, ensureGatewayProfile } from '@/store/profile'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { setSettingsScope } from '@/store/settings-scope'

import { startSimplifiedAgentSession } from './simplified-new-session-model'
import { useFleetRoster } from './use-fleet-roster'
import { useSimplifiedAgents } from './use-simplified-agents'

export function SimplifiedAgentList({
  sessions,
  renderSessions,
  searching = false
}: {
  searching?: boolean
  sessions: SessionInfo[]
  renderSessions: (sessions: SessionInfo[]) => ReactNode
}) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const colors = useStore($profileColors)
  const roster = useStore($fleetRoster)
  const registry = useStore($connectionsRegistry)
  const activeConnection = useStore($activeConnectionId)
  const focusedId = useStore($focusedStoredSessionId)
  const activeProfile = useStore($activeGatewayProfile)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  useEffect(() => setExpanded({}), [focusedId, activeProfile, activeConnection])
  const [pending, setPending] = useState<string | null>(null)
  useFleetRoster(true)
  const { agents: sorted, owner, sessionsByAgent } = useSimplifiedAgents(sessions)

  async function openAgent(agent: (typeof sorted)[number]) {
    setPending(agent.id)

    try {
      if (agent.connectionId && registry?.connections.some(connection => connection.id === agent.connectionId)) {
        await selectConnection(agent.connectionId, { profile: agent.profile })

        if ($activeConnectionId.get() !== agent.connectionId || $activeGatewayProfile.get() !== agent.profile) {
          return
        }
      } else {
        if (agent.connectionId !== owner) {
          throw new Error(t.profiles.fleet.gatewayUnreachable(agent.gateway))
        }

        await ensureGatewayProfile(agent.profile)
      }

      setSettingsScope(agent.profile)
      navigate(`${SETTINGS_ROUTE}?tab=config:model`)
    } catch (error) {
      notifyError(error, t.errors.genericFailure)
    } finally {
      setPending(null)
    }
  }

  const gateways = [...new Set(sorted.map(agent => agent.connectionId))]
  const focusedAgent = focusedId ? JSON.stringify([owner, activeProfile || 'default']) : null
  const isOpen = (id: string, focused: boolean) => searching || (expanded[id] ?? (!focusedAgent || focused))
  const toggle = (id: string, open: boolean) => setExpanded(current => ({ ...current, [id]: !open }))

  return (
    <section aria-label={t.simplifiedAgents.agents} className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
      <SidebarPanelLabel className="px-2 py-2 [&>span:first-child]:mx-1">{t.simplifiedAgents.agents}</SidebarPanelLabel>
      {gateways.map(connectionId => {
        const group = sorted.filter(agent => agent.connectionId === connectionId)
        const gatewayId = JSON.stringify(['gateway', connectionId])
        const open = gateways.length === 1 || isOpen(gatewayId, connectionId === owner)
        const reachable = roster?.sources.find(source => source.connectionId === connectionId)?.reachable

        return (
          <div className="shrink-0" key={gatewayId}>
            <div className="grid min-h-8.5 grid-cols-[1rem_minmax(0,1fr)_1.5rem] items-center gap-2 px-2 text-xs text-(--ui-text-tertiary)">
              <span aria-hidden className="grid size-4 place-items-center">
                <span className={`size-1.5 rounded-full ${reachable ? 'bg-green-500' : 'bg-(--ui-text-quaternary)'}`} />
              </span>
              <span className="truncate" title={group[0].gateway}>
                {group[0].gateway}
              </span>
              {gateways.length > 1 && (
                <Button
                  aria-expanded={open}
                  aria-label={group[0].gateway}
                  onClick={() => toggle(gatewayId, open)}
                  size="icon-xs"
                  variant="grip"
                >
                  <Codicon name={open ? 'chevron-down' : 'chevron-right'} size={12} />
                </Button>
              )}
            </div>
            {open &&
              group.map(agent => {
                const agentOpen = group.length === 1 || isOpen(agent.id, agent.id === focusedAgent)

                return (
                  <div aria-label={`${agent.label} · ${agent.gateway}`} key={agent.id} role="group">
                    <div className="group/agent grid min-h-8.5 grid-cols-[1.5rem_minmax(0,1fr)_1.5rem] items-center gap-1 px-2">
                      {group.length > 1 ? (
                        <Button
                          aria-expanded={agentOpen}
                          aria-label={agent.label}
                          onClick={() => toggle(agent.id, agentOpen)}
                          size="icon-xs"
                          variant="grip"
                        >
                          <Codicon name={agentOpen ? 'chevron-down' : 'chevron-right'} size={12} />
                        </Button>
                      ) : (
                        <span aria-hidden />
                      )}
                      <RowButton
                        aria-label={`${t.simplifiedAgents.settings}: ${agent.label} · ${agent.gateway}`}
                        className="flex min-h-7.5 min-w-0 items-center gap-2 rounded-sm px-2 text-left text-xs font-medium hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50"
                        disabled={pending !== null}
                        onClick={() => void openAgent(agent)}
                      >
                        <ProfileGlyph
                          color={resolveProfileColor(agent.profile, colors)}
                          isDefault={agent.profile === 'default'}
                          name={agent.label}
                        />
                        <span className="truncate">{agent.label}</span>
                      </RowButton>
                      <Button
                        aria-label={t.sidebar.newSessionIn(agent.label)}
                        className="opacity-0 group-hover/agent:opacity-100 group-focus-within/agent:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
                        onClick={() => {
                          recordAction('session.new', 'click')
                          startSimplifiedAgentSession(agent)
                        }}
                        size="icon-xs"
                        title={t.sidebar.newSessionIn(agent.label)}
                        variant="ghost"
                      >
                        <Codicon name="add" size={14} />
                      </Button>
                    </div>
                    {agentOpen && renderSessions(sessionsByAgent.get(agent.id) ?? [])}
                  </div>
                )
              })}
          </div>
        )
      })}
    </section>
  )
}
