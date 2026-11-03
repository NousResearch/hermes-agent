import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'
import { useNavigate } from 'react-router'

import { SETTINGS_ROUTE } from '@/app/routes'
import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { recordAction } from '@/store/desktop-metrics'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profileColors, ensureGatewayProfile } from '@/store/profile'
import { setSettingsScope } from '@/store/settings-scope'

import { SimplifiedGatewayHeader } from './simplified-gateway-header'
import { startSimplifiedAgentSession } from './simplified-new-session-model'
import { useFleetRoster } from './use-fleet-roster'
import type { useSimplifiedAgents } from './use-simplified-agents'

export function SimplifiedAgentList({
  agents,
  expanded,
  onToggle,
  renderSessions,
  searching = false
}: {
  searching?: boolean
  agents: ReturnType<typeof useSimplifiedAgents>
  expanded: Record<string, boolean>
  onToggle: (id: string, open: boolean) => void
  renderSessions: (sessions: SessionInfo[]) => ReactNode
}) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const colors = useStore($profileColors)
  const registry = useStore($connectionsRegistry)
  const activeConnection = useStore($activeConnectionId)
  const [pending, setPending] = useState<string | null>(null)
  useFleetRoster(true)
  const { agents: sorted, owner, sessionsByAgent } = agents

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
  const isOpen = (id: string) => searching || (expanded[id] ?? true)

  return (
    <section aria-label={t.simplifiedAgents.agents} className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
      <SidebarPanelLabel className="px-2 py-2 [&>span:first-child]:mx-1">{t.simplifiedAgents.agents}</SidebarPanelLabel>
      {gateways.map(connectionId => {
        const group = sorted.filter(agent => agent.connectionId === connectionId)
        const gatewayId = JSON.stringify(['gateway', connectionId])
        const open = gateways.length === 1 || isOpen(gatewayId)
        const selected = connectionId === (activeConnection ?? owner)

        return (
          <div className="shrink-0" key={gatewayId}>
            <SimplifiedGatewayHeader
              connectionId={connectionId}
              fallbackLabel={group[0].gateway}
              onToggle={gateways.length > 1 ? () => onToggle(gatewayId, open) : undefined}
              open={open}
              selected={selected}
            />
            {open &&
              group.map(agent => {
                const agentOpen = group.length === 1 || isOpen(agent.id)
                const toggleLabel = `${agentOpen ? t.common.collapse : t.common.expand}: ${agent.label}`

                return (
                  <div aria-label={`${agent.label} · ${agent.gateway}`} key={agent.id} role="group">
                    <div className="group/agent grid min-h-8.5 grid-cols-[1.5rem_minmax(0,1fr)_1.5rem] items-center gap-1 px-2">
                      {group.length > 1 ? (
                        <Tip label={toggleLabel}>
                          <Button
                            aria-expanded={agentOpen}
                            aria-label={toggleLabel}
                            onClick={() => onToggle(agent.id, agentOpen)}
                            size="icon-xs"
                            variant="grip"
                          >
                            <Codicon name={agentOpen ? 'chevron-down' : 'chevron-right'} size={12} />
                          </Button>
                        </Tip>
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
                      <Tip label={t.sidebar.newSessionIn(agent.label)}>
                        <Button
                          aria-label={t.sidebar.newSessionIn(agent.label)}
                          className="opacity-0 group-hover/agent:opacity-100 group-focus-within/agent:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
                          onClick={() => {
                            recordAction('session.new', 'click')
                            startSimplifiedAgentSession(agent)
                          }}
                          size="icon-xs"
                          variant="ghost"
                        >
                          <Codicon name="add" size={14} />
                        </Button>
                      </Tip>
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
