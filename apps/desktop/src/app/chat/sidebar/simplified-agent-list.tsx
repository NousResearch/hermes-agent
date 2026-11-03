import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { SETTINGS_ROUTE } from '@/app/routes'
import { settingsScopeLabel } from '@/app/settings/profile-scope'
import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profileColors, $profiles, ensureGatewayProfile } from '@/store/profile'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { setSettingsScope } from '@/store/settings-scope'

import { gatewayLabel } from './gateway-group-model'
import { useFleetRoster } from './use-fleet-roster'

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
  const profiles = useStore($profiles)
  const colors = useStore($profileColors)
  const roster = useStore($fleetRoster)
  const registry = useStore($connectionsRegistry)
  const activeConnection = useStore($activeConnectionId)
  const apiScope = useStore($apiRequestScope)
  const focusedId = useStore($focusedStoredSessionId)
  const activeProfile = useStore($activeGatewayProfile)
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  useEffect(() => setExpanded({}), [focusedId, activeProfile, activeConnection])
  const [pending, setPending] = useState<string | null>(null)
  useFleetRoster(true)
  const owner = apiScope.connectionId ?? roster?.primaryConnectionId ?? activeConnection ?? registry?.primary ?? null

  const agents = new Map<
    string,
    { id: string; profile: string; label: string; connectionId: string | null; gateway: string }
  >()

  for (const agent of roster?.agents ?? []) {
    const id = JSON.stringify([agent.connectionId, agent.profile])
    agents.set(id, {
      id,
      profile: agent.profile,
      label: agent.profile,
      connectionId: agent.connectionId,
      gateway: agent.connectionLabel
    })
  }

  for (const profile of profiles) {
    const id = JSON.stringify([owner, profile.name])
    agents.set(id, {
      id,
      profile: profile.name,
      label: settingsScopeLabel(profile),
      connectionId: owner,
      gateway:
        roster?.sources.find(source => source.connectionId === owner)?.label ??
        gatewayLabel(owner ?? 'local', registry, t.settings.gateway.webappHostTitle)
    })
  }

  const sessionsByAgent = new Map<string, SessionInfo[]>()

  for (const session of sessions) {
    const connectionId = session.connection_id || owner
    const profile = session.profile || 'default'
    const id = JSON.stringify([connectionId, profile])
    const rows = sessionsByAgent.get(id) ?? []
    rows.push(session)
    sessionsByAgent.set(id, rows)

    // Keep chats reachable while the profile roster is loading or incomplete.
    if (!agents.has(id)) {
      agents.set(id, {
        id,
        profile,
        label: profile,
        connectionId,
        gateway:
          roster?.sources.find(source => source.connectionId === connectionId)?.label ??
          gatewayLabel(connectionId ?? 'local', registry, t.settings.gateway.webappHostTitle)
      })
    }
  }

  const sorted = [...agents.values()].sort(
    (a, b) =>
      a.gateway.localeCompare(b.gateway) ||
      (a.connectionId ?? '').localeCompare(b.connectionId ?? '') ||
      a.label.localeCompare(b.label) ||
      a.id.localeCompare(b.id)
  )

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
      const params = new URLSearchParams({ agent: agent.profile })

      if (agent.connectionId) {params.set('connection', agent.connectionId)}
      navigate(`${SETTINGS_ROUTE}?${params}`)
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
      <SidebarPanelLabel>{t.simplifiedAgents.agents}</SidebarPanelLabel>
      {gateways.map(connectionId => {
        const group = sorted.filter(agent => agent.connectionId === connectionId)
        const gatewayId = JSON.stringify(['gateway', connectionId])
        const open = gateways.length === 1 || isOpen(gatewayId, connectionId === owner)
        const reachable = roster?.sources.find(source => source.connectionId === connectionId)?.reachable

        return (
          <div className="shrink-0" key={gatewayId}>
            <Button
              aria-expanded={open}
              className="w-full justify-start text-xs text-(--ui-text-tertiary)"
              onClick={() => toggle(gatewayId, open)}
              variant="ghost"
            >
              {gateways.length > 1 && <Codicon name={open ? 'chevron-down' : 'chevron-right'} />}
              {reachable !== undefined && (
                <span
                  aria-hidden
                  className={`size-1.5 shrink-0 rounded-full ${reachable ? 'bg-green-500' : 'bg-(--ui-text-tertiary)'}`}
                />
              )}
              <span className="truncate">{group[0].gateway}</span>
            </Button>
            {open &&
              group.map(agent => {
                const agentOpen = group.length === 1 || isOpen(agent.id, agent.id === focusedAgent)

                return (
                  <div aria-label={`${agent.label} · ${agent.gateway}`} key={agent.id} role="group">
                    <div className="flex items-center">
                      {group.length > 1 && (
                        <Button
                          aria-expanded={agentOpen}
                          aria-label={agent.label}
                          onClick={() => toggle(agent.id, agentOpen)}
                          size="icon"
                          variant="ghost"
                        >
                          <Codicon name={agentOpen ? 'chevron-down' : 'chevron-right'} />
                        </Button>
                      )}
                      <Button
                        aria-label={`${t.simplifiedAgents.settings}: ${agent.label} · ${agent.gateway}`}
                        className="min-w-0 flex-1 justify-start"
                        disabled={pending !== null}
                        onClick={() => void openAgent(agent)}
                        variant="ghost"
                      >
                        <ProfileGlyph
                          color={resolveProfileColor(agent.profile, colors)}
                          isDefault={agent.profile === 'default'}
                          name={agent.label}
                        />
                        <span className="truncate">{agent.label}</span>
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
