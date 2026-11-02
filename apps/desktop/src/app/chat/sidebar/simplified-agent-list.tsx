import { useStore } from '@nanostores/react'
import { type ReactNode, useState } from 'react'
import { useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { SETTINGS_ROUTE } from '@/app/routes'
import { settingsScopeLabel } from '@/app/settings/profile-scope'
import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Button } from '@/components/ui/button'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { resolveProfileColor } from '@/lib/profile-color'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profileColors, $profiles, ensureGatewayProfile } from '@/store/profile'
import { setSettingsScope } from '@/store/settings-scope'

import { gatewayLabel } from './gateway-group-model'
import { useFleetRoster } from './use-fleet-roster'

export function SimplifiedAgentList({
  sessions,
  renderSessions
}: {
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
      navigate(`${SETTINGS_ROUTE}?tab=config:model`)
    } catch (error) {
      notifyError(error, t.errors.genericFailure)
    } finally {
      setPending(null)
    }
  }

  return (
    <section aria-label={t.simplifiedAgents.agents} className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
      <SidebarPanelLabel>{t.simplifiedAgents.agents}</SidebarPanelLabel>
      {sorted.map((agent, index) => (
        <div key={agent.id} role="group" aria-label={`${agent.label} · ${agent.gateway}`} className="shrink-0">
          {(index === 0 || sorted[index - 1].connectionId !== agent.connectionId) && (
            <p className="px-2 pt-2 text-xs text-(--ui-text-tertiary)">{agent.gateway}</p>
          )}
          <Button
            aria-label={`${t.simplifiedAgents.settings}: ${agent.label} · ${agent.gateway}`}
            className="w-full justify-start"
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
          {renderSessions(sessionsByAgent.get(agent.id) ?? [])}
        </div>
      ))}
    </section>
  )
}
