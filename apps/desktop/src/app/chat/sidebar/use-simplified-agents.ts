import { useStore } from '@nanostores/react'

import { $apiRequestScope } from '@/api/client'
import { settingsScopeLabel } from '@/app/settings/profile-scope'
import type { DesktopConnectionKind } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'

import { gatewayLabel } from './gateway-group-model'

export interface SimplifiedAgent {
  id: string
  profile: string
  label: string
  connectionId: string | null
  gateway: string
  connectionKind?: DesktopConnectionKind
  targetProfile?: string
}

export function useSimplifiedOwner() {
  const registry = useStore($connectionsRegistry)
  const roster = useStore($fleetRoster)
  const activeConnection = useStore($activeConnectionId)
  const apiScope = useStore($apiRequestScope)

  return apiScope.connectionId ?? roster?.primaryConnectionId ?? activeConnection ?? registry?.primary ?? null
}

export function useSimplifiedAgents(sessions: SessionInfo[]) {
  const { t } = useI18n()
  const profiles = useStore($profiles)
  const roster = useStore($fleetRoster)
  const registry = useStore($connectionsRegistry)
  const owner = useSimplifiedOwner()

  const agents = new Map<string, SimplifiedAgent>()

  for (const agent of roster?.agents ?? []) {
    const id = JSON.stringify([agent.connectionId, agent.profile])
    agents.set(id, {
      id,
      profile: agent.profile,
      label: agent.profile,
      connectionId: agent.connectionId,
      gateway: agent.connectionLabel,
      connectionKind: agent.connectionKind,
      targetProfile: agent.targetProfile
    })
  }

  for (const profile of profiles) {
    const id = JSON.stringify([owner, profile.name])
    agents.set(id, {
      id,
      profile: profile.name,
      label: settingsScopeLabel(profile),
      connectionKind:
        agents.get(id)?.connectionKind ??
        roster?.sources.find(source => source.connectionId === owner)?.kind ??
        registry?.connections.find(connection => connection.id === owner)?.kind,
      targetProfile: agents.get(id)?.targetProfile,
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

  return { agents: sorted, owner, sessionsByAgent }
}
