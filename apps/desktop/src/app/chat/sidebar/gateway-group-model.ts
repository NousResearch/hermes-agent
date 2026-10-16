import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import type { DesktopConnectionsRegistry } from '@/global'
import type { SessionInfo } from '@/hermes'
import { resolveProfileColor } from '@/lib/profile-color'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $profileColors, normalizeProfileKey } from '@/store/profile'

import type { SidebarSessionGroup } from './projects/workspace-groups'

/** One group per exact owner `[connectionId, profile]`. Group identity never
 *  depends on a mutable label, URL, or the active gateway. */
export function buildGatewaySessionGroups(
  sessions: SessionInfo[],
  registry: DesktopConnectionsRegistry | null,
  colors: Record<string, string>
): SidebarSessionGroup[] {
  const groups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    const profile = normalizeProfileKey(session.profile)
    const connectionId = session.connection_id || null
    const id = JSON.stringify([connectionId, profile])
    const gateway = registry?.connections.find(connection => connection.id === connectionId)
    const label = connectionId ? `${gateway?.label || connectionId} · ${profile}` : profile

    const group: SidebarSessionGroup = groups.get(id) ?? {
      id,
      label,
      connectionId,
      profile,
      mode: 'profile',
      path: null,
      color: resolveProfileColor(profile, colors),
      sessions: []
    }

    group.sessions.push(session)
    groups.set(id, group)
  }

  return [...groups.values()].sort((a, b) => a.label.localeCompare(b.label) || a.id.localeCompare(b.id))
}

/** Re-key owner groups shown inside another section (a messaging platform) so
 *  their collapse/alias/order preferences never touch the recents groups. No
 *  gateway header sits above them there, so the gateway stays in the label only
 *  when the section mixes gateways. */
export function scopeGatewaySessionGroups(groups: SidebarSessionGroup[], scope: string): SidebarSessionGroup[] {
  const mixed = new Set(groups.map(group => group.connectionId)).size > 1

  return groups.map(group => ({
    ...group,
    id: JSON.stringify([scope, group.id]),
    label: mixed ? group.label : group.profile!
  }))
}

export function useGatewaySessionGroups(sessions: SessionInfo[], enabled: boolean) {
  const registry = useStore($connectionsRegistry)
  const colors = useStore($profileColors)

  return useMemo(
    () => (enabled ? buildGatewaySessionGroups(sessions, registry, colors) : undefined),
    [sessions, enabled, registry, colors]
  )
}
