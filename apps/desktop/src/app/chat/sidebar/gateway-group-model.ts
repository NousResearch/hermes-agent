import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'

import type { DesktopConnectionsRegistry } from '@/global'
import type { SessionInfo } from '@/hermes'
import { resolveProfileColor } from '@/lib/profile-color'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $profileColors, $profiles, $profilesByConnection, normalizeProfileKey, profileLabel } from '@/store/profile'
import { $connection } from '@/store/session'
import type { ProfileInfo } from '@/types/hermes'

import type { SidebarSessionGroup } from './projects/workspace-groups'

/** Presentation-only name for an owner group's profile. */
export type OwnerProfileLabeler = (connectionId: null | string, profile: string) => string

const slugLabel: OwnerProfileLabeler = (_connectionId, profile) => profile

interface OwnerProfileLists {
  /** The active backend's list (`$profiles`). */
  active: ProfileInfo[]
  /** The id of the registry connection the active backend serves, if any. */
  activeConnectionId: null | string
  /** Lists cached per source (`$profilesByConnection`). */
  byConnection: ReadonlyMap<string, ProfileInfo[]>
  registry: DesktopConnectionsRegistry | null
}

/** Names each group from the profile list served by the group's OWN backend,
 *  so a renamed profile (`display_name`, e.g. a renamed `default`) reads the
 *  same here as in the rail. Bare rows and rows tagged with the registry
 *  primary belong to the primary, so they read the active list only while the
 *  primary is what's active. A group whose backend list isn't known keeps its
 *  key rather than borrowing a same-named profile's name from another backend. */
export function ownerProfileLabeler({
  active,
  activeConnectionId,
  byConnection,
  registry
}: OwnerProfileLists): OwnerProfileLabeler {
  const primary = registry?.primary?.trim() || null
  const activeId = activeConnectionId?.trim() || null
  const primaryList = !activeId || activeId === primary ? active : primary ? byConnection.get(primary) : undefined

  return (connectionId, profile) => {
    const list = !connectionId || connectionId === primary ? primaryList : byConnection.get(connectionId)
    const match = list?.find(entry => normalizeProfileKey(entry.name) === profile)

    return match ? profileLabel(match) : profile
  }
}

/** One group per exact owner `[connectionId, profile]`. Group identity never
 *  depends on a mutable label, URL, or the active gateway. */
export function buildGatewaySessionGroups(
  sessions: SessionInfo[],
  registry: DesktopConnectionsRegistry | null,
  colors: Record<string, string>,
  labelFor: OwnerProfileLabeler = slugLabel
): SidebarSessionGroup[] {
  const groups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    const profile = normalizeProfileKey(session.profile)
    const connectionId = session.connection_id || null
    const id = JSON.stringify([connectionId, profile])
    let group = groups.get(id)

    if (!group) {
      const gateway = registry?.connections.find(connection => connection.id === connectionId)
      const profileName = labelFor(connectionId, profile)

      group = {
        id,
        label: connectionId ? `${gateway?.label || connectionId} · ${profileName}` : profileName,
        connectionId,
        profile,
        profileLabel: profileName,
        mode: 'profile',
        path: null,
        color: resolveProfileColor(profile, colors),
        sessions: []
      }
      groups.set(id, group)
    }

    group.sessions.push(session)
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
    label: mixed ? group.label : (group.profileLabel ?? group.profile!)
  }))
}

function useOwnerProfileLabeler(): OwnerProfileLabeler {
  const registry = useStore($connectionsRegistry)
  const active = useStore($profiles)
  const byConnection = useStore($profilesByConnection)
  const activeConnectionId = useStoreSelector($connection, connection => connection?.connectionId ?? null)

  return useMemo(
    () => ownerProfileLabeler({ active, activeConnectionId, byConnection, registry }),
    [active, activeConnectionId, byConnection, registry]
  )
}

/** Recents and every messaging platform build owner groups with the same
 *  registry, colors and profile names. */
export function useOwnerGroupBuilder(): (sessions: SessionInfo[]) => SidebarSessionGroup[] {
  const registry = useStore($connectionsRegistry)
  const colors = useStore($profileColors)
  const labelFor = useOwnerProfileLabeler()

  return useCallback(
    (sessions: SessionInfo[]) => buildGatewaySessionGroups(sessions, registry, colors, labelFor),
    [registry, colors, labelFor]
  )
}

export function useGatewaySessionGroups(sessions: SessionInfo[], enabled: boolean) {
  const build = useOwnerGroupBuilder()

  return useMemo(() => (enabled ? build(sessions) : undefined), [sessions, enabled, build])
}
