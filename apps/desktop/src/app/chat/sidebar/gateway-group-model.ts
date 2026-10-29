import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { $apiRequestScope } from '@/api/client'
import type { DesktopConnectionsRegistry } from '@/global'
import type { SessionInfo } from '@/hermes'
import { isBrowserHostedDesktop } from '@/lib/platform'
import { resolveProfileColor } from '@/lib/profile-color'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $sidebarShowArchived } from '@/store/layout'
import { $profileColors, $profiles, normalizeProfileKey } from '@/store/profile'
import { $sessionsLoadError } from '@/store/session'

import type { SidebarSessionGroup } from './projects/workspace-groups'

/** A gateway section's name: its registry label, else `localLabel` for the `local` source a
 *  registry does not list (the Webapp's one gateway, the host serving it), else the raw id. */
export function gatewayLabel(
  connectionId: string,
  registry: DesktopConnectionsRegistry | null,
  localLabel: string
): string {
  const registered = registry?.connections.find(connection => connection.id === connectionId)?.label

  return registered || (connectionId === LOCAL_CONNECTION_ID ? localLabel : connectionId)
}

/** Whether a profile group header shows the profile-keyed usage totals. Those totals are keyed only
 *  by profile, so never attribute them to a registry gateway that happens to expose the same profile
 *  name, or to one messaging platform (`embedded`). A Webapp tab's `local` rows are the serving
 *  host's own, the gateway those totals were read from. */
export function showsProfileTotals(connectionId: string | null | undefined, embedded: boolean): boolean {
  if (embedded) {
    return false
  }

  return !connectionId || (connectionId === LOCAL_CONNECTION_ID && isBrowserHostedDesktop())
}

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
    const group = groups.get(id) ?? ownerGroup(connectionId, profile, registry, colors)

    group.sessions.push(session)
    groups.set(id, group)
  }

  return sortGroups([...groups.values()])
}

function ownerGroup(
  connectionId: null | string,
  profile: string,
  registry: DesktopConnectionsRegistry | null,
  colors: Record<string, string>
): SidebarSessionGroup {
  const gateway = registry?.connections.find(connection => connection.id === connectionId)

  return {
    id: JSON.stringify([connectionId, profile]),
    label: connectionId ? `${gateway?.label || connectionId} · ${profile}` : profile,
    connectionId,
    profile,
    mode: 'profile',
    path: null,
    color: resolveProfileColor(profile, colors),
    sessions: []
  }
}

function sortGroups(groups: SidebarSessionGroup[]): SidebarSessionGroup[] {
  return groups.sort((a, b) => a.label.localeCompare(b.label) || a.id.localeCompare(b.id))
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

/** The owner id the profile list belongs to, in the same vocabulary rows are tagged with: a
 *  non-local request connection stamps its rows with its id; the primary pool stays untagged
 *  natively, and a Webapp tab tags every row with `local` (its one gateway). `local` is that
 *  untagged pool only while it is the aggregate base (the registry primary): as a secondary beside
 *  a remote base its rows carry the `local` tag, and the untagged pool is the remote's. */
export function profileListConnectionId(
  apiConnectionId: null | string,
  browserHosted: boolean,
  localIsPrimary: boolean
): null | string {
  const owner = apiConnectionId?.trim()

  if (owner && (owner !== LOCAL_CONNECTION_ID || !localIsPrimary)) {
    return owner
  }

  return browserHosted ? LOCAL_CONNECTION_ID : null
}

/** Give every listed profile its own section, chats or not, so a new or quiet profile stays one
 *  click away. Groups already present (by exact owner) keep their rows; missing ones are added
 *  empty under `connectionId`, the owner the profile list was read from. */
export function withEmptyProfileGroups(
  groups: SidebarSessionGroup[],
  profiles: readonly string[],
  connectionId: null | string,
  registry: DesktopConnectionsRegistry | null,
  colors: Record<string, string>
): SidebarSessionGroup[] {
  const present = new Set(groups.map(group => group.id))

  const missing = profiles
    .map(normalizeProfileKey)
    .filter((profile, index, all) => all.indexOf(profile) === index)
    .filter(profile => !present.has(JSON.stringify([connectionId, profile])))
    .map(profile => ownerGroup(connectionId, profile, registry, colors))

  return missing.length ? sortGroups([...groups, ...missing]) : groups
}

/** Owner groups for the recents list. Every listed profile keeps its section even with no chats,
 *  unless the list is a narrowed view (filters, archive) or failed to load. */
export function useGatewaySessionGroups(sessions: SessionInfo[], enabled: boolean, narrowed = true) {
  const registry = useStore($connectionsRegistry)
  const colors = useStore($profileColors)
  const profiles = useStore($profiles)
  const { connectionId: apiConnectionId } = useStore($apiRequestScope)
  // The untagged pool is the registry primary's: Electron reads it as the aggregate's base. The
  // window primary's own mode speaks for it only while the window primary is that source; a
  // window booted on another registered route (the saved default route) leaves the base where it
  // is. Never the active gateway's ($connection): a foregrounded local secondary reports `local`
  // beside a remote primary. Unknown keeps local as the primary pool.
  const primary = useStore($primaryGatewayRoute)
  const windowIsBase = !registry || !primary.connectionId || primary.connectionId === registry.primary
  const localIsPrimary = windowIsBase ? primary.mode !== 'remote' : registry.primary === LOCAL_CONNECTION_ID
  const showArchived = useStore($sidebarShowArchived)
  const loadError = useStore($sessionsLoadError)
  const includeEmptyProfiles = !narrowed && !showArchived && !loadError

  return useMemo(() => {
    if (!enabled) {
      return undefined
    }

    const groups = buildGatewaySessionGroups(sessions, registry, colors)

    if (!includeEmptyProfiles) {
      return groups
    }

    const owner = profileListConnectionId(apiConnectionId, isBrowserHostedDesktop(), localIsPrimary)

    return withEmptyProfileGroups(
      groups,
      profiles.map(profile => profile.name),
      owner,
      registry,
      colors
    )
  }, [sessions, enabled, includeEmptyProfiles, registry, colors, profiles, apiConnectionId, localIsPrimary])
}
