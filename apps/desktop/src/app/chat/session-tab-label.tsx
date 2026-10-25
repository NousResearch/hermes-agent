import { useStore } from '@nanostores/react'

import { $workspaceOwnerLabels } from '@/components/pane-shell/workspace-scope'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles, $profilesByConnection, profileLabel } from '@/store/profile'
import { $sessionTabAgentNames } from '@/store/session-tab-agent-names'
import type { SessionInfo } from '@/types/hermes'

interface SessionTabOwnerScope {
  workspaceOwnerKey?: string
}

interface SessionTabOwnerSources {
  profiles: ReturnType<typeof $profiles.get>
  profilesByConnection: ReturnType<typeof $profilesByConnection.get>
  roster: ReturnType<typeof $fleetRoster.get>
  workspaceOwnerLabels: ReturnType<typeof $workspaceOwnerLabels.get>
}

const currentSources = (): SessionTabOwnerSources => ({
  profiles: $profiles.get(),
  profilesByConnection: $profilesByConnection.get(),
  roster: $fleetRoster.get(),
  workspaceOwnerLabels: $workspaceOwnerLabels.get()
})

/** Presentation identity for a session owner. Routing remains keyed by the
 * canonical (connection, profile) pair; this only turns it into tab chrome. */
export function sessionTabOwnerLabel(
  session: Pick<SessionInfo, 'connection_id' | 'profile'> | null | undefined,
  scope?: SessionTabOwnerScope,
  sources: SessionTabOwnerSources = currentSources()
): string | null {
  if (!session) {
    return null
  }

  const profile = (session.profile ?? '').trim()
  const connectionId = (session.connection_id ?? '').trim()

  const profileInfo = (connectionId ? sources.profilesByConnection.get(connectionId) : sources.profiles)?.find(
    candidate => candidate.name === (profile || 'default')
  )

  const ownerLabel = scope?.workspaceOwnerKey
    ? (sources.workspaceOwnerLabels[scope.workspaceOwnerKey] ?? '').trim()
    : ''

  const friendly = (profileInfo?.bot_title ?? '').trim() || (profileInfo ? profileLabel(profileInfo) : '') || ownerLabel

  const rosterAgent = sources.roster?.agents.find(
    candidate => candidate.connectionId === connectionId && candidate.profile === (profile || 'default')
  )

  const label = friendly || rosterAgent?.profile || profile

  if (!label) {
    return null
  }

  // The same profile slug can exist on several registered sources. Keep those
  // tabs distinguishable without replacing a friendly agent name with the
  // roster's technical @profile-device handle.
  const duplicateAcrossConnections = Boolean(
    rosterAgent &&
    sources.roster?.agents.some(
      candidate => candidate.profile === rosterAgent.profile && candidate.connectionId !== rosterAgent.connectionId
    )
  )

  return duplicateAcrossConnections && rosterAgent ? `${label} (${rosterAgent.connectionLabel})` : label
}

export function sessionTabText(owner: string | null | undefined, title: string): string {
  if (!$sessionTabAgentNames.get()) {
    return title
  }

  const cleanOwner = owner?.trim()
  const cleanTitle = title.trim()

  if (!cleanOwner) {
    return cleanTitle
  }

  if (!cleanTitle) {
    return cleanOwner
  }

  return `${cleanOwner} · ${cleanTitle}`
}

export function SessionTabLabel({
  scope,
  session,
  title
}: {
  scope?: SessionTabOwnerScope
  session: Pick<SessionInfo, 'connection_id' | 'profile'> | null | undefined
  title: string
}) {
  const enabled = useStore($sessionTabAgentNames)
  const profiles = useStore($profiles)
  const profilesByConnection = useStore($profilesByConnection)
  const roster = useStore($fleetRoster)
  const workspaceOwnerLabels = useStore($workspaceOwnerLabels)

  const owner = sessionTabOwnerLabel(session, scope, {
    profiles,
    profilesByConnection,
    roster,
    workspaceOwnerLabels
  })

  if (!enabled || !owner) {
    return title
  }

  return (
    <span
      aria-label={sessionTabText(owner, title)}
      className="flex min-w-0 items-center gap-1 normal-case tracking-normal"
    >
      <span aria-hidden="true" className="shrink-0 font-bold">
        {owner}
      </span>
      <span aria-hidden="true" className="shrink-0 text-(--ui-text-quaternary)">
        ·
      </span>
      <span aria-hidden="true" className="min-w-0 truncate text-inherit">
        {title}
      </span>
    </span>
  )
}
