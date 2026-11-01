import type { HermesConnection } from '@/global'

import { isActivePrimary } from './gateway'
import { $connection } from './session'
import { $focusedSessionIsTile, $focusedStoredSessionId } from './session-focus'
import type { SessionOwnerScope } from './session-request-router'
import { knownOwnerForSession, ownerConnectionMode } from './session-states'

/** The backend identity captured when a terminal tab is created. */
export interface WorkspaceExecutionOwner {
  connectionId: string
  endpoint?: string | null
  mode: 'local' | 'remote'
  profile: string
}

interface ExecutionConnection extends HermesConnection {
  // Already returned by native SSH bootstrap; unavailable from older hosts.
  ssh?: { effectiveConfigFingerprint?: string; remoteHermesPath?: string; remoteProfile?: string }
}

export function parseWorkspaceExecutionOwner(value: unknown): WorkspaceExecutionOwner | null | undefined {
  if (value === undefined) {
    return undefined
  }

  if (!value || typeof value !== 'object') {
    return null
  }

  const owner = value as Partial<WorkspaceExecutionOwner>

  if (typeof owner.connectionId !== 'string' || typeof owner.profile !== 'string' ||
      (owner.mode !== 'local' && owner.mode !== 'remote')) {
    return null
  }

  return {
    connectionId: owner.connectionId,
    profile: owner.profile,
    mode: owner.mode,
    ...(owner.endpoint !== undefined ? { endpoint: typeof owner.endpoint === 'string' ? owner.endpoint : null } : {})
  }
}

function activeOwner(connection: ExecutionConnection | null): WorkspaceExecutionOwner | null {
  if (!connection || connection.mode === 'local') {
    return {
      connectionId: connection?.connectionId || 'local',
      mode: 'local',
      profile: connection?.profile?.trim() || 'default'
    }
  }

  // Legacy SSH/URL primary connections predate registry ids. Their stable
  // remote identity lets a saved terminal stay attached to that same host;
  // without it, guessing would send a path to whichever host is active now.
  const identity = connection.connectionId?.trim() || legacyRemoteIdentity(connection)

  if (!identity) {
    return null
  }

  return {
    connectionId: identity,
    endpoint: legacyRemoteIdentity(connection) || null,
    mode: 'remote',
    profile: connection.profile?.trim() || 'default'
  }
}

function legacyRemoteIdentity(connection: ExecutionConnection): string {
  if (connection.remoteKind === 'ssh') {
    // `remoteHost` is a display label, not enough to distinguish two Hermes
    // homes on the same SSH host. Use the stable SSH config fingerprint.
    const ssh = connection.ssh

    return ssh?.effectiveConfigFingerprint
      ? JSON.stringify(['ssh', ssh.effectiveConfigFingerprint, ssh.remoteHermesPath || '', ssh.remoteProfile || 'default'])
      : ''
  }

  if (connection.remoteKind === 'url' || connection.remoteKind === 'cloud') {
    try {
      const url = new URL(connection.baseUrl)
      url.username = ''
      url.password = ''
      url.search = ''
      url.hash = ''

      return `url:${url.toString().replace(/\/$/, '')}`
    } catch {
      return ''
    }
  }

  // A remote host label can collapse several URL path prefixes / backends to
  // the same value. Missing stable identity means the owner is unknown.
  return ''
}

function ownerForRoute(owner: SessionOwnerScope): WorkspaceExecutionOwner | null {
  if (!owner || typeof owner === 'string') {
    return null
  }

  const connectionId = owner.connectionId.trim()

  if (!connectionId) {
    return null
  }

  return {
    connectionId,
    ...(ownerConnectionMode(owner) === 'local' ? {} : {
      endpoint: activeWorkspaceExecutionOwner()?.connectionId === connectionId
        ? activeWorkspaceExecutionOwner()?.endpoint ?? null
        : null
    }),
    mode: ownerConnectionMode(owner) ?? owner.mode ?? 'remote',
    profile: owner.profile.trim() || 'default'
  }
}

function sameOwner(left: WorkspaceExecutionOwner | null, right: WorkspaceExecutionOwner | null): boolean {
  return Boolean(
    left &&
      right &&
      left.mode === right.mode &&
      left.connectionId === right.connectionId &&
      left.profile === right.profile &&
      (left.mode === 'local' || Boolean(left.endpoint && left.endpoint === right.endpoint))
  )
}

export function activeWorkspaceExecutionOwner(): WorkspaceExecutionOwner | null {
  return activeOwner($connection.get())
}

export function workspaceExecutionOwnerForSession(storedSessionId: string): WorkspaceExecutionOwner | null {
  const owner = knownOwnerForSession(storedSessionId)
  const exact = ownerForRoute(owner)

  if (exact) {
    return exact
  }

  if (typeof owner === 'string' && owner.trim()) {
    const mode = ownerConnectionMode(owner)

    if (mode === 'local') {
      return { connectionId: 'local', mode, profile: owner.trim() }
    }

    // A legacy profile-only remote owner is resolved through the primary
    // connection. `knownOwnerForSession` has already preferred an exact tile
    // route and persisted hint, so this fallback names only that primary.
    const active = activeWorkspaceExecutionOwner()

    if (mode === 'remote' && active?.mode === 'remote' && isActivePrimary() && active.profile === owner.trim()) {
      return { ...active, profile: owner.trim() }
    }
  }

  // A missing/ambiguous owner is not proof that a remote path belongs to the
  // active gateway. It must never inherit whichever connection is selected.
  return null
}

export function focusedWorkspaceExecutionOwner(): WorkspaceExecutionOwner | null {
  if (!$focusedSessionIsTile.get()) {
    return activeWorkspaceExecutionOwner()
  }

  const focused = $focusedStoredSessionId.get()

  return focused ? workspaceExecutionOwnerForSession(focused) : null
}

export function workspaceExecutionOwnerMatchesActiveBackend(
  owner: WorkspaceExecutionOwner | null | undefined
): boolean {
  const active = activeWorkspaceExecutionOwner()

  if (owner === null) {
    // An explicitly unresolved focused owner must not inherit the selected
    // backend, even when the selected backend happens to be local.
    return false
  }

  // Old pre-feature terminals had no owner stamp. Their local PTY remains on
  // this machine; a remote saved shell cannot safely be assigned to a host by
  // its restore key alone.
  if (owner === undefined) {
    return active?.mode === 'local'
  }

  return sameOwner(owner, active)
}
