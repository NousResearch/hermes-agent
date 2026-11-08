import type { SessionInfo } from '@/hermes'
import { isActivePrimary, openGatewayForAgent, openGatewayForProfile } from '@/store/gateway'
import { $showAllProfiles, ensureGatewayAgent, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $connection, setSessionOwnerHint } from '@/store/session'
import type { SessionOwnerScope, SessionProfileRoute } from '@/store/session-request-router'

/**
 * Connection that supplied the current session list, captured before resume's
 * async metadata lookup.
 *
 * A connection switch clears/reloads the session rows before this path runs,
 * so an untagged row belongs to the connection that supplied the current list.
 * If we reduce it to the profile string `default`, requestForSessionProfile
 * resolves the local default socket and sends an SSH session id to the wrong
 * machine ("resume failed: session not found").
 */
export function ambientResumeConnectionId(): string {
  const ambientConnection = $connection.get()

  // Keep the legacy primary-local profile door: main may resolve a named
  // profile to its own remote override (#94166). A registry secondary is
  // already an explicit source, even when it is This device under Home.
  return ambientConnection?.mode === 'remote' || (ambientConnection?.registryScoped && !isActivePrimary())
    ? ambientConnection.connectionId?.trim() || ''
    : ''
}

/**
 * Route a resumed session by the composite (connection, profile).
 *
 * A row spliced from a CONNECTED registry gateway (#88880) carries its owning
 * connection. A row fetched directly after activating a registry gateway can
 * be untagged, so retain the captured ambient connection too. Either way,
 * route by the composite (connection, profile), never by a same-named profile
 * alone.
 */
export function resolveResumeOwner(
  ownerRoute: SessionProfileRoute | undefined,
  storedForProfile: SessionInfo | undefined,
  ambientConnectionId: string
): { resolvedConnectionId: string; sessionOwner: SessionOwnerScope } {
  const sessionProfile = storedForProfile?.profile
  const resolvedConnectionId = ownerRoute?.connectionId || storedForProfile?.connection_id || ambientConnectionId

  const sessionOwner: SessionOwnerScope =
    ownerRoute ||
    (resolvedConnectionId
      ? {
          connectionId: resolvedConnectionId,
          profile: sessionProfile || 'default'
        }
      : sessionProfile)

  return { resolvedConnectionId, sessionOwner }
}

/**
 * Preserve this resolved source for later prompt/approval RPCs too; otherwise
 * an untagged row falls back to its bare profile after resume. Only for a row
 * the ambient source actually returned: an id that did not resolve (deep link,
 * routed restore) proves nothing about its owner, and a persisted hint would
 * pin it to whichever source was in front.
 */
export function rememberResolvedResumeOwner(
  storedSessionId: string,
  ownerRoute: SessionProfileRoute | undefined,
  storedForProfile: SessionInfo | undefined,
  sessionOwner: SessionOwnerScope
): void {
  if (
    !ownerRoute &&
    storedForProfile &&
    !storedForProfile.connection_id &&
    sessionOwner &&
    typeof sessionOwner === 'object'
  ) {
    setSessionOwnerHint(storedSessionId, sessionOwner)
  }
}

/**
 * Dial the backend that owns a resumed session.
 *
 * All-profiles / plugin navigation must not steal chrome API-home: dial the
 * owning backend without moving $activeGatewayProfile.
 */
export async function openResumeGateway(
  resolvedConnectionId: string,
  ownerRoute: SessionProfileRoute | undefined,
  sessionProfile: string | undefined
): Promise<void> {
  if ($showAllProfiles.get()) {
    if (resolvedConnectionId) {
      await openGatewayForAgent(resolvedConnectionId, ownerRoute?.profile || sessionProfile || 'default', {
        spawnPriority: 'foreground'
      })
    } else if (sessionProfile) {
      await openGatewayForProfile(normalizeProfileKey(sessionProfile), { spawnPriority: 'foreground' })
    }
  } else if (resolvedConnectionId) {
    await ensureGatewayAgent(resolvedConnectionId, ownerRoute?.profile || sessionProfile || 'default')
  } else {
    await ensureGatewayProfile(sessionProfile)
  }
}

/**
 * `session.resume` params for a primary-view resume.
 *
 * REST is the transcript authority for Desktop. Avoid duplicating a
 * potentially huge compression lineage in the WebSocket response. Watch
 * windows attach lazily (live mirror). Every other cold resume gets the
 * gateway's default deferred build: the RPC returns the transcript immediately
 * instead of blocking the switch on _make_agent (MCP discovery / prompt
 * build), and the agent pre-warms in the background while the prefetch paints
 * the transcript.
 */
export function primaryResumeParams(
  storedSessionId: string,
  watchWindow: boolean,
  authoritativeSnapshot: boolean | undefined,
  sessionProfile: string | undefined
): Record<string, unknown> {
  return {
    session_id: storedSessionId,
    cols: 96,
    source: 'desktop',
    defer_history: authoritativeSnapshot ? false : !watchWindow,
    ...(authoritativeSnapshot ? { omit_messages: false } : watchWindow ? { lazy: true } : { omit_messages: true }),
    ...(sessionProfile ? { profile: sessionProfile } : {})
  }
}
