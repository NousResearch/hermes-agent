import type { MutableRefObject } from 'react'

import { capabilityScoped, type ResolvedOwner } from '@/api/client'
import { activeGatewayConnectionId, gatewayActivationEpoch, requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile, type AgentProfileRoute, normalizeProfileKey } from '@/store/profile'
import { $freshDraftKey, $freshDraftReady } from '@/store/session'
import { profileScopeForSessionOwner } from '@/store/session-request-router'
import { releaseSessionOwnerHold } from '@/store/session-states'

import { isNewChatRoute } from '../../../routes'

import type { SessionCreateOverrides } from './create-overrides'

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>
type VoiceDraft = NonNullable<SessionCreateOverrides['voiceDraft']>

/**
 * A New Chat session created as a GPT-Live call's destination. The call may bind
 * to it only while its draft is still the one on screen (same route, nothing
 * selected) and, on the ambient path, while the active owner has not changed:
 * otherwise the call's transcript would land in a chat the user is not looking at.
 */
export class VoiceDraftCreate {
  private owner: ResolvedOwner & { profile: string } = { connectionId: null, profile: '' }
  private dispatchEpoch: number | undefined
  private route: AgentProfileRoute | null = null

  constructor(
    private readonly draft: VoiceDraft,
    private readonly startingRouteToken: string,
    private readonly routeToken: () => string,
    private readonly selectedStoredId: () => null | string
  ) {}

  /** The draft this create started from is still the one on screen. */
  owns(): boolean {
    return (
      this.draft.isCurrent() &&
      isNewChatRoute(this.startingRouteToken.split(':', 1)[0]) &&
      $freshDraftReady.get() &&
      $freshDraftKey.get() === this.draft.key &&
      this.routeToken() === this.startingRouteToken &&
      !this.selectedStoredId()
    )
  }

  /** Pin the owner. On the ambient path, read the actual active route after profile
   *  readiness, not the window's REST scope before it was activated. */
  bind(route: AgentProfileRoute | null, profile: string): void {
    this.route = route
    this.owner = { connectionId: route ? route.connectionId : activeGatewayConnectionId(), profile: route?.profile ?? profile }
    this.dispatchEpoch = route ? undefined : gatewayActivationEpoch()
  }

  /** An ambient create can lose its activation race without throwing: the active
   *  owner moved, so the session it minted is not the call's. */
  ownerChanged(): boolean {
    return (
      this.dispatchEpoch !== undefined &&
      (gatewayActivationEpoch() !== this.dispatchEpoch ||
        activeGatewayConnectionId() !== this.owner.connectionId ||
        normalizeProfileKey($activeGatewayProfile.get()) !== this.owner.profile)
    )
  }

  lost(): boolean {
    return !this.owns() || this.ownerChanged()
  }

  /** The ambient `session.create` transport: compatibility retries must not follow
   *  a newly active owner either. */
  guard(requestGateway: RequestGateway): RequestGateway {
    if (this.dispatchEpoch === undefined) {
      return requestGateway
    }

    return <T>(method: string, params?: Record<string, unknown>) => {
      if (this.lost()) {
        throw new Error('The fresh chat destination changed before session creation.')
      }

      return requestGateway<T>(method, params)
    }
  }

  /** Hand the created chat to the call. */
  assign(runtimeSessionId: string, storedSessionId: string): void {
    const restProfile = this.route && capabilityScoped(profileScopeForSessionOwner(this.route)).profile

    this.draft.onAssigned({
      backendProfile: restProfile || this.owner.profile,
      owner: this.owner,
      runtimeSessionId,
      storedSessionId
    })
  }

  /** One voice create at a time (`flight` holds its intent): while it runs no other
   *  create starts, and it never starts beside another create. */
  static blocked(flight: MutableRefObject<null | string>, voice: null | VoiceDraftCreate, creating: boolean): boolean {
    return Boolean(flight.current) || Boolean(voice && (creating || !voice.owns()))
  }

  claim(flight: MutableRefObject<null | string>, intent: string): void {
    flight.current = intent
  }

  release(flight: MutableRefObject<null | string>, intent: string): void {
    if (flight.current === intent) {
      flight.current = null
    }
  }
}

export function voiceDraftCreate(
  draft: undefined | VoiceDraft,
  startingRouteToken: string,
  routeToken: () => string,
  selectedStoredId: () => null | string
): null | VoiceDraftCreate {
  return draft ? new VoiceDraftCreate(draft, startingRouteToken, routeToken, selectedStoredId) : null
}

export function guardedCreateTransport(voice: null | VoiceDraftCreate, requestGateway: RequestGateway): RequestGateway {
  return voice ? voice.guard(requestGateway) : requestGateway
}

/**
 * Whether a just-minted session no longer applies to this create: the user moved to
 * a different chat mid-create (drift), or a voice draft lost its destination. Decided
 * synchronously, in the same turn as the caller's selection and hand-off.
 */
export function createdSessionAbandoned(drift: unknown, storedSessionId: null | string, voice: null | VoiceDraftCreate): boolean {
  return Boolean(drift) || Boolean(voice && (!storedSessionId || voice.lost()))
}

interface AbandonedSession {
  drift: unknown
  /** The active owner moved during an ambient create (read when abandoning). */
  ownerLost: boolean
  requestGateway: RequestGateway
  route: AgentProfileRoute | null
  runtimeSessionId: string
  storedSessionId: null | string
}

/** Close an abandoned session on the backend that minted it. */
export async function closeAbandonedSession({
  drift,
  ownerLost,
  requestGateway,
  route,
  runtimeSessionId,
  storedSessionId
}: AbandonedSession): Promise<void> {
  if (drift) {
    console.warn('[submit-drift-abort]', drift, { phase: 'mid-create' })
  }

  // The ambient socket is a different machine/profile for a routed create and would
  // 4001 while the orphan lives on (and later ws-orphan-reaps) there. After an ambient
  // owner change, never send the orphan's runtime id to the new owner.
  const closeCreated = route
    ? requestGatewayForAgent(route.connectionId, route.profile, 'session.close', { session_id: runtimeSessionId })
    : ownerLost
      ? undefined
      : requestGateway('session.close', { session_id: runtimeSessionId })

  await closeCreated?.catch(() => undefined)

  if (storedSessionId) {
    releaseSessionOwnerHold(storedSessionId)
  }
}
