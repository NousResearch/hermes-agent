/** Exact-session delivery for Desktop plugins: one stored session, one route.
 *
 * The sequence's lifetimes are invisible in the RPC shapes — an unheld route
 * socket is reaped between the calls, and a submit released at its ACK detaches
 * the runtime mid-turn — so the order and both release paths live here. */
import type { PluginProfileRoute } from './index'

export interface PluginSessionSubmitInput {
  /** Durable session id as stored on the backend — never a live runtime id. */
  storedSessionId: string
  text: string
}

/** The backend's typed `prompt.submit` status. `null` on the one reply that
 *  carries none (the typed stop phrase). */
export type PluginSessionSubmitStatus = 'streaming' | 'queued' | 'steered' | 'redirected' | null

export interface PluginSessionSubmitResult {
  /** Live runtime id `session.resume` returned. Ephemeral: it does not outlive
   *  the backend process, unlike `storedSessionId`. */
  runtimeSessionId: string
  status: PluginSessionSubmitStatus
}

/** `prompt.submit`'s reply: a typed status once a turn starts, or the
 *  typed-stop-phrase acknowledgement (`voice_stopped`), which starts none. */
interface PluginPromptSubmitReply {
  status?: PluginSessionSubmitStatus
  voice_stopped?: boolean | null
}

/** The SDK primitives the sequence composes. */
export interface PluginSessionDeliveryDeps {
  /** One JSON-RPC call on the exact route — `host.requestProfile`. */
  request: <T>(route: PluginProfileRoute | string, method: string, params: Record<string, unknown>) => Promise<T>
  /** Hold a route's pooled socket across the sequence —
   *  `store/gateway::retainGatewayForAgent`. A null connectionId is the legacy
   *  local/sole-source route. */
  retainRoute: (connectionId: null | string, profile: string) => Promise<() => void>
  /** Hold a routed runtime until its turn's terminal session event —
   *  `store/gateway::retainGatewayForSessionTurn`. */
  retainTurn: (connectionId: null | string, profile: string, runtimeSessionId: string) => Promise<() => void>
}

interface ResolvedPluginRoute {
  connectionId: null | string
  profile: string
  targetProfile: string
}

/** Translate the caller's route into the three identities the sequence needs.
 *  A descriptor stays connection-qualified; a bare profile is the legacy
 *  local-only overload. Null means unusable, refused before dialing. */
function resolvePluginRoute(route: PluginProfileRoute | string): null | ResolvedPluginRoute {
  if (typeof route === 'string') {
    const profile = route.trim()

    return profile ? { connectionId: null, profile, targetProfile: profile } : null
  }

  const connectionId = route?.connectionId?.trim() ?? ''
  const profile = route?.profile?.trim() ?? ''
  const targetProfile = route?.targetProfile?.trim() ?? ''

  return connectionId && profile && targetProfile ? { connectionId, profile, targetProfile } : null
}

/** Submit one turn to one exact stored session on one exact route.
 *
 * Never retried automatically: a submit that timed out may already have been
 * accepted, so the caller reconciles the transcript instead of resending. */
export async function submitToPluginSession(
  deps: PluginSessionDeliveryDeps,
  route: PluginProfileRoute | string,
  input: PluginSessionSubmitInput
): Promise<PluginSessionSubmitResult> {
  const target = resolvePluginRoute(route)

  if (!target) {
    throw new Error('Profile route must include connectionId, profile, and targetProfile')
  }

  const storedSessionId = input?.storedSessionId?.trim() ?? ''

  if (!storedSessionId) {
    throw new Error('submitToSession requires a stored session id')
  }

  const text = input?.text ?? ''

  if (!text.trim()) {
    throw new Error('submitToSession requires non-empty text')
  }

  // Each requestProfile call is its own request lease: a secondary socket at
  // refcount 0 closes between the RPCs and the gateway reaps the runtime it
  // minted, failing the next call with 4001.
  const releaseRoute = await deps.retainRoute(target.connectionId, target.profile)

  try {
    const resumed = await deps.request<{ session_id?: string }>(route, 'session.resume', {
      session_id: storedSessionId,
      source: 'desktop',
      omit_messages: true,
      profile: target.targetProfile
    })

    const runtimeSessionId = resumed?.session_id?.trim() ?? ''

    if (!runtimeSessionId) {
      throw new Error('session.resume returned no runtime session id')
    }

    // prompt.submit ACKs when the turn STARTS, so the routed runtime needs its
    // own hold until the terminal session event releases it; the request lease
    // alone would detach it and the turn would be cut as client_gone.
    const releaseTurn = await deps.retainTurn(target.connectionId, target.profile, runtimeSessionId)

    try {
      const submitted = await deps.request<PluginPromptSubmitReply>(route, 'prompt.submit', {
        session_id: runtimeSessionId,
        text,
        // Fixed, never a caller option: a background delivery queues behind the
        // active turn and must never steer, redirect, or interrupt it.
        queued: true
      })

      // The typed stop phrase ends the voice chat and starts no turn, so no
      // terminal session event will ever release this hold.
      if (submitted?.voice_stopped) {
        releaseTurn()
      }

      return { runtimeSessionId, status: submitted?.status ?? null }
    } catch (error) {
      // A refused submit has no turn to keep alive; a timed-out one may already
      // have been accepted, so this releases the hold without resending.
      releaseTurn()
      throw error
    }
  } finally {
    releaseRoute()
  }
}
