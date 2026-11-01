import { atom } from 'nanostores'

import { $sessionControlBySession } from '@/store/session-control'
import type { SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession, requestForOwnedSession } from '@/store/session-states'

/** `runtime.describe.drain`: an owner finishing its turns before `hermes update` replaces it. */
export interface GatewayUpdateDrain {
  reason: 'update'
  target?: null | string
  waiting_for?: Record<string, number>
}

type Requester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
type Described = { drain?: GatewayUpdateDrain | null } | null | undefined

/** The update each gateway owner is draining for, keyed by {@link drainOwnerKey}. Two chats on two
 *  gateways never share a cause: each is read from, and stored under, the owner that refused. */
export const $gatewayUpdateDrains = atom<Record<string, GatewayUpdateDrain>>({})

/** How long a known update may keep the gateway away before the chip falls back to "offline". */
export const UPDATE_WAIT_MS = 15 * 60_000

let ambient: null | Requester = null
const inFlight = new Set<string>()
const generations = new Map<string, number>()

/** One owner identity: a registry route (connection + served profile), a bare profile, or ambient. */
export function drainOwnerKey(owner: SessionOwnerScope): string {
  if (!owner) {
    return 'ambient'
  }

  if (typeof owner === 'string') {
    return `profile:${owner}`
  }

  return `route:${owner.connectionId}\u0000${owner.targetProfile ?? owner.profile}`
}

export function gatewayUpdateDrainFor(drains: Record<string, GatewayUpdateDrain>, key: string): GatewayUpdateDrain | null {
  return drains[key] ?? null
}

function store(key: string, drain: GatewayUpdateDrain | null): void {
  const rest = { ...$gatewayUpdateDrains.get() }
  delete rest[key]
  $gatewayUpdateDrains.set(drain ? { ...rest, [key]: drain } : rest)
}

/** Forget *key*'s cause; a describe still in flight for it can no longer bring it back. */
export function clearGatewayUpdateDrain(key: string): void {
  generations.set(key, (generations.get(key) ?? 0) + 1)
  store(key, null)
}

/** Ask the owner behind *key* why it is refusing work. A closed socket keeps the last cause: the
 *  owner that drained is gone, but the update it drained for is not finished until a new one answers. */
export async function refreshGatewayUpdateDrain(key: string, describe: () => Promise<Described>): Promise<void> {
  if (inFlight.has(key)) {
    return
  }

  inFlight.add(key)
  const generation = generations.get(key) ?? 0

  try {
    const described = await describe()

    if ((generations.get(key) ?? 0) === generation) {
      store(key, described?.drain?.reason === 'update' ? described.drain : null)
    }
  } catch {
    // The owner went away mid-question: the reconnect decides what is true next.
  } finally {
    inFlight.delete(key)
  }
}

/** Ask the owner of *sessionId* on that owner's own socket, never the focused tile's. */
export function refreshSessionOwnerDrain(sessionId: string): Promise<void> {
  const request = ambient

  if (!request) {
    return Promise.resolve()
  }

  return refreshGatewayUpdateDrain(drainOwnerKey(knownOwnerForSession(sessionId)), () =>
    requestForOwnedSession<Described>(sessionId, request, 'runtime.describe')
  )
}

/** The drain of the gateway that owns *sessionId*, or null. */
export function sessionUpdateDrain(
  drains: Record<string, GatewayUpdateDrain>,
  sessionId: string
): GatewayUpdateDrain | null {
  return gatewayUpdateDrainFor(drains, drainOwnerKey(knownOwnerForSession(sessionId)))
}

/** The drain's live work units ("finishing N turns"); every kind counts as a turn to finish. */
export function updateDrainTurns(drain: GatewayUpdateDrain): number {
  return Object.values(drain.waiting_for ?? {}).reduce((sum, n) => sum + (Number.isInteger(n) && n > 0 ? n : 0), 0)
}

/** Wire the ambient requester and ask each refusing session's own owner why it refused. */
export function bindGatewayUpdateDrain(request: Requester): () => void {
  ambient = request

  const stop = $sessionControlBySession.listen(entries => {
    for (const [sessionId, entry] of Object.entries(entries)) {
      if (entry?.error === 'runtime_draining') {
        void refreshSessionOwnerDrain(sessionId)
      }
    }
  })

  return () => {
    stop()

    if (ambient === request) {
      ambient = null
    }
  }
}
