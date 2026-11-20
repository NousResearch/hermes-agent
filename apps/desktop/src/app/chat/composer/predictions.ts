import { atom, computed } from 'nanostores'

import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'

export interface ComposerPredictionScope {
  sessionId: string
  profile: string
  connectionId: string
}

/** Complete next-message previews. Capture the scope BEFORE starting async work.
 * Unregistered connections, new drafts and tiles fail closed. */
export interface ComposerPredictionsPayload extends ComposerPredictionScope {
  suggestions: string[]
}

/** Opaque publication receipt: retain the exact returned object for cleanup.
 * It is not serializable and cannot be reconstructed from a session id. */
export interface ComposerPredictionHandle {
  readonly publication: symbol
}

export interface ComposerPredictions extends ComposerPredictionsPayload {
  index: number
  handle: ComposerPredictionHandle
}

// Reject oversized batches rather than silently truncating a valid message.
export const MAX_PREDICTION_CANDIDATES = 8
export const MAX_PREDICTION_TEXT_LENGTH = 16_384
const MAX_SCOPE_LENGTH = 512

export const $composerPredictions = atom<ComposerPredictions | null>(null)
export const $composerPredictionAuthority = atom(0)

interface Receiver {
  scope: ComposerPredictionScope
  canReceive: () => boolean
}
let receiver: Receiver | null = null

export function isComposerPredictionScope(value: unknown): value is ComposerPredictionScope {
  if (!value || typeof value !== 'object') {
    return false
  }

  return ['sessionId', 'profile', 'connectionId'].every(key => {
    const part = (value as Record<string, unknown>)[key]

    return typeof part === 'string' && part.length > 0 && part.length <= MAX_SCOPE_LENGTH && part.trim() === part
  })
}

function sameScope(a: ComposerPredictionScope, b: ComposerPredictionScope) {
  return a.sessionId === b.sessionId && a.profile === b.profile && a.connectionId === b.connectionId
}

function isActive(scope: ComposerPredictionScope) {
  return (
    scope.sessionId === $activeSessionId.get() &&
    scope.profile === $activeGatewayProfile.get() &&
    scope.connectionId === $connection.get()?.connectionId
  )
}

// Invalidate the receiver as well as its preview at the authoritative edge.
// Even A → B → A before React renders must not resurrect the old binding.
function invalidateAuthority() {
  receiver = null
  $composerPredictions.set(null)
  $composerPredictionAuthority.set($composerPredictionAuthority.get() + 1)
}

$activeSessionId.listen(invalidateAuthority)
$selectedStoredSessionId.listen(invalidateAuthority)
// Match host.state.profile and the socket's routing authority, not the
// separately refreshed HTTP/display profile, which may lag a route change.
$activeGatewayProfile.listen(invalidateAuthority)

// Window geometry also updates $connection; only routing changes revoke a
// receiver. A disconnect/reconnect or endpoint replacement is an authority edge.
const $predictionConnection = computed($connection, connection =>
  connection
    ? JSON.stringify([
        connection.connectionId,
        connection.baseUrl,
        connection.wsUrl,
        connection.profile,
        connection.mode
      ])
    : null
)

$predictionConnection.listen(invalidateAuthority)

export function clearComposerPredictions(handle: ComposerPredictionHandle): boolean {
  const current = $composerPredictions.get()

  if (!current || current.handle !== handle) {
    return false
  }

  $composerPredictions.set(null)

  return true
}

export function setComposerPredictions(payload: ComposerPredictionsPayload): ComposerPredictionHandle | null {
  if (!isComposerPredictionScope(payload) || !receiver || !sameScope(payload, receiver.scope) || !isActive(payload)) {
    return null
  }

  const { suggestions } = payload

  if (
    !Array.isArray(suggestions) ||
    !suggestions.length ||
    suggestions.length > MAX_PREDICTION_CANDIDATES ||
    suggestions.some(text => typeof text !== 'string' || text.length > MAX_PREDICTION_TEXT_LENGTH)
  ) {
    return null
  }

  // Preserve every character of accepted text; whitespace-only entries and
  // exact duplicates have no value, but are never permission to clear an owner.
  const cleaned = [...new Set(suggestions.filter(text => text.trim().length > 0))]

  if (!cleaned.length || !receiver.canReceive()) {
    return null
  }

  const handle: ComposerPredictionHandle = Object.freeze({ publication: Symbol('composer-prediction') })
  $composerPredictions.set({
    sessionId: payload.sessionId,
    profile: payload.profile,
    connectionId: payload.connectionId,
    suggestions: cleaned,
    index: 0,
    handle
  })

  return handle
}

/** A mount owns a receiver, not every preview sharing its session string. */
export function bindComposerPredictions(scope: ComposerPredictionScope, canReceive: () => boolean) {
  const binding: Receiver = { scope: { ...scope }, canReceive }

  if (isComposerPredictionScope(scope) && isActive(scope)) {
    receiver = binding
    $composerPredictions.set(null)
  }

  const current = () => {
    const value = $composerPredictions.get()

    return receiver === binding && isActive(binding.scope) && value && sameScope(value, binding.scope) ? value : null
  }

  const clear = () => {
    const value = current()

    return value ? clearComposerPredictions(value.handle) : false
  }

  return {
    current,
    clear,
    cycle: () => {
      const value = current()

      if (value) {
        $composerPredictions.set({ ...value, index: (value.index + 1) % value.suggestions.length })
      }
    },
    dispose: () => {
      if (receiver === binding) {
        clear()
        receiver = null
      }
    }
  }
}
