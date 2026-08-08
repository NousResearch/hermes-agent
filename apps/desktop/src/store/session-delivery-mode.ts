/**
 * O5 session delivery mode — "极昼·无影灯" soft switch (default OFF).
 *
 * Scope: per sessionId only. NOT device-global. NOT goal auto-continue (that is
 * Advanced → cross-session experiment C / long-horizon-auto-continue).
 *
 * When on: intensity=deep + delivery=premium for this session; trivial still light.
 * Unattended/mass_parallel only if existing gates pass — mode does not skip gates.
 * Deep mode also prefers product-forcing-questions (头脑风暴) before heavy build.
 */
import { atom, computed } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'
import { $activeSessionId } from '@/store/session'

export type SessionDeliveryMode = 'off' | 'deep_premium'

const KEY_PREFIX = 'hermes.desktop.sessionDeliveryMode.v1:'

function storageKey(sessionId: string): string {
  return `${KEY_PREFIX}${sessionId.trim()}`
}

/** Map sessionId → mode. Missing key = off. */
export const $sessionDeliveryModes = atom<Record<string, SessionDeliveryMode>>({})

function readStored(sessionId: string): SessionDeliveryMode {
  if (typeof window === 'undefined' || !sessionId.trim()) {
    return 'off'
  }

  return storedBoolean(storageKey(sessionId), false) ? 'deep_premium' : 'off'
}

export function sessionDeliveryModeFor(sessionId: string | null | undefined): SessionDeliveryMode {
  const id = sessionId?.trim()

  if (!id) {
    return 'off'
  }

  const cached = $sessionDeliveryModes.get()[id]

  if (cached) {
    return cached
  }

  return readStored(id)
}

export function setSessionDeliveryMode(sessionId: string | null | undefined, mode: SessionDeliveryMode): void {
  const id = sessionId?.trim()

  if (!id) {
    return
  }

  const next: SessionDeliveryMode = mode === 'deep_premium' ? 'deep_premium' : 'off'
  $sessionDeliveryModes.set({ ...$sessionDeliveryModes.get(), [id]: next })

  if (typeof window !== 'undefined') {
    persistBoolean(storageKey(id), next === 'deep_premium')
    window.hermesDesktop?.setSessionDeliveryMode?.({ sessionId: id, mode: next })
  }
}

export function toggleSessionDeliveryMode(sessionId: string | null | undefined): SessionDeliveryMode {
  const cur = sessionDeliveryModeFor(sessionId)
  const next: SessionDeliveryMode = cur === 'deep_premium' ? 'off' : 'deep_premium'
  setSessionDeliveryMode(sessionId, next)

  return next
}

/** Active chat surface mode (primary composer). */
export const $activeSessionDeliveryMode = computed(
  [$activeSessionId, $sessionDeliveryModes],
  (activeId, map): SessionDeliveryMode => {
    const id = activeId?.trim()

    if (!id) {
      return 'off'
    }

    if (map[id]) {
      return map[id]
    }

    return readStored(id)
  }
)

/** Hydrate one session from localStorage into the atom (call on session focus). */
export function hydrateSessionDeliveryMode(sessionId: string | null | undefined): void {
  const id = sessionId?.trim()

  if (!id) {
    return
  }

  const mode = readStored(id)
  const cur = $sessionDeliveryModes.get()

  if (cur[id] === mode) {
    return
  }

  $sessionDeliveryModes.set({ ...cur, [id]: mode })
}
