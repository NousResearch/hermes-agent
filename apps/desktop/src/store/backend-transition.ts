import { atom } from 'nanostores'

import type { SandboxCopy } from '@/i18n/types_sandbox'

// Planned backend transitions of a profile a plugin runs (host.setProfileBackend):
// its sandbox starting, restarting or stopping on purpose. While one lasts, a
// dropped or refused connection to that profile's backend is the transition,
// not a failure: the status bar and composer name it instead of "Backend
// offline" / "Reconnecting to Hermes…", and requests wait for the backend to
// come back instead of failing (use-gateway-request).
//
// Sources: the plugin's status route (launcher-transition.ts) and the SDK's
// switch (sdk/profile-backend.ts). A transition always ends: on the backend's
// return, on its source's say-so, or at its deadline, so a real failure still
// surfaces.

export type BackendTransitionKind = 'restarting' | 'starting' | 'stopping'

export interface BackendTransition {
  kind: BackendTransitionKind
  profile: string
  since: number
  /** Who started it; only that source (or the deadline) ends it. */
  source: 'launcher' | 'switch'
}

/** The longest a planned transition may hide a dead backend. */
export const BACKEND_TRANSITION_MAX_MS = 180_000

export const $backendTransitions = atom<Readonly<Record<string, BackendTransition>>>({})

type Waiter = (ended: boolean) => void

const waiters = new Map<string, Set<Waiter>>()
const deadlines = new Map<string, ReturnType<typeof setTimeout>>()

export function transitionProfileKey(profile: null | string | undefined): string {
  return String(profile ?? '').trim() || 'default'
}

export function backendTransitionFor(profile: null | string | undefined): BackendTransition | null {
  return $backendTransitions.get()[transitionProfileKey(profile)] ?? null
}

/**
 * A start or a stop hands the profile to another backend; a restart brings the
 * same backend, with the same sessions, back.
 */
export function changesBackend(kind: BackendTransitionKind | null | undefined): boolean {
  return kind === 'starting' || kind === 'stopping'
}

function clear(key: string, ended: boolean) {
  const timer = deadlines.get(key)

  if (timer !== undefined) {
    clearTimeout(timer)
    deadlines.delete(key)
  }

  if (key in $backendTransitions.get()) {
    const { [key]: _gone, ...rest } = $backendTransitions.get()
    $backendTransitions.set(rest)
  }

  const pending = waiters.get(key)
  waiters.delete(key)
  pending?.forEach(resolve => resolve(ended))
}

/** Start (or update) a planned transition for a profile. */
export function beginBackendTransition(
  profile: null | string | undefined,
  kind: BackendTransitionKind,
  options: { maxMs?: number; source: BackendTransition['source'] }
): void {
  const key = transitionProfileKey(profile)
  const current = $backendTransitions.get()[key]

  if (current?.kind === kind && current.source === options.source) {
    return
  }

  $backendTransitions.set({
    ...$backendTransitions.get(),
    [key]: { kind, profile: key, since: current?.since ?? Date.now(), source: options.source }
  })

  // Another kind or another source is a new transition, with a deadline of its own.
  clearTimeout(deadlines.get(key))
  deadlines.set(
    key,
    setTimeout(() => clear(key, false), options.maxMs ?? BACKEND_TRANSITION_MAX_MS)
  )
}

/**
 * End a profile's transition. `ok` false means it did not end the way it was
 * planned (the backend failed to come back): waiters give up and the normal
 * error path shows. A `source` only ends a transition it started.
 */
export function endBackendTransition(
  profile: null | string | undefined,
  options: { ok?: boolean; source?: BackendTransition['source'] } = {}
): void {
  const key = transitionProfileKey(profile)
  const current = $backendTransitions.get()[key]

  if (!current || (options.source && current.source !== options.source)) {
    return
  }

  clear(key, options.ok !== false)
}

/**
 * Resolve true once the profile's transition ends as planned, false when it
 * ends otherwise or `timeoutMs` passes first. True at once when none runs.
 */
export function waitForBackendTransitionEnd(profile: null | string | undefined, timeoutMs: number): Promise<boolean> {
  const key = transitionProfileKey(profile)

  if (!$backendTransitions.get()[key]) {
    return Promise.resolve(true)
  }

  return new Promise(resolve => {
    const set = waiters.get(key) ?? new Set<Waiter>()

    const timer = setTimeout(() => {
      set.delete(done)
      resolve(false)
    }, timeoutMs)

    function done(ended: boolean) {
      clearTimeout(timer)
      resolve(ended)
    }

    set.add(done)
    waiters.set(key, set)
  })
}

/** The one line the status bar and composer show for a transition. */
export function backendTransitionLabel(transition: BackendTransition, copy: SandboxCopy): string {
  return copy[transition.kind]
}

/** Test seam: forget every transition and waiter. */
export function resetBackendTransitionsForTests(): void {
  for (const key of new Set([...deadlines.keys(), ...waiters.keys()])) {
    clear(key, false)
  }

  $backendTransitions.set({})
}
