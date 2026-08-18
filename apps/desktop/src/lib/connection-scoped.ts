import { atom, type WritableAtom } from 'nanostores'

import { type Codec, Codecs } from './persisted'
import { readKey, writeKey } from './storage'

// ── Connection-scoped persistence ───────────────────────────────────────────
// Multiple Desktop windows share one renderer origin — and therefore one
// localStorage area — while each window can be connected to a DIFFERENT
// gateway (primary vs per-session secondaries, local vs remote registry
// connections). Any gateway-bound list persisted under a single global key is
// silently shared between those windows, and per-window reconciliation (pin
// sync, session ordering) then bleeds one backend's state into another's
// sidebar (#77318).
//
// Persisted state must declare its scope in its own key. This module owns the
// CONNECTION scope: a `connectionScopedAtom` behaves like `persistentAtom`,
// but its storage key follows the active connection. The local connection
// keeps the BARE key — byte-identical behavior for single-backend users, the
// same contract as `backendScopeKey` in @hermes/shared — while a remote
// connection gets `<key>.remote.<encoded baseUrl>.<encoded profile>`, the
// shape `workspaceCwdKey` already established.
//
// Legacy globally-keyed values are deliberately NOT migrated into remote
// scopes: those keys accumulated writes from every window, so ownership of
// any given row is unknowable — adopting them wholesale is exactly the
// cross-connection bleed this scope prevents (see the #67709 precedent for
// profile keys). Backend-mirrored lists (pins) self-heal from the gateway's
// own rows instead.

/** Minimal slice of HermesConnection this module keys on. */
export interface ConnectionScopeDescriptor {
  baseUrl?: string
  mode?: 'local' | 'remote'
  profile?: null | string
}

/** The storage-key suffix for a connection. Local (and unknown) connections
 *  map to the bare key; remote connections get their own namespace. */
export function connectionScopeSuffix(connection: ConnectionScopeDescriptor | null | undefined): string {
  if (connection?.mode !== 'remote') {
    return ''
  }

  const base = encodeURIComponent(connection.baseUrl || 'remote')
  const profile = encodeURIComponent(connection.profile || 'default')

  return `.remote.${base}.${profile}`
}

interface ScopedEntry<T> {
  $value: WritableAtom<T>
  codec: Codec<T>
  fallback: T
  key: string
  /** True while a rescope is applying a loaded value — the persistence
   *  subscriber must not echo that read back into storage. */
  applying: boolean
}

let activeSuffix = ''

const registry: ScopedEntry<any>[] = []
const scopeListeners = new Set<() => void>()

/** The suffix for the connection the window is currently on. */
export function activeConnectionScopeSuffix(): string {
  return activeSuffix
}

/** Observe scope changes (fires BEFORE the scoped atoms repaint, so
 *  per-connection bookkeeping can reset ahead of the reload). */
export function onConnectionScopeChange(listener: () => void): () => void {
  scopeListeners.add(listener)

  return () => void scopeListeners.delete(listener)
}

function loadEntry<T>(entry: ScopedEntry<T>): T {
  const raw = readKey(entry.key + activeSuffix)

  if (raw === null) {
    return entry.fallback
  }

  try {
    return entry.codec.decode(raw)
  } catch {
    return entry.fallback
  }
}

/**
 * A `persistentAtom` whose storage key carries the active connection scope.
 * Reads seed from the current scope's key; writes land under it. When the
 * window's connection changes, every scoped atom reloads from the new scope.
 */
export function connectionScopedAtom<T>(key: string, fallback: T, codec: Codec<T> = Codecs.json<T>()): WritableAtom<T> {
  const entry: ScopedEntry<T> = { $value: atom<T>(fallback), applying: false, codec, fallback, key }
  entry.$value.set(loadEntry(entry))
  registry.push(entry)

  // Persist CHANGES only — same creation-emission and rescope suppression
  // rationale as persistentAtom: echoing a just-read value back out can
  // clobber a storage snapshot another window is about to read.
  let creationEmission = true

  entry.$value.subscribe(value => {
    if (creationEmission) {
      creationEmission = false

      return
    }

    if (entry.applying) {
      return
    }

    writeKey(entry.key + activeSuffix, entry.codec.encode(value))
  })

  return entry.$value
}

/**
 * Point every connection-scoped atom at `connection`'s storage scope.
 *
 * Called whenever the window's connection descriptor is published. A null
 * descriptor is an ordinary disconnect/reconnect state — not evidence the
 * user selected another backend — so it keeps the current scope (the same
 * contract as syncCronModelImpactConnection).
 */
export function rescopeConnectionScopedStores(connection: ConnectionScopeDescriptor | null | undefined): void {
  if (!connection) {
    return
  }

  const next = connectionScopeSuffix(connection)

  if (next === activeSuffix) {
    return
  }

  activeSuffix = next

  // Bookkeeping listeners first: pin-sync's mirrored/pending sets describe
  // the PREVIOUS backend and must be gone before the reload below triggers
  // their reconcile against the new scope's lists.
  for (const listener of scopeListeners) {
    listener()
  }

  for (const entry of registry) {
    entry.applying = true

    try {
      entry.$value.set(loadEntry(entry))
    } finally {
      entry.applying = false
    }
  }
}
