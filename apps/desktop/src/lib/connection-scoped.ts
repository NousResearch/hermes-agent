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
// connection gets `<key>.remote.<encoded baseUrl>.<encoded profile>` by
// default (the shape `workspaceCwdKey` already established). Gateway-wide
// mirrors (pins) pass `{ includeProfile: false }` so a profile switch cannot
// fragment the cache and re-assert a stale copy.
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

export interface ConnectionScopeOptions {
  /**
   * When false, a remote suffix is `.remote.<baseUrl>` only. Use this for
   * gateway-wide mirrors (pins) so a profile switch cannot reload a stale
   * per-profile copy. Defaults to true — session order and other
   * profile-local lists stay isolated.
   */
  includeProfile?: boolean
}

/** The storage-key suffix for a connection. Local (and unknown) connections
 *  map to the bare key; remote connections get their own namespace. */
export function connectionScopeSuffix(
  connection: ConnectionScopeDescriptor | null | undefined,
  includeProfile = true
): string {
  if (connection?.mode !== 'remote') {
    return ''
  }

  const base = encodeURIComponent(connection.baseUrl || 'remote')

  if (!includeProfile) {
    return `.remote.${base}`
  }

  const profile = encodeURIComponent(connection.profile || 'default')

  return `.remote.${base}.${profile}`
}

interface ScopedEntry<T> {
  $value: WritableAtom<T>
  codec: Codec<T>
  fallback: T
  includeProfile: boolean
  key: string
  /** Last suffix this entry loaded or persisted under. */
  suffix: string
  /** True while a rescope is applying a loaded value — the persistence
   *  subscriber must not echo that read back into storage. */
  applying: boolean
}

let activeConnection: ConnectionScopeDescriptor | null | undefined
let activeSuffix = ''
let activeGatewaySuffix = ''

const registry: ScopedEntry<any>[] = []
const scopeListeners = new Set<() => void>()

function suffixFor(entry: Pick<ScopedEntry<unknown>, 'includeProfile'>): string {
  return connectionScopeSuffix(activeConnection, entry.includeProfile)
}

/** The suffix for the connection the window is currently on. */
export function activeConnectionScopeSuffix(): string {
  return activeSuffix
}

/** Observe gateway-identity changes (fires BEFORE scoped atoms that
 *  follow the connection reload, so pin-sync bookkeeping can reset).
 *  Profile-only switches do not fire: pin state is gateway-wide. */
export function onConnectionScopeChange(listener: () => void): () => void {
  scopeListeners.add(listener)

  return () => void scopeListeners.delete(listener)
}

function loadEntry<T>(entry: ScopedEntry<T>): T {
  const raw = readKey(entry.key + suffixFor(entry))

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
export function connectionScopedAtom<T>(
  key: string,
  fallback: T,
  codec: Codec<T> = Codecs.json<T>(),
  options?: ConnectionScopeOptions
): WritableAtom<T> {
  const includeProfile = options?.includeProfile !== false
  const entry: ScopedEntry<T> = {
    $value: atom<T>(fallback),
    applying: false,
    codec,
    fallback,
    includeProfile,
    key,
    suffix: connectionScopeSuffix(activeConnection, includeProfile)
  }
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

    writeKey(entry.key + suffixFor(entry), entry.codec.encode(value))
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
  const nextGateway = connectionScopeSuffix(connection, false)

  if (next === activeSuffix) {
    return
  }

  activeConnection = connection
  activeSuffix = next

  // Pin-sync's mirrored/pending sets describe the PREVIOUS gateway. Fire
  // only when the connection (not the profile) changes — pins are
  // gateway-wide, so a profile switch must not reset bookkeeping and then
  // re-PATCH a leftover per-profile copy.
  if (nextGateway !== activeGatewaySuffix) {
    activeGatewaySuffix = nextGateway

    for (const listener of scopeListeners) {
      listener()
    }
  }

  for (const entry of registry) {
    const entrySuffix = suffixFor(entry)

    if (entrySuffix === entry.suffix) {
      continue
    }

    entry.suffix = entrySuffix
    entry.applying = true

    try {
      entry.$value.set(loadEntry(entry))
    } finally {
      entry.applying = false
    }
  }
}
