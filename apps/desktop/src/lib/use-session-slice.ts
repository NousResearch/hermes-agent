import { useCallback, useRef, useSyncExternalStore } from 'react'

interface SliceStore<T> {
  get(): Record<string, T[] | undefined>
  listen(listener: () => void): () => void
}

// Stable empty result so an absent key never yields a fresh array (which would
// defeat the snapshot bail-out and re-render on every store write).
const EMPTY: readonly never[] = []

/**
 * Subscribe to ONE session's slice of a `Record<sessionId, T[]>` nanostore,
 * re-rendering only when *that* slice's reference changes — not on writes to
 * other sessions. The map reference churns on every cross-session update, so a
 * plain `useStore(map)` re-renders all consumers globally; reading `map[key]`
 * through `useSyncExternalStore` bails out whenever the keyed array is
 * unchanged (the stores update immutably per key). Returns a shared empty array
 * when the key is null/absent.
 *
 * Note: only helps stores whose per-key arrays are referentially stable across
 * unrelated writes (plain atoms with immutable per-key updates). A `computed`
 * that rebuilds the whole map churns every slice — use a presence/edge selector
 * there instead.
 */
export function useSessionSlice<T>(store: SliceStore<T>, key: string | null): T[] {
  return useSyncExternalStore(
    onChange => store.listen(onChange),
    () => (key ? (store.get()[key] ?? (EMPTY as unknown as T[])) : (EMPTY as unknown as T[]))
  )
}

interface ReadableStore<T> {
  get(): T
  listen(listener: () => void): () => void
}

/**
 * Subscribe to a SCALAR derived from a hot store, re-rendering only when that
 * scalar changes by `Object.is` — not on every write to the store it came from.
 *
 * `useStore($someHotStore)` bails out on reference equality alone, so a store
 * republished per streaming token re-renders every consumer even when the two
 * or three fields they actually read are identical. `$sessionStates` is the
 * canonical case: it is republished on every message delta, so a component
 * reading only `busy` or `turnStartedAt` off it pays for the whole transcript's
 * churn.
 *
 * `select` must return a PRIMITIVE (or a referentially stable value). Returning
 * a fresh object or array defeats the bail-out and reintroduces the churn this
 * exists to remove — derive one scalar per call instead.
 */
export function useStoreSelector<T, S>(store: ReadableStore<T>, select: (value: T) => S): S {
  // `select` is read through a ref so an inline arrow at the call site doesn't
  // resubscribe on every render; useSyncExternalStore re-reads the snapshot on
  // each render anyway, so the latest selector is always applied.
  const selectRef = useRef(select)
  selectRef.current = select

  const subscribe = useCallback((onChange: () => void) => store.listen(onChange), [store])

  return useSyncExternalStore(subscribe, () => selectRef.current(store.get()))
}

/**
 * `useStoreSelector` for a scalar whose inputs span SEVERAL stores: recomputes
 * when any of them notifies, still re-rendering only when the scalar changes.
 *
 * Subscribing to one store while the selector reads others is the failure this
 * exists to prevent — it looks correct for as long as the subscribed store
 * happens to churn on its own, then silently goes stale when it doesn't. If a
 * selector reads it, list it.
 */
export function useStoresSelector<S>(stores: readonly ReadableStore<unknown>[], select: () => S): S {
  const selectRef = useRef(select)
  selectRef.current = select

  // Hold the array identity steady: call sites pass an inline literal of
  // module-level singletons, so only a genuine store swap should resubscribe.
  const storesRef = useRef(stores)

  if (storesRef.current.length !== stores.length || storesRef.current.some((store, i) => store !== stores[i])) {
    storesRef.current = stores
  }

  const stable = storesRef.current

  const subscribe = useCallback(
    (onChange: () => void) => {
      const stops = stable.map(store => store.listen(onChange))

      return () => {
        for (const stop of stops) {
          stop()
        }
      }
    },
    [stable]
  )

  return useSyncExternalStore(subscribe, () => selectRef.current())
}
