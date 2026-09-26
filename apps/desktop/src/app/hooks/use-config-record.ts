import { useStore } from '@nanostores/react'
import { replaceEqualDeep, useQuery } from '@tanstack/react-query'

import {
  getHermesConfigRecord,
  peekConfigReadOrigin,
  type ProfileScope,
  profileScopeKey,
  retainConfigReadOrigin
} from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $activeConnectionId } from '@/store/connections'
import type { HermesConfigRecord } from '@/types/hermes'

// One shared cache for the whole profile config record (`GET /api/config`).
// Every settings surface (MCP, model, config) reads and writes through this key
// so a save in one shows in the others, and revisiting a tab paints the cache
// instead of blanking on a fresh fetch.
//
// Distinct from session/hooks/use-hermes-config.ts, which is side-effecting —
// it pushes personality/cwd/voice/… into the session stores for live chat.
export const HERMES_CONFIG_KEY = ['hermes-config-record'] as const

// Slot for one gateway inside the existing config-record cache. The id is
// `$activeConnectionId` — the resolved descriptor identity the rest of the app
// already uses — not a second cache. A bare root key let a settings save after
// a gateway switch paint the previous machine's record and PUT it onto the
// other config.yaml. An explicit pin that already names a connection keeps
// profileScopeKey's suffix; a string profile (settings "Applies to") is
// namespaced the same way so two gateways' `default` profiles do not share a row.
function configRecordSlot(profile: ProfileScope | undefined, connectionId: null | string | undefined): string | null {
  const active = (connectionId ?? '').trim()

  if (profile && typeof profile === 'object') {
    const pinned = (profile.connectionId ?? '').trim()

    if (pinned || !active) {
      return profileScopeKey(profile)
    }

    return profileScopeKey({ ...profile, connectionId: active })
  }

  if (profile == null) {
    return active || null
  }

  const scope = profileScopeKey(profile)

  return active ? `${active}::${scope}` : scope
}

export const hermesConfigKey = (
  profile?: ProfileScope,
  connectionId: null | string | undefined = $activeConnectionId.get()
) => {
  const slot = configRecordSlot(profile, connectionId)

  return slot == null ? HERMES_CONFIG_KEY : ([...HERMES_CONFIG_KEY, slot] as const)
}

// staleTime 0 → serve cache instantly, background-revalidate on every mount.
// `profile` scopes both the query key and the fetch. Omitting it still targets
// the app-wide active profile (`profileScoped(undefined)` fallback), but the
// cache slot is the active gateway's — not the bare root key.
export const useHermesConfigRecord = (profile?: ProfileScope) => {
  // Reactive read, not a store getter: under the React Compiler a value with
  // no reactive inputs is computed once per component instance, so a
  // getter-based key would freeze on the first gateway and keep serving its
  // record after a switch.
  const connectionId = useStore($activeConnectionId)

  const query = useQuery({
    queryKey: hermesConfigKey(profile, connectionId),
    // null/undefined both mean "no override" → fetch with undefined so
    // capabilityScoped falls back to the app-wide active profile (passing null
    // would wrongly target the primary backend).
    queryFn: () => {
      // $activeConnectionId.listen invalidates profile queries in the same
      // turn it publishes the new id, before this observer moves to the new
      // key. A refetch of the slot we are leaving must not store the new
      // gateway's record there — that is the other machine's config.yaml.
      if (connectionId && $activeConnectionId.get() !== connectionId) {
        const cached = queryClient.getQueryData<HermesConfigRecord>(hermesConfigKey(profile, connectionId))

        if (cached !== undefined) {
          return cached
        }
      }

      return getHermesConfigRecord(profile ?? undefined)
    },
    staleTime: 0,
    // Keep structural sharing so an unchanged refetch (every consumer mount at
    // staleTime 0, every invalidate) yields the SAME object and consumers'
    // memos/autosave effects don't re-arm. The read origin lives in a WeakMap
    // keyed by the record, so re-stamp whatever object survives the merge with
    // the origin of the NEW fetch (`next`, bound by getHermesConfigRecord) —
    // otherwise a retained object would keep routing writes to the gateway
    // that served the previous GET.
    structuralSharing: (previous: unknown, next: unknown) =>
      retainConfigReadOrigin(
        replaceEqualDeep(previous as HermesConfigRecord | undefined, next as HermesConfigRecord),
        next as object
      )
  })

  // Attach `writeScope` as a lazy getter instead of spreading `query`: useQuery
  // hands back a tracked-props Proxy, and spreading enumerates EVERY key, which
  // subscribes each consumer to fetchStatus/dataUpdatedAt/… churn. The getter
  // reads `query.data` through the proxy, so only `data` is tracked.
  //
  // `undefined`, never `null`: callers hand this straight to saveHermesConfig
  // with sparse `setNested({}, …)` patches, so the WeakMap misses and the
  // fallback is capabilityScoped(writeScope) → profileScoped(writeScope).
  // profileScoped(undefined) keeps the app-wide `_apiProfile`; profileScoped
  // (null) drops it and would write the PRIMARY profile before the first
  // GET resolves.
  Object.defineProperty(query, 'writeScope', {
    get: () => peekConfigReadOrigin(query.data) ?? undefined,
    configurable: true,
    enumerable: false
  })

  return query as typeof query & { writeScope: ReturnType<typeof peekConfigReadOrigin> }
}

// setHermesConfigCache writes the active gateway's record. The key is resolved
// at WRITE time, not when the writer is created, so a writer memoized by a
// long-lived settings panel (keyed only on the profile name) lands on whichever
// gateway is active when the save happens — the same row its query reads.
const writeHermesConfigCache =
  (keyFor: () => ReturnType<typeof hermesConfigKey>) =>
  (
    next:
      HermesConfigRecord | undefined | ((previous: HermesConfigRecord | undefined) => HermesConfigRecord | undefined)
  ) =>
    void queryClient.setQueryData<HermesConfigRecord>(keyFor(), previous => {
      const record = typeof next === 'function' ? next(previous) : next

      // setQueryData also runs the hook's structuralSharing (query.setData →
      // replaceData), but that pass stamps the origin of `record` (the NEW
      // value), which optimistic patches do not carry — and it only applies once
      // the observer has built the query. So the previous record's origin is
      // carried over explicitly here.
      return record ? retainConfigReadOrigin(record, previous) : record
    })

export const setHermesConfigCache = writeHermesConfigCache(() => hermesConfigKey())
export const hermesConfigCacheWriter = (profile?: ProfileScope) =>
  writeHermesConfigCache(() => hermesConfigKey(profile))

export const invalidateHermesConfig = (profile?: ProfileScope) =>
  queryClient.invalidateQueries({ queryKey: hermesConfigKey(profile) })
