import { queryClient } from '@/lib/query-client'
import { HUB_SOURCES_KEY, OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'
import type { SkillHubSourcesResponse } from '@/types/hermes'
import type { CatalogSource } from '@/types/skill-catalog'

export type BootstrapOrigin = 'official' | 'hub'

interface OfficialBootstrap {
  provider_sources?: CatalogSource[]
}

/** The slice of a React Query result the lifecycle reads: cached body and its cache clock. */
interface BootstrapQuery<T> {
  data: T | undefined
  dataUpdatedAt: number
  enabled: boolean
}

interface Reply {
  issued: number
  landed: boolean
}

interface Seen {
  updatedAt: number
  generation: number
}

/** The ordering evidence of one connection/profile scope. */
interface Lifecycle {
  clock: number
  replies: Partial<Record<BootstrapOrigin, Reply>>
  seen: Partial<Record<BootstrapOrigin, Seen>>
}

export interface PublishedSources {
  /** Which bootstrap reply the live descriptors come from; undefined before any reply. */
  origin: BootstrapOrigin | undefined
  sources: CatalogSource[]
}

const ROOTS: Record<BootstrapOrigin, string> = { official: OFFICIAL_SKILLS_KEY[0], hub: HUB_SOURCES_KEY[0] }
const ORIGINS = Object.keys(ROOTS) as BootstrapOrigin[]

/** The scoped cache entry a bootstrap body lives in. */
export const bootstrapKey = (origin: BootstrapOrigin, scope: string) => [ROOTS[origin], scope] as const

const hubProviders = (data: SkillHubSourcesResponse | undefined): CatalogSource[] =>
  (data?.sources ?? []).filter((source): source is typeof source & CatalogSource => 'provider' in source)

// Ordering lives with the cached bodies it orders, not with the component that observes them: a
// scoped catalog remounts on every connection/profile change while React Query keeps both bodies
// fresh, and a reply may land after the component that issued its request is gone.
const lifecycles = new Map<string, Lifecycle>()

// A body's ordering lasts exactly as long as the body. When the cache drops a bootstrap entry
// (clear, garbage collection, removal) its reply and observation go with it, so the next body for
// that origin is a new request or a new observation, never the reply to a request whose body
// nobody holds any more.
let watching = false

const watchEvidence = () => {
  if (watching) {
    return
  }

  watching = true
  queryClient.getQueryCache().subscribe(event => {
    if (event.type !== 'removed') {
      return
    }

    const [root, scope, ...rest] = event.query.queryKey
    const origin = ORIGINS.find(candidate => ROOTS[candidate] === root)
    const lifecycle = rest.length === 0 && typeof scope === 'string' ? lifecycles.get(scope) : undefined

    if (origin && lifecycle) {
      delete lifecycle.replies[origin]
      delete lifecycle.seen[origin]
    }
  })
}

const lifecycleOf = (scope: string): Lifecycle => {
  watchEvidence()
  let lifecycle = lifecycles.get(scope)

  if (!lifecycle) {
    lifecycle = { clock: 0, replies: {}, seen: {} }
    lifecycles.set(scope, lifecycle)
  }

  return lifecycle
}

/**
 * One scoped authority for which provider descriptors are live.
 *
 * Both bootstrap replies (official and hub `/sources`) are full snapshots of the same
 * registry, so they are never merged: the reply to the LATEST REQUEST publishes the whole
 * descriptor set, and the other reply is retired with its sources. A reply is stamped with
 * the generation of the request that produced it, not with its arrival, so a cached or
 * delayed reply to an older request (a pre-disable hub response landing after the
 * providerless official refresh) can never resurrect a retired source. A body written to the
 * cache without a request counts as observed now; bodies first observed together are ordered by
 * when the cache received them. The ordering is keyed by connection/profile scope and outlives
 * the component, so a remount or a round trip through another scope within the bodies'
 * freshness publishes the same set the last request did.
 */
export function sourceLifecycle(scope: string) {
  const lifecycle = lifecycleOf(scope)

  /** Wrap a bootstrap queryFn so its reply carries the generation of its request. */
  const issue = <T,>(origin: BootstrapOrigin, fetch: () => Promise<T>) => {
    const reply: Reply = { issued: ++lifecycle.clock, landed: false }
    lifecycle.replies[origin] = reply

    return fetch().then(body => {
      reply.landed = true

      return body
    })
  }

  // Generation of the body a query holds now. New data takes the generation of the request
  // that landed it; data that arrived without a landed request is observed as newest.
  const generation = (origin: BootstrapOrigin, query: BootstrapQuery<unknown>) => {
    const last = lifecycle.seen[origin]

    if (last?.updatedAt === query.dataUpdatedAt) {
      return last.generation
    }

    const reply = lifecycle.replies[origin]
    const landed = reply?.landed ? reply.issued : undefined

    if (landed !== undefined) {
      delete lifecycle.replies[origin]
    }

    const next = { updatedAt: query.dataUpdatedAt, generation: landed ?? ++lifecycle.clock }
    lifecycle.seen[origin] = next

    return next.generation
  }

  const publish = (
    official: BootstrapQuery<OfficialBootstrap>,
    hub: BootstrapQuery<SkillHubSourcesResponse>
  ): PublishedSources => {
    const candidates = [
      { origin: 'official' as const, query: official, sources: official.data?.provider_sources ?? [] },
      { origin: 'hub' as const, query: hub, sources: hubProviders(hub.data) }
    ]
      // A disabled input is not maintained, so its cached body is not evidence.
      .filter(candidate => candidate.query.enabled && candidate.query.data !== undefined)
      // Observed in cache order, so a first observation never depends on candidate order.
      .sort((a, b) => a.query.dataUpdatedAt - b.query.dataUpdatedAt)
      .map(candidate => ({ ...candidate, generation: generation(candidate.origin, candidate.query) }))
      .sort((a, b) => b.generation - a.generation)

    const authority = candidates[0]

    return { origin: authority?.origin, sources: (authority?.sources ?? []).filter(source => source.state !== 'hidden') }
  }

  return { issue, publish }
}
