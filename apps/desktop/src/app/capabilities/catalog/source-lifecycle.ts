import { useRef } from 'react'

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

export interface PublishedSources {
  /** Which bootstrap reply the live descriptors come from; undefined before any reply. */
  origin: BootstrapOrigin | undefined
  sources: CatalogSource[]
}

const hubProviders = (data: SkillHubSourcesResponse | undefined): CatalogSource[] =>
  (data?.sources ?? []).filter((source): source is typeof source & CatalogSource => 'provider' in source)

/**
 * One scoped authority for which provider descriptors are live.
 *
 * Both bootstrap replies (official and hub `/sources`) are full snapshots of the same
 * registry, so they are never merged: the reply to the LATEST REQUEST publishes the whole
 * descriptor set, and the other reply is retired with its sources. A reply is stamped with
 * the generation of the request that produced it, not with its arrival, so a cached or
 * delayed reply to an older request (a pre-disable hub response landing after the
 * providerless official refresh) can never resurrect a retired source. A body written to the
 * cache without a request counts as observed now. The component that owns this hook is keyed
 * by connection/profile, so the clock is scoped with it.
 */
export function useSourceLifecycle() {
  const clock = useRef(0)
  const replies = useRef<Partial<Record<BootstrapOrigin, Reply>>>({})
  const seen = useRef<Partial<Record<BootstrapOrigin, Seen>>>({})

  /** Wrap a bootstrap queryFn so its reply carries the generation of its request. */
  const issue = <T,>(origin: BootstrapOrigin, fetch: () => Promise<T>) => {
    const reply: Reply = { issued: ++clock.current, landed: false }
    replies.current[origin] = reply

    return fetch().then(body => {
      reply.landed = true

      return body
    })
  }

  // Generation of the body a query holds now. New data takes the generation of the request
  // that landed it; data that arrived without a landed request is observed as newest.
  const generation = (origin: BootstrapOrigin, query: BootstrapQuery<unknown>) => {
    const last = seen.current[origin]

    if (last?.updatedAt === query.dataUpdatedAt) {
      return last.generation
    }

    const reply = replies.current[origin]
    const landed = reply?.landed ? reply.issued : undefined

    if (landed !== undefined) {
      delete replies.current[origin]
    }

    const next = { updatedAt: query.dataUpdatedAt, generation: landed ?? ++clock.current }
    seen.current[origin] = next

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
      .map(candidate => ({ ...candidate, generation: generation(candidate.origin, candidate.query) }))
      .sort((a, b) => b.generation - a.generation)

    const authority = candidates[0]

    return { origin: authority?.origin, sources: (authority?.sources ?? []).filter(source => source.state !== 'hidden') }
  }

  return { issue, publish }
}
