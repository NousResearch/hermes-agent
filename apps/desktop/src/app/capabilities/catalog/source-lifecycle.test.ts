import { beforeEach, expect, it } from 'vitest'

import { queryClient } from '@/lib/query-client'
import type { SkillHubSourcesResponse } from '@/types/hermes'
import type { CatalogSource } from '@/types/skill-catalog'

import readyFixture from './provider-fixtures/official-ready.json'
import stockFixture from './provider-fixtures/official-stock.json'
import { bootstrapKey, sourceLifecycle } from './source-lifecycle'

const official = readyFixture as unknown as { provider_sources: CatalogSource[] }
const stock = stockFixture as unknown as { provider_sources?: CatalogSource[] }
const source = official.provider_sources[0]
const hub = (sources = [source]) => ({ sources, installed: {}, featured: [] }) as unknown as SkillHubSourcesResponse
const cached = <T,>(data: T, dataUpdatedAt: number) => ({ data, dataUpdatedAt, enabled: true })

let scope: string
let run = 0

beforeEach(() => {
  queryClient.clear()
  scope = `unit::lifecycle-${++run}`
})

it('keeps the order of the latest request across remounts in both directions', async () => {
  const mounted = sourceLifecycle(scope)
  await mounted.issue('official', async () => official)
  await mounted.issue('hub', async () => hub())
  expect(mounted.publish(cached(official, 1), cached(hub(), 2)).origin).toBe('hub')

  await mounted.issue('official', async () => stock)
  expect(mounted.publish(cached(stock, 3), cached(hub(), 2))).toEqual({ origin: 'official', sources: [] })
  expect(sourceLifecycle(scope).publish(cached(stock, 3), cached(hub(), 2))).toEqual({ origin: 'official', sources: [] })

  await sourceLifecycle(scope).issue('hub', async () => hub([]))
  expect(sourceLifecycle(scope).publish(cached(stock, 3), cached(hub([]), 4))).toEqual({ origin: 'hub', sources: [] })
})

it('stamps a reply landing after a remount with the order of its request', async () => {
  const mounted = sourceLifecycle(scope)
  let land!: (body: unknown) => void
  const pending = mounted.issue('official', () => new Promise(done => { land = done }))
  await mounted.issue('hub', async () => hub([]))
  const remounted = sourceLifecycle(scope)
  land(official)
  await pending
  expect(remounted.publish(cached(official, 5), cached(hub([]), 2))).toEqual({ origin: 'hub', sources: [] })
})

it('drops the order of a body the cache no longer holds', async () => {
  const mounted = sourceLifecycle(scope)
  let land!: (body: unknown) => void
  const pending = mounted.issue('official', () => new Promise(done => { land = done }))
  await mounted.issue('hub', async () => hub())
  queryClient.setQueryData(bootstrapKey('official', scope), official)
  queryClient.removeQueries({ queryKey: bootstrapKey('official', scope), exact: true })
  land(official)
  await pending
  expect(sourceLifecycle(scope).publish(cached(stock, 9), cached(hub(), 2))).toEqual({ origin: 'official', sources: [] })
})

it('orders bodies first observed together by when the cache received them', () => {
  expect(sourceLifecycle(scope).publish(cached(stock, 20), cached(hub(), 10))).toEqual({ origin: 'official', sources: [] })
  expect(sourceLifecycle(scope).publish(cached(stock, 20), cached(hub(), 10)).origin).toBe('official')
  expect(sourceLifecycle(`${scope}:reverse`).publish(cached(official, 10), cached(hub([]), 20))).toEqual({ origin: 'hub', sources: [] })
})

it('keeps the order of each connection/profile scope separate', async () => {
  const first = sourceLifecycle(`${scope}:first`)
  const second = sourceLifecycle(`${scope}:second`)
  await first.issue('official', async () => official)
  await first.issue('hub', async () => hub())
  await second.issue('hub', async () => hub())
  await second.issue('official', async () => stock)
  expect(first.publish(cached(official, 1), cached(hub(), 2)).origin).toBe('hub')
  expect(second.publish(cached(stock, 1), cached(hub(), 2))).toEqual({ origin: 'official', sources: [] })
})
