import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render } from '@testing-library/react'
import { type ComponentType } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

// @ts-expect-error Runtime plugins are plain JavaScript SDK consumers.
import realmsPlugin, { realmQueryOptions } from '../../../../plugins/hermes-realms/desktop/plugin.js'

import { SESSION_AREAS, type SessionContribution, type SessionContributionProps } from './session'

const session = { connectionId: 'local', profile: 'default', storedSessionId: 'history', runtimeSessionId: null }
const clients: QueryClient[] = []

afterEach(() => {
  cleanup()
  clients.splice(0).forEach(client => client.clear())
  vi.useRealTimers()
})

it('keeps unassociated session badges absent throughout failed polling and retries', async () => {
  vi.useFakeTimers()
  const rest = vi.fn().mockRejectedValue(new Error('Plugin backend unavailable'))
  const renders = new Map<string, ComponentType<SessionContributionProps>>()

  const ctx = {
    rest,
    register: ({ area, data }: { area: string; data: SessionContribution }) => renders.set(area, data.render)
  }

  realmsPlugin.register(ctx)
  const client = new QueryClient()
  clients.push(client)
  const ListBadge = renders.get(SESSION_AREAS.listBadge)!
  const TileBadge = renders.get(SESSION_AREAS.tileBadge)!
  const StatusRow = renders.get(SESSION_AREAS.statusStack)!

  const view = render(
    <QueryClientProvider client={client}>
      <ListBadge session={session} />
      <TileBadge session={session} />
      <StatusRow session={session} />
    </QueryClientProvider>
  )

  const statuses = new Set<string | undefined>()

  for (let tick = 0; tick < 24; tick++) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(view.container.textContent).toBe('')
    statuses.add(client.getQueryState(realmQueryOptions(ctx, session).queryKey)?.status)
  }

  expect(rest.mock.calls.length).toBeGreaterThan(2)
  expect(statuses).toEqual(new Set(['pending', 'error']))
})

it('preserves a known realm badge through refetch errors, then reconciles authoritative removal', async () => {
  const rest = vi.fn().mockRejectedValue(new Error('Temporary transport failure'))
  const renders = new Map<string, ComponentType<SessionContributionProps>>()

  const ctx = {
    rest,
    register: ({ area, data }: { area: string; data: SessionContribution }) => renders.set(area, data.render)
  }

  realmsPlugin.register(ctx)
  const client = new QueryClient()
  clients.push(client)
  const options = { ...realmQueryOptions(ctx, session), retry: false }
  client.setQueryData(options.queryKey, {
    realms: [{ id: 'owned', stored_session_id: 'history', state: 'live', window_count: 2 }]
  })
  const ListBadge = renders.get(SESSION_AREAS.listBadge)!

  const view = render(
    <QueryClientProvider client={client}>
      <ListBadge session={session} />
    </QueryClientProvider>
  )

  expect(view.container.textContent).toBe('Realm · 2')
  await act(async () => {
    await client.fetchQuery({ ...options, staleTime: 0 }).catch(() => {})
    await new Promise(resolve => setTimeout(resolve, 0))
  })
  expect(view.container.textContent).toBe('Realm · 2')
  expect(view.getByTitle(/status unavailable/i)).toBeTruthy()
  await act(async () => {
    rest.mockResolvedValue({ realms: [] })
    await client.fetchQuery({ ...options, staleTime: 0 })
    await new Promise(resolve => setTimeout(resolve, 0))
  })
  expect(view.container.textContent).toBe('')
})
