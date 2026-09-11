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

it('shows setup failure instead of waiting for apps when realm execution is blocked', () => {
  const renders = new Map<string, ComponentType<SessionContributionProps>>()
  const ctx = {
    rest: vi.fn(),
    register: ({ area, data }: { area: string; data: SessionContribution }) => renders.set(area, data.render)
  }

  realmsPlugin.register(ctx)
  const client = new QueryClient()
  clients.push(client)
  client.setQueryData(realmQueryOptions(ctx, session).queryKey, {
    mode: 'realm',
    realms: [],
    setup: { ready: false, message: 'Run hermes realms install-driver in this profile.' }
  })
  const StatusRow = renders.get(SESSION_AREAS.statusStack)!

  const view = render(
    <QueryClientProvider client={client}>
      <StatusRow session={session} />
    </QueryClientProvider>
  )

  expect(view.getByRole('alert').textContent).toContain('hermes realms install-driver')
  expect(view.container.textContent).not.toContain('waiting for apps')
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

it('names the Omarchy VM kind in the badge and its cost on hover', () => {
  // "Which machine am I on" is the one thing a glance must answer: a VM realm
  // is a different machine from the labwc realm, and it costs its whole -m
  // figure in host RAM while it runs, so the hover carries that cost.
  const renders = new Map<string, ComponentType<SessionContributionProps>>()
  const ctx = {
    rest: vi.fn(),
    register: ({ area, data }: { area: string; data: SessionContribution }) => renders.set(area, data.render)
  }

  realmsPlugin.register(ctx)
  const client = new QueryClient()
  clients.push(client)
  const options = realmQueryOptions(ctx, session)
  client.setQueryData(options.queryKey, {
    kind: 'omarchy-vm',
    realms: [{
      id: 'vm', stored_session_id: 'history', kind: 'omarchy-vm', state: 'live',
      memory_mb: 3072, network: true,
      stats: { memory_bytes: 3313926144, disk_bytes: 18546688 }
    }]
  })
  const ListBadge = renders.get(SESSION_AREAS.listBadge)!

  const view = render(
    <QueryClientProvider client={client}>
      <ListBadge session={session} />
    </QueryClientProvider>
  )

  expect(view.container.textContent).toBe('Omarchy VM')
  const title = view.container.querySelector('[title]')!.getAttribute('title')!
  expect(title).toContain('Omarchy VM')
  expect(title).toContain('live')
  expect(title).toContain('3.1 GB RAM')
})

it('reports a network-disabled VM realm on hover', () => {
  // restrict=on is a safety posture the user chose; the badge must not imply
  // the guest can reach the internet when it cannot.
  const renders = new Map<string, ComponentType<SessionContributionProps>>()
  const ctx = {
    rest: vi.fn(),
    register: ({ area, data }: { area: string; data: SessionContribution }) => renders.set(area, data.render)
  }

  realmsPlugin.register(ctx)
  const client = new QueryClient()
  clients.push(client)
  const options = realmQueryOptions(ctx, session)
  client.setQueryData(options.queryKey, {
    kind: 'omarchy-vm',
    realms: [{
      id: 'vm', stored_session_id: 'history', kind: 'omarchy-vm', state: 'live',
      memory_mb: 3072, network: false, stats: { memory_bytes: 3313926144 }
    }]
  })
  const ListBadge = renders.get(SESSION_AREAS.listBadge)!

  const view = render(
    <QueryClientProvider client={client}>
      <ListBadge session={session} />
    </QueryClientProvider>
  )

  expect(view.container.querySelector('[title]')!.getAttribute('title')!).toContain('no network')
})
