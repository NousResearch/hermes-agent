import { host, type PluginRestOptions, type SessionRouteContext } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi, fetchOriginTasks } from './api'
import { en, KANBAN_LOCALES } from './i18n'
import {
  deriveOriginView,
  isTerminalState,
  type OriginQueryState,
  refState,
  summarizeOrigin,
  useOriginView
} from './origin-links'
import { OriginRowBadge, OriginStrip } from './origin-ui'
import type { OriginActivity, OriginRef, OriginTasksResponse } from './types'
import { $requestedTask } from './ui'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const ref = (
  id: string,
  activity: OriginActivity,
  over: Partial<OriginRef> = {},
  title = `Task ${id}`
): OriginRef => ({
  board: 'default',
  evidence: 'ok',
  indexed_at: 100,
  origin_session_id: 'tip-1',
  source: 'create',
  task: { activity, activity_evidence: 'test', id, status: 'running', title },
  task_id: id,
  ...over
})

const answer = (over: Partial<OriginTasksResponse> = {}): OriginTasksResponse => ({
  now: 1,
  profile: 'default',
  refs: [],
  truncated: { lineage: false, refs: false, sessions: false, total_refs: 0 },
  unknown_sessions: [],
  ...over
})

const success = (data: OriginTasksResponse): OriginQueryState => ({ data, status: 'success' })
const ids = ['root-1', 'tip-1']

describe('origin view reduction', () => {
  it('never turns a lookup it could not make into "no tasks"', () => {
    expect(deriveOriginView(false, ids, success(answer()))).toEqual({ kind: 'out-of-scope' })

    expect(deriveOriginView(true, ids, { data: undefined, status: 'error' })).toEqual({
      kind: 'unavailable',
      reason: 'request'
    })

    expect(deriveOriginView(true, ids, { data: undefined, status: 'pending' })).toEqual({ kind: 'loading' })

    // The answering profile's store knows neither id: it is not this conversation's owner.
    expect(deriveOriginView(true, ids, success(answer({ unknown_sessions: ids })))).toEqual({
      kind: 'unavailable',
      reason: 'unknown-session'
    })
  })

  it('keeps this conversation’s lineage, orders loudest first and reports truncation', () => {
    const other = ref('t_other', 'background', { origin_session_id: 'someone-else' })

    const view = deriveOriginView(
      true,
      ids,
      success(
        answer({
          refs: [ref('t_done', 'done'), ref('t_wait', 'waiting'), ref('t_ask', 'needs-input'), other],
          truncated: { lineage: false, refs: true, sessions: false, total_refs: 9 }
        })
      )
    )

    expect(view.kind).toBe('ready')

    if (view.kind === 'ready') {
      expect(view.refs.map(r => r.task_id)).toEqual(['t_ask', 't_wait', 't_done'])
      expect(view.truncated).toEqual({ shown: 3, total: 9 })
    }
  })

  it('lists completed work without lighting a badge, but an unconfirmed ref does', () => {
    const finished = [ref('t_done', 'done'), ref('t_old', 'archived')]
    const gap = ref('t_gone', 'queued', { evidence: 'task_missing', task: null })

    expect(summarizeOrigin(finished)).toEqual({ live: 0, state: 'done' })
    expect(isTerminalState(summarizeOrigin(finished).state!)).toBe(true)
    expect(refState(gap)).toBe('unavailable')
    expect(summarizeOrigin([...finished, gap])).toEqual({ live: 1, state: 'unavailable' })
  })
})

const context = (over: Partial<SessionRouteContext> = {}): SessionRouteContext => ({
  ambient: true,
  connectionId: '',
  lineageIds: ids,
  profile: 'default',
  sessionId: 'root-1',
  ...over
})

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void
let reply: (path: string) => unknown

const rest = vi.fn(async (path: string, _options?: PluginRestOptions): Promise<unknown> => reply(path))

// bindApi itself refreshes boards through the same door; only origin lookups are under test.
const originCalls = () => rest.mock.calls.map(([path]) => path).filter(path => path.startsWith('/origin-tasks'))
const askedIds = (path: string) => decodeURIComponent(path.split('=')[1] ?? '').split(',')

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={client}>{children}</QueryClientProvider>
)

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  reply = () => answer()

  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, remove: vi.fn(), set: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  $requestedTask.set(null)
  vi.restoreAllMocks()
  rest.mockClear()
})

describe('per-conversation, route-scoped lookup', () => {
  it('shares one request for an identical seed set and never merges different conversations', async () => {
    await Promise.all([
      fetchOriginTasks('local', 'default', ['a-1', 'a-2']),
      // The row and the composer of one conversation: same ids, any order.
      fetchOriginTasks('local', 'default', ['a-2', 'a-1']),
      fetchOriginTasks('local', 'default', ['b-1']),
      fetchOriginTasks('local', 'other', ['a-1', 'a-2'])
    ])

    // One request per (route, conversation): a's two callers share, b and the other route are their own.
    expect(originCalls().map(askedIds).map(group => group.join('+')).sort()).toEqual(['a-1+a-2', 'a-1+a-2', 'b-1'])
  })

  it('keeps a crowded conversation’s cap and count off a sparse one', async () => {
    const crowded = Array.from({ length: 200 }, (_, n) => ref(`t_c${n}`, 'queued', { origin_session_id: 'crowded-1' }))

    reply = path => {
      if (!path.startsWith('/origin-tasks')) {
        return answer()
      }

      return askedIds(path).includes('crowded-1')
        ? answer({ refs: crowded, truncated: { lineage: false, refs: true, sessions: false, total_refs: 350 } })
        : answer({ refs: [ref('t_s1', 'waiting', { origin_session_id: 'sparse-1' })] })
    }

    const crowdedView = renderHook(
      () => useOriginView(context({ lineageIds: ['crowded-1'], sessionId: 'crowded-1' })),
      { wrapper }
    )

    const sparseView = renderHook(() => useOriginView(context({ lineageIds: ['sparse-1'], sessionId: 'sparse-1' })), {
      wrapper
    })

    await waitFor(() => expect(crowdedView.result.current.kind).toBe('ready'))
    await waitFor(() => expect(sparseView.result.current.kind).toBe('ready'))

    // The crowded conversation is capped and says so with ITS OWN total…
    expect(crowdedView.result.current).toMatchObject({ truncated: { shown: 200, total: 350 } })
    // …and the sparse one keeps every link, with no inherited truncation or count.
    expect(sparseView.result.current).toMatchObject({ refs: [{ task_id: 't_s1' }], truncated: null })
    expect(originCalls().map(askedIds)).toEqual(expect.arrayContaining([['crowded-1'], ['sparse-1']]))
  })

  it('answers each route with its own response, whatever order they resolve in', async () => {
    let releaseSlow: () => void = () => undefined
    const slow = new Promise<void>(resolve => (releaseSlow = resolve))

    reply = path => {
      if (!path.startsWith('/origin-tasks')) {
        return answer()
      }

      return askedIds(path).includes('slow-1')
        ? slow.then(() => answer({ profile: 'slow-profile', refs: [ref('t_slow', 'queued')] }))
        : answer({ profile: 'fast-profile', refs: [ref('t_fast', 'queued')] })
    }

    const slowAnswer = fetchOriginTasks('local', 'slow-profile', ['slow-1'])
    const fastAnswer = fetchOriginTasks('local', 'fast-profile', ['fast-1'])

    // The later route settles first and carries only its own refs…
    expect((await fastAnswer).refs.map(r => r.task_id)).toEqual(['t_fast'])

    releaseSlow()

    // …and the earlier one still gets its own, not the other's.
    expect((await slowAnswer).refs.map(r => r.task_id)).toEqual(['t_slow'])
  })

  it('never queries on behalf of a conversation the plugin route does not reach', async () => {
    const { result } = renderHook(() => useOriginView(context({ ambient: false, profile: 'other' })), { wrapper })

    await new Promise(resolve => setTimeout(resolve, 60))

    expect(result.current).toEqual({ kind: 'out-of-scope' })
    expect(originCalls()).toEqual([])
  })

  it('treats a profile switch as a clean miss: nothing from the previous profile is shown', async () => {
    reply = path =>
      path.startsWith('/origin-tasks')
        ? answer({ refs: [ref('t_default', 'queued', { origin_session_id: 'root-1' })] })
        : answer()

    const { rerender, result } = renderHook(({ ctx }) => useOriginView(ctx), {
      initialProps: { ctx: context() },
      wrapper
    })

    await waitFor(() => expect(result.current.kind).toBe('ready'))

    // The user moves to another profile whose route the plugin does not reach yet…
    rerender({ ctx: context({ ambient: false, profile: 'research' }) })
    expect(result.current).toEqual({ kind: 'out-of-scope' })

    // …and once it does, the default profile's cached answer is not painted for it.
    reply = path =>
      path.startsWith('/origin-tasks') ? answer({ profile: 'research', unknown_sessions: ids }) : answer()

    rerender({ ctx: context({ profile: 'research' }) })
    expect(result.current.kind).not.toBe('ready')

    await waitFor(() => expect(result.current).toEqual({ kind: 'unavailable', reason: 'unknown-session' }))
  })
})

describe('conversation surfaces', () => {
  it('a claim without execution evidence is "reserved", never the running badge', async () => {
    reply = () => answer({ refs: [ref('t_a', 'reserved')] })

    render(<OriginRowBadge context={context()} />, { wrapper })

    const badge = await screen.findByRole('status')

    expect(badge.getAttribute('data-kanban-origin')).toBe('reserved')
    expect(badge.getAttribute('aria-label')).toBe(en.origin.state.reserved)
  })

  it('keeps linked tasks reachable after they finish and opens their board log on request', async () => {
    const navigate = vi.spyOn(host, 'navigate').mockImplementation(() => undefined)

    reply = () =>
      answer({
        refs: [ref('t_fin', 'done', { board: 'ops' }, 'Ship the report'), ref('t_run', 'background', { board: 'ops' })]
      })

    render(<OriginStrip context={context()} />, { wrapper })

    // The strip is the in-session door: it starts collapsed, finished tasks included once opened.
    fireEvent.click(await screen.findByRole('button', { name: en.origin.expand }))
    expect(screen.getByRole('button', { name: en.origin.openTask('Ship the report') })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: en.origin.logs('Ship the report') }))
    expect($requestedTask.get()).toEqual({ board: 'ops', id: 't_fin', section: 'log' })
    expect(navigate).toHaveBeenCalledWith('/kanban')
  })

  it('says so when the lookup fails instead of painting nothing', async () => {
    reply = path => {
      if (path.startsWith('/origin-tasks')) {
        throw new Error('boom')
      }

      return answer()
    }

    render(<OriginStrip context={context()} />, { wrapper })

    await waitFor(() => expect(screen.getByText(en.origin.unreadable)).toBeTruthy())
  })

  it.each([
    [
      'a bound was hit and no count exists',
      { lineage: true, refs: false, sessions: false, total_refs: 0 },
      en.origin.unknownHere
    ],
    [
      'a known larger total exists',
      { lineage: false, refs: true, sessions: false, total_refs: 7 },
      en.origin.truncated(0, 7)
    ]
  ])('says an incomplete answer is incomplete even with no ref returned (%s)', async (_case, truncated, words) => {
    reply = path => (path.startsWith('/origin-tasks') ? answer({ truncated }) : answer())

    const { container } = render(
      <>
        <OriginStrip context={context()} />
        <OriginRowBadge context={context()} />
      </>,
      { wrapper }
    )

    // The strip does not vanish into "nothing linked"…
    await waitFor(() => expect(container.querySelector('[data-kanban-origin-truncated]')?.textContent).toBe(words))
    // …and the row does not go quiet: it carries the partial-evidence mark, never an idle look.
    expect(container.querySelector('[data-kanban-origin]')?.getAttribute('data-kanban-origin')).toBe('unavailable')
    expect(container.querySelector('[data-kanban-origin]')?.getAttribute('aria-label')).toContain(words)
  })
})
