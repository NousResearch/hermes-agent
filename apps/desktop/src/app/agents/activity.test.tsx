import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import type { AgentOverview, OverviewSession } from '@/global'
import { mergeOverview, overviewCache } from '@/store/agent-overview'
import { overlayObservedStatuses } from '@/store/agent-overview-status'

import { isRecentActivity } from './activity'
import { SessionOverview } from './sessions'

const NOW = Date.UTC(2026, 8, 5, 12)
const MINUTE = 60_000

function snapshot(sessions: OverviewSession[]): AgentOverview {
  return {
    fetchedAt: Date.now(),
    sources: [
      {
        connectionId: 'hq',
        label: 'HQ source',
        kind: 'remote',
        state: 'ready',
        sessions: [],
        canonical: [],
        live: sessions,
        profiles: [],
        total: sessions.length,
        offset: sessions.length,
        complete: true,
        liveCoverage: 'process',
        errors: []
      }
    ]
  }
}

function session(id: string, age: number, extra: Partial<OverviewSession> = {}): OverviewSession {
  return { id, profile: 'coder', title: id, last_active: (NOW - age) / 1000, ...extra }
}

async function mount(data: AgentOverview) {
  overviewCache.state.set({ data: mergeOverview(undefined, data), loading: false, error: null })
  window.hermesDesktop = {
    ...window.hermesDesktop,
    getAgentOverview: vi.fn(async () => ({ ...data, fetchedAt: Date.now() }))
  } as typeof window.hermesDesktop

  const view = render(
    <PaneVisibleContext.Provider value={true}>
      <SessionOverview />
    </PaneVisibleContext.Provider>
  )

  await act(async () => {})

  return view
}

const titles = () => screen.queryAllByTestId('agent-row').map(row => row.querySelector('.font-medium')?.textContent)

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(NOW)
})
afterEach(() => {
  cleanup()
  overviewCache.state.set({ loading: false, error: null })
  vi.restoreAllMocks()
  vi.useRealTimers()
})

it('defaults to active work of any age and only the last 15 minutes of idle activity', async () => {
  await mount(
    snapshot([
      session('old working', 24 * 60 * MINUTE, { status: 'working', runtime_id: 'work' }),
      session('old needs input', 24 * 60 * MINUTE, { status: 'waiting', runtime_id: 'wait' }),
      session('recent idle', 14 * MINUTE, { status: 'idle' }),
      session('recent finished unread', MINUTE, { status: 'idle', unread: true }),
      session('old idle', 16 * MINUTE, { status: 'idle' }),
      session('old unread', 16 * MINUTE, { status: 'idle', unread: true })
    ])
  )
  expect(titles()).toEqual(['old needs input', 'old working', 'recent finished unread', 'recent idle'])
  expect(screen.queryByText(/of 4 sessions/)).toBeNull()
  expect(screen.getByTestId('agent-source-coverage').textContent).toContain('HQ source')
})

it('ages idle out at the cutoff without new data, preserving selection and the reply draft', async () => {
  vi.spyOn(overviewCache, 'watch').mockReturnValue(() => {})
  await mount(snapshot([session('finishing', 15 * MINUTE - 1000, { status: 'idle' })]))
  fireEvent.click(screen.getByRole('button', { name: /finishing/ }))
  fireEvent.change(screen.getByRole('textbox', { name: 'Reply' }), { target: { value: 'keep this draft' } })
  const data = overviewCache.state.get().data
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1000)
  })
  expect(titles()).toEqual(['finishing'])
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1)
  })
  expect(titles()).toEqual([])
  expect(overviewCache.state.get().data).toBe(data)
  expect(screen.getByText('All quiet')).toBeTruthy()
  expect((screen.getByRole('textbox', { name: 'Reply' }) as HTMLTextAreaElement).value).toBe('keep this draft')
})

it('offers All sessions as an explicit history filter with truthful empty copy and counts', async () => {
  await mount(snapshot([session('history', 60 * MINUTE, { status: 'idle', unread: true })]))
  expect(screen.getByText('All quiet')).toBeTruthy()
  expect(
    screen.getByText(
      'Running and waiting sessions stay here. Finished chats fade out after 15 minutes; switch to All sessions for history.'
    )
  ).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'All sessions' }))
  expect(titles()).toEqual(['history'])
  fireEvent.change(screen.getByRole('textbox', { name: 'Search sessions' }), { target: { value: 'absent' } })
  expect(screen.getByText('No matching sessions')).toBeTruthy()
  expect(screen.queryByText('All quiet')).toBeNull()
})

it('retains formerly active stale work, not stale historical idle, without reviving live authority', async () => {
  const initial = snapshot([
    session('lost worker', 24 * 60 * MINUTE, { status: 'working', runtime_id: 'work' }),
    session('lost input', 24 * 60 * MINUTE, { status: 'waiting', runtime_id: 'wait' }),
    session('old idle', 24 * 60 * MINUTE, { status: 'idle' }),
    session('old unread', 24 * 60 * MINUTE, { status: 'idle', unread: true })
  ])

  await mount(initial)

  const outage: AgentOverview = {
    ...initial,
    sources: initial.sources.map(source => ({
      ...source,
      state: 'offline',
      complete: false,
      live: [],
      error: 'unreachable'
    }))
  }

  await act(async () => {
    overviewCache.state.set({
      data: mergeOverview(overviewCache.state.get().data, outage),
      loading: false,
      error: null
    })
  })
  expect(titles()).toEqual(['lost worker', 'lost input'])
  expect(screen.queryByRole('region', { name: 'Working' })).toBeNull()
  expect(screen.queryByRole('region', { name: 'Needs you' })).toBeNull()
  expect(screen.getAllByText(/^Stale · /)).toHaveLength(2)
  expect(overviewCache.state.get().data?.rows.find(row => row.id === 'lost worker')).toMatchObject({
    stale: true,
    attention: 'idle',
    runtimeId: undefined,
    lastKnownAttention: 'working'
  })
  fireEvent.click(screen.getByRole('button', { name: /lost worker/ }))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  await act(async () => {
    overviewCache.state.set({
      data: mergeOverview(overviewCache.state.get().data, outage),
      loading: false,
      error: null
    })
  })
  expect(titles()).toEqual(['lost worker', 'lost input'])
  const recovered = snapshot(initial.sources[0]!.live.map(row => ({ ...row, status: 'idle' })))
  await act(async () => {
    overviewCache.state.set({
      data: mergeOverview(overviewCache.state.get().data, recovered),
      loading: false,
      error: null
    })
  })
  expect(titles()).toEqual([])
})

it('retains last-known renderer attention through stale snapshots, never trusting disconnected ledgers', () => {
  const data = snapshot([session('observed', 60 * MINUTE, { status: 'idle' })])
  const fresh = mergeOverview(undefined, data)

  const observation = {
    connectionId: 'hq',
    profile: 'coder',
    id: 'observed',
    runtimeId: 'run',
    status: 'needs-input' as const,
    socketOpen: true
  }

  const observed = overlayObservedStatuses(fresh.rows, [observation])
  expect(observed[0]?.attention).toBe('needs-you')

  const stale = mergeOverview(fresh, {
    ...data,
    sources: data.sources.map(source => ({ ...source, state: 'offline', complete: false }))
  })

  const result = overlayObservedStatuses(stale.rows, [{ ...observation, socketOpen: false }], observed)
  expect(result[0]).toMatchObject({
    stale: true,
    attention: 'idle',
    runtimeId: undefined,
    lastKnownAttention: 'needs-you'
  })
  expect(isRecentActivity(result[0]!, Date.now())).toBe(true)
  const repeated = overlayObservedStatuses(stale.rows, [], result)
  expect(repeated[0]?.lastKnownAttention).toBe('needs-you')
  expect(overlayObservedStatuses(stale.rows, [observation])[0]?.lastKnownAttention).toBeUndefined()
  const recovered = overlayObservedStatuses(fresh.rows, [], result)
  expect(recovered[0]?.lastKnownAttention).toBeUndefined()
  expect(isRecentActivity(recovered[0]!, Date.now())).toBe(false)
})

it('uses the freshest session activity when a live summary lags persisted messages, never fetch time', () => {
  const data = snapshot([session('same', 60 * MINUTE, { status: 'idle' })])
  data.sources[0]!.sessions = [session('same', MINUTE)]
  const projected = mergeOverview(undefined, data)
  expect(projected.rows[0]?.lastActive).toBe((NOW - MINUTE) / 1000)
  expect(isRecentActivity(projected.rows[0]!, Date.now())).toBe(true)
  const missing = mergeOverview(undefined, snapshot([{ id: 'missing timestamp', profile: 'coder' }]))
  expect(isRecentActivity(missing.rows[0]!, Date.now())).toBe(false)
})

it('polling unchanged backend data does not refresh activity or postpone expiry', async () => {
  await mount(snapshot([session('polling idle', 14 * MINUTE, { status: 'idle' })]))
  const rows = overviewCache.state.get().data?.rows
  await act(async () => {
    await vi.advanceTimersByTimeAsync(MINUTE + 1)
  })
  expect(window.hermesDesktop!.getAgentOverview).toHaveBeenCalledTimes(13)
  expect(overviewCache.state.get().data?.rows).toBe(rows)
  expect(rows?.[0]?.lastActive).toBe((NOW - 14 * MINUTE) / 1000)
  expect(titles()).toEqual([])
})

it('keeps source/profile/provider filters intersected with activity while coverage remains unfiltered', async () => {
  const data = snapshot([
    session('target recent', MINUTE, { provider: 'openai' }),
    session('target old', 60 * MINUTE, { provider: 'openai' }),
    session('other provider', MINUTE, { provider: 'anthropic' }),
    session('other profile', MINUTE, { provider: 'openai', profile: 'analyst' })
  ])

  data.sources.push({
    ...data.sources[0]!,
    connectionId: 'elsewhere',
    label: 'Elsewhere',
    live: [session('other source', MINUTE, { provider: 'openai' })]
  })
  await mount(data)

  const choose = (label: string, option: string) => {
    fireEvent.keyDown(screen.getByRole('combobox', { name: label }), { key: 'Enter' })
    fireEvent.click(screen.getByRole('option', { name: option }))
  }

  choose('All sources', 'HQ source')
  choose('All profiles', 'coder')
  choose('All providers', 'openai')
  expect(titles()).toEqual(['target recent'])
  fireEvent.click(screen.getByRole('button', { name: 'All sessions' }))
  expect(titles()).toEqual(['target recent', 'target old'])
  expect(screen.getByTestId('agent-source-coverage').textContent).toContain('Elsewhere')
})

it('pauses the presentation clock while hidden and applies overdue expiry on reveal without fetching', async () => {
  vi.spyOn(overviewCache, 'watch').mockReturnValue(() => {})
  const view = await mount(snapshot([session('hidden idle', 14 * MINUTE, { status: 'idle' })]))
  view.rerender(
    <PaneVisibleContext.Provider value={false}>
      <SessionOverview />
    </PaneVisibleContext.Provider>
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2 * MINUTE)
  })
  expect(titles()).toEqual(['hidden idle'])
  view.rerender(
    <PaneVisibleContext.Provider value={true}>
      <SessionOverview />
    </PaneVisibleContext.Provider>
  )
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
  expect(titles()).toEqual([])
  expect(window.hermesDesktop!.getAgentOverview).not.toHaveBeenCalled()
  view.unmount()
})
