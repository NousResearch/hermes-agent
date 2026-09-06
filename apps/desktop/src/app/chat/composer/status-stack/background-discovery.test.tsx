import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $backgroundStatusBySession, resetBackgroundPollingGuard } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-discovery'

// Longer than any plausible idle discovery cadence, so the test pins "the stack
// keeps asking while empty" rather than one tuned interval value.
const PAST_IDLE_TICK_MS = 60_000

// The fast running cadence is a safety net for silent exits; a discovery tick
// must stay well below that rate.
const RUNNING_POLL_WINDOW_MS = 20_000

function renderStack(sessionId: null | string = SID) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={sessionId} />
      </I18nProvider>
    </MemoryRouter>
  )
}

// A background process spawned through ANOTHER gateway client (a Telegram
// `terminal(background=true)` job) reaches this window only through the shared
// process registry — nothing broadcasts an event here. The poll used to be armed
// only while a running row was already on screen, so with an empty stack nothing
// ever asked again and the first row never appeared.
describe('ComposerStatusStack background discovery', () => {
  const listCalls = (request: ReturnType<typeof vi.fn>, sid = SID) =>
    request.mock.calls.filter(([method, params]) => method === 'process.list' && params?.session_id === sid).length

  beforeEach(() => {
    vi.useFakeTimers()
    resetBackgroundPollingGuard()
    $backgroundStatusBySession.set({})
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    $gateway.set(null as never)
    $backgroundStatusBySession.set({})
    resetBackgroundPollingGuard()
  })

  it('discovers a process that appears on a later poll, with no remount and no gateway event', async () => {
    let processes: Record<string, unknown>[] = []
    const request = vi.fn(async (method: string) => (method === 'process.list' ? { processes } : {}))

    $gateway.set({ request } as never)
    renderStack()

    // Mount seed: the registry is still empty, exactly like the reported bug
    // (the Desktop tile opened before the Telegram job started).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect($backgroundStatusBySession.get()[SID]).toBeUndefined()

    // The job starts elsewhere. Nothing notifies this window.
    processes = [{ command: 'sleep 600', peer: true, session_id: 'proc_e99debacd2eb', status: 'running' }]

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS)
    })

    expect($backgroundStatusBySession.get()[SID]).toMatchObject([{ id: 'proc_e99debacd2eb', state: 'running' }])
  })

  it('polls the empty stack slowly and speeds up once a row is running', async () => {
    let processes: Record<string, unknown>[] = []
    const request = vi.fn(async (method: string) => (method === 'process.list' ? { processes } : {}))

    $gateway.set({ request } as never)
    renderStack()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(RUNNING_POLL_WINDOW_MS)
    })

    const idleCalls = listCalls(request)

    processes = [{ command: 'sleep 600', session_id: 'proc_running', status: 'running' }]

    await act(async () => {
      await vi.advanceTimersByTimeAsync(RUNNING_POLL_WINDOW_MS)
    })

    // Same wall-clock window, many more polls: the existing fast cadence for a
    // visible running row is preserved, and idle discovery is strictly slower.
    expect(listCalls(request) - idleCalls).toBeGreaterThan(idleCalls)
  })

  it('never stacks a second timer for the same session and drops it on switch', async () => {
    const request = vi.fn(async (method: string) => (method === 'process.list' ? { processes: [] } : {}))

    $gateway.set({ request } as never)

    const view = renderStack()
    // A parent re-render with the SAME session must not arm another timer.
    view.rerender(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerStatusStack queue={null} sessionId={SID} />
        </I18nProvider>
      </MemoryRouter>
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    // One mount seed for the pair of renders, not one per render.
    const seed = listCalls(request)

    expect(seed).toBe(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS)
    })

    const ticksForOneStack = listCalls(request) - seed

    expect(ticksForOneStack).toBeGreaterThan(0)

    const twin = renderStack()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    const beforeSecondWindow = listCalls(request)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS)
    })

    // Two mounted stacks tick exactly twice as often: one timer each, never two.
    expect(listCalls(request) - beforeSecondWindow).toBe(ticksForOneStack * 2)

    twin.unmount()
    view.rerender(
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ComposerStatusStack queue={null} sessionId="sess-other" />
        </I18nProvider>
      </MemoryRouter>
    )

    const frozen = listCalls(request)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS)
    })

    expect(listCalls(request)).toBe(frozen)
    expect(listCalls(request, 'sess-other')).toBeGreaterThan(0)
  })

  it('stops polling after unmount so no timer leaks', async () => {
    const request = vi.fn(async (method: string) => (method === 'process.list' ? { processes: [] } : {}))

    $gateway.set({ request } as never)

    const view = renderStack()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS)
    })

    const whileMounted = listCalls(request)

    expect(whileMounted).toBeGreaterThan(1)

    view.unmount()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS * 2)
    })

    expect(listCalls(request)).toBe(whileMounted)
  })

  it('drops peer mirrors when the stack stops watching a session, keeping local rows', async () => {
    const processes = [
      { command: 'local job', session_id: 'proc_mine', status: 'running' },
      { command: 'their job', peer: true, session_id: 'proc_theirs', status: 'running' }
    ]

    const request = vi.fn(async (method: string) => (method === 'process.list' ? { processes } : {}))

    $gateway.set({ request } as never)

    const view = renderStack()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect($backgroundStatusBySession.get()[SID]?.map(i => i.id)).toEqual(['proc_mine', 'proc_theirs'])

    // Crossing into the fast running cadence re-runs the poll effect; that must
    // not be mistaken for "no longer watching".
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS)
    })

    expect($backgroundStatusBySession.get()[SID]?.map(i => i.id)).toEqual(['proc_mine', 'proc_theirs'])

    view.unmount()

    // Nothing refreshes this session now, so the mirror would claim "running"
    // forever; the locally-owned row stays, still fed by this window's events.
    expect($backgroundStatusBySession.get()[SID]?.map(i => i.id)).toEqual(['proc_mine'])
  })

  it('never arms discovery against a runtime the gateway has latched gone', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'process.list') {
        throw new Error('session not found')
      }

      return {}
    })

    $gateway.set({ request } as never)
    renderStack()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(PAST_IDLE_TICK_MS * 3)
    })

    // The mount seed 4001s once and latches; discovery must not become the
    // storm the latch exists to stop.
    expect(listCalls(request)).toBe(1)
  })
})
