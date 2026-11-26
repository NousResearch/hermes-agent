import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as History from './canonical-group-history'
import type * as RoomStatus from './group-room-status'

// Plan step 3: the room's derived state (outcomes, round, handoffs) is computed once per log/state change,
// and only the working chips tick each second; the whole room does not re-render on the clock.
const request = vi.hoisted(() => vi.fn())
const handoffScans = vi.hoisted(() => ({ count: 0 }))
const cycleRuns = vi.hoisted(() => ({ count: 0 }))
const outcomeRuns = vi.hoisted(() => ({ count: 0 }))
vi.mock('./group-room-status', async importOriginal => {
  const real = await importOriginal<typeof RoomStatus>()

  return {
    ...real,
    roomHandoffs: (...args: Parameters<typeof real.roomHandoffs>) => {
      handoffScans.count++

      return real.roomHandoffs(...args)
    },
    roomCycle: (...args: Parameters<typeof real.roomCycle>) => {
      cycleRuns.count++

      return real.roomCycle(...args)
    },
    memberOutcomes: (...args: Parameters<typeof real.memberOutcomes>) => {
      outcomeRuns.count++

      return real.memberOutcomes(...args)
    }
  }
})
const historyRenders = vi.hoisted(() => ({ count: 0 }))
vi.mock('./canonical-group-history', async importOriginal => {
  const real = await importOriginal<typeof History>()

  return {
    ...real,
    CanonicalGroupHistory: (props: Parameters<typeof real.CanonicalGroupHistory>[0]) => {
      historyRenders.count++

      return real.CanonicalGroupHistory(props)
    }
  }
})
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')

  return {
    ...(await pluginSdkMock(gateway.host)),
    atom,
    useValue: useStore,
    ConfirmDialog,
    useQuery: (await import('@tanstack/react-query')).useQuery,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) =>
      CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: request }
  }
})

import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const members = [{ member_id: 'ops', profile: 'ops', handle: 'ops', display_name: 'Ops' }]
const binding = { connectionId: 'local', profile: 'team', roomId: 'busy', roomName: 'Room' }

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
})
afterEach(() => {
  cleanup()
  request.mockReset()
  localStorage.clear()
  vi.useRealTimers()
  handoffScans.count = 0
  cycleRuns.count = 0
  outcomeRuns.count = 0
  historyRenders.count = 0
})

it('ticks only the working member’s time, never re-deriving the room each second', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  vi.setSystemTime(new Date(1_000_000_000))
  const started = 1_000_000 - 5
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? {
          room: { name: 'Room', members, latest_seq: 1 },
          change_events: true,
          driver_status: {
            working: true,
            member_activity: [{ member_id: 'ops', status: 'running', started_at: started }]
          }
        }
      : method === 'groups.log'
        ? {
            events: [
              {
                seq: 1,
                event_id: 'u1',
                kind: 'message.user',
                actor: { kind: 'user', id: 'u' },
                payload: { text: '@ops go', thread_id: 't' }
              }
            ]
          }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(screen.getAllByText(/5s/).length).toBeGreaterThan(0))
  const scans = handoffScans.count
  const renders = historyRenders.count

  for (let second = 1; second <= 5; second++) {
    await act(async () => {
      vi.advanceTimersByTime(1000)
    })
  }

  expect(screen.getAllByText(/10s/).length).toBeGreaterThan(0) // the chip still counts
  expect(handoffScans.count - scans).toBe(0) // the room's derivations did not rerun
  expect(historyRenders.count - renders).toBe(0) // nor did the room (its transcript) re-render
})

it('turns a quiet member stalled on the clock alone, and back to working on its next frame', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  vi.setSystemTime(new Date(2_000_000_000))
  const now = 2_000_000
  let progressAt = now - 115
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? {
          room: { name: 'Room', members, latest_seq: 0 },
          driver_status: {
            working: true,
            member_activity: [
              { member_id: 'ops', status: 'running', started_at: now - 200, progress_at: progressAt, tool: 'terminal' }
            ]
          }
        }
      : method === 'groups.log'
        ? { events: [] }
        : {}
  )
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  const chip = () => view.container.querySelector('[data-member-status]')?.getAttribute('data-member-status')
  await vi.waitFor(() => expect(chip()).toBe('working'))

  const reads = request.mock.calls.length
  await act(async () => {
    vi.advanceTimersByTime(1000)
  }) // 116 s quiet
  expect(chip()).toBe('working')

  for (let second = 0; second < 4; second++) {
    await act(async () => {
      vi.advanceTimersByTime(1000)
    })
  }
  expect(chip()).toBe('stalled') // 120 s quiet: from the clock, with no new groups.state needed
  expect(screen.getAllByText(/no progress for 2m00s/i).length).toBeGreaterThan(0)
  expect(request.mock.calls.slice(reads).filter(call => call[1] === 'groups.state').length).toBe(0)

  progressAt = now + 5 // the member's next frame arrives with the next read (the real 2 s poll)
  await vi.waitFor(() => expect(chip()).toBe('working'), { timeout: 4000 })
})

it('does not rescan the log for handoffs on each keystroke in the composer', async () => {
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? { room: { name: 'Room', members, latest_seq: 1 }, driver_status: {} }
      : method === 'groups.log'
        ? {
            events: [
              {
                seq: 1,
                event_id: 'u1',
                kind: 'message.user',
                actor: { kind: 'user', id: 'u' },
                payload: { text: 'hi', thread_id: 't' }
              }
            ]
          }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  const input = (await screen.findByRole('textbox')) as HTMLTextAreaElement
  await vi.waitFor(() => expect(input.disabled).toBe(false))
  const scans = handoffScans.count
  const outcomes = outcomeRuns.count

  for (const text of ['h', 'he', 'hel', 'hell', 'hello']) {
    await act(async () => {
      fireEvent.change(input, { target: { value: text } })
    })
  }

  expect(input.value).toBe('hello')
  expect(handoffScans.count - scans).toBe(0)
  expect(outcomeRuns.count - outcomes).toBe(0)
})

it('moves the round strip when only groups.state changes (the log stays the same)', async () => {
  let round = 0
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? {
          room: { name: 'Room', members, latest_seq: 1 },
          driver_status: {
            working: true,
            max_rounds: 3,
            member_activity: [
              { member_id: 'ops', status: 'running', thread_id: 't', round_index: round, started_at: 1 }
            ]
          }
        }
      : method === 'groups.log'
        ? {
            events: [
              {
                seq: 1,
                event_id: 'u1',
                kind: 'message.user',
                actor: { kind: 'user', id: 'u' },
                payload: { text: 'go', thread_id: 't' }
              }
            ]
          }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(screen.getByRole('status').textContent).toContain('Round 1 of 3'))
  round = 1
  await vi.waitFor(() => expect(screen.getByRole('status').textContent).toContain('Round 2 of 3'), { timeout: 4000 })
})

it('re-derives nothing (handoffs, round) when a poll brings the same room back', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? // A fresh object graph every read, as over the wire: same roster and live rows, new arrays.
        {
          room: { name: 'Room', members: members.map(member => ({ ...member })), latest_seq: 1 },
          driver_status: {
            working: true,
            member_activity: [{ member_id: 'ops', status: 'running', started_at: 1, round_index: 1 }]
          }
        }
      : method === 'groups.log'
        ? {
            events: [
              {
                seq: 1,
                event_id: 'u1',
                kind: 'message.user',
                actor: { kind: 'user', id: 'u' },
                payload: { text: 'hi', thread_id: 't' }
              }
            ]
          }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.log')).toBe(true))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(100)
  })
  const scans = handoffScans.count
  const cycles = cycleRuns.count
  const outcomes = outcomeRuns.count
  const reads = request.mock.calls.filter(call => call[1] === 'groups.state').length

  await act(async () => {
    await vi.advanceTimersByTimeAsync(6100)
  }) // three 2 s polls

  expect(request.mock.calls.filter(call => call[1] === 'groups.state').length - reads).toBeGreaterThanOrEqual(3)
  expect(handoffScans.count - scans).toBe(0)
  expect(cycleRuns.count - cycles).toBe(0)
  expect(outcomeRuns.count - outcomes).toBe(0)
})

it('ticks every busy part of one member on one clock: the chip and its details row agree each second', async () => {
  const { MemberStatusChip } = await import('./group-room-ui')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  vi.setSystemTime(new Date(4_000_000_000))
  const labels = CANONICAL_GROUP_LOCALES.en as never
  const status = { kind: 'working' as const, startedAt: 4_000_000 - 3 }
  const view = render(
    <ul>
      <MemberStatusChip avatar={null} labels={labels} name="Ops" status={status} />
    </ul>
  )
  await act(async () => {
    vi.advanceTimersByTime(500)
  }) // the second view mounts mid-second
  view.rerender(
    <ul>
      <MemberStatusChip avatar={null} labels={labels} name="Ops" status={status} />
      <MemberStatusChip avatar={null} labels={labels} name="Ops" status={status} />
    </ul>
  )
  expect(vi.getTimerCount()).toBe(1)

  for (let second = 0; second < 3; second++) {
    await act(async () => {
      vi.advanceTimersByTime(1000)
    })
    const [first, second_] = [...view.container.querySelectorAll('li')].map(chip => chip.textContent)
    expect(second_).toBe(first)
  }

  view.unmount()
  expect(vi.getTimerCount()).toBe(0)
})

it('shows a chip that turns busy with fresh elapsed time before its first busy paint', async () => {
  const { MemberStatusChip } = await import('./group-room-ui')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  vi.setSystemTime(new Date(3_000_000_000))
  const labels = CANONICAL_GROUP_LOCALES.en as never

  const view = render(
    <ul>
      <MemberStatusChip avatar={null} labels={labels} name="Ops" status={{ kind: 'idle' }} />
    </ul>
  )
  vi.setSystemTime(new Date(3_000_090_000)) // 90 s later, with no tick on the idle chip
  view.rerender(
    <ul>
      <MemberStatusChip avatar={null} labels={labels} name="Ops" status={{ kind: 'working', startedAt: 3_000_060 }} />
    </ul>
  )
  // Layout activation refreshes the shared snapshot and this leaf synchronously before browser paint.
  expect(view.container.textContent).toContain('30s')
})

it('turns a requested Stop into stopped when the state confirms it, with the same log and no live rows', async () => {
  const { renderHook } = await import('@testing-library/react')
  const { useCanonicalRoomLive } = await import('./canonical-group-activity')

  const events = [
    { seq: 1, event_id: 'd', kind: 'message.user', payload: { text: 'go' } },
    { seq: 2, kind: 'room.stop_requested', payload: {} }
  ] satisfies History.CanonicalGroupEvent[]

  const driver = (stopping: number) => ({
    working: false,
    member_activity: [],
    pending_actions: [],
    counts: { stopping }
  })
  const room = renderHook(({ state }) => useCanonicalRoomLive(events, state, []), {
    initialProps: { state: driver(1) }
  })
  expect(room.result.current.cycle).toEqual({ kind: 'stopRequested' })
  room.rerender({ state: driver(0) }) // the memo's inputs (log, live rows, focus) are unchanged
  expect(room.result.current.cycle).toEqual({ kind: 'stopped' })
})

it('commits a consistent clock snapshot when busy views render across the stall boundary', async () => {
  const { useShownStatus } = await import('./group-room-ui')
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  const start = 5_000_000
  vi.setSystemTime(new Date((start + 119.9) * 1000))
  let crossed = false

  function Probe({ cross = false }: { cross?: boolean }) {
    const shown = useShownStatus({ kind: 'working', startedAt: start, lastSignAt: start })

    // Simulate the wall clock crossing the threshold during the render, before the second view reads it.
    if (cross && !crossed) {
      crossed = true
      vi.setSystemTime(new Date((start + 120.1) * 1000))
    }

    return <span data-clock-view="">{shown.kind}</span>
  }

  const view = render(
    <>
      <Probe cross />
      <Probe />
    </>
  )
  expect([...view.container.querySelectorAll('[data-clock-view]')].map(node => node.textContent)).toEqual([
    'stalled',
    'stalled'
  ])
})

it('synchronises existing busy views when another mounts across a second boundary', async () => {
  const { MemberStatusChip } = await import('./group-room-ui')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  const start = 5_000_000
  vi.setSystemTime(new Date((start + 119.9) * 1000))

  const props = {
    avatar: null,
    name: 'Ops',
    labels: CANONICAL_GROUP_LOCALES.en as never,
    status: { kind: 'working' as const, startedAt: start, lastSignAt: start }
  }

  const first = render(
    <ul>
      <MemberStatusChip {...props} />
    </ul>
  )

  await act(async () => {
    vi.advanceTimersByTime(200)
  })
  // A separate root leaves the first view untouched by this render; the interval has not fired.
  const second = render(
    <ul>
      <MemberStatusChip {...props} />
    </ul>
  )
  const chips = [first, second].map(view => view.container.querySelector('li'))
  expect(chips.map(chip => chip?.getAttribute('data-member-status'))).toEqual(['stalled', 'stalled'])
  expect(chips[0]?.textContent).toBe(chips[1]?.textContent)
  expect(vi.getTimerCount()).toBe(1)
  first.rerender(
    <ul>
      <MemberStatusChip {...props} status={{ kind: 'idle' }} />
    </ul>
  )
  expect(vi.getTimerCount()).toBe(1)
  second.rerender(
    <ul>
      <MemberStatusChip {...props} status={{ kind: 'queued' }} />
    </ul>
  )
  expect(vi.getTimerCount()).toBe(0)
})

it('keeps existing subscribers on the published second during independent renders', async () => {
  const { MemberStatusChip } = await import('./group-room-ui')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  const start = 5_000_000
  vi.setSystemTime(new Date((start + 119.9) * 1000))

  const props = {
    avatar: null,
    name: 'Ops',
    labels: CANONICAL_GROUP_LOCALES.en as never,
    status: { kind: 'working' as const, startedAt: start, lastSignAt: start }
  }

  const first = render(
    <ul>
      <MemberStatusChip {...props} />
    </ul>
  )
  const second = render(
    <ul>
      <MemberStatusChip {...props} />
    </ul>
  )
  const kinds = () =>
    [first, second].map(view => view.container.querySelector('li')?.getAttribute('data-member-status'))
  await act(async () => {
    vi.advanceTimersByTime(200)
  })
  first.rerender(
    <ul>
      <MemberStatusChip {...props} name="Ops updated" />
    </ul>
  )
  expect(kinds()).toEqual(['working', 'working'])
  await act(async () => {
    vi.advanceTimersByTime(800)
  })
  expect(kinds()).toEqual(['stalled', 'stalled'])
})

it('preserves fractional start and progress times through elapsed, stall and recovery', async () => {
  const { MemberStatusChip } = await import('./group-room-ui')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  vi.setSystemTime(new Date(1_030_900))
  const props = { avatar: null, name: 'Ops', labels: CANONICAL_GROUP_LOCALES.en as never }

  const view = render(
    <ul>
      <MemberStatusChip {...props} status={{ kind: 'working', startedAt: 1000.1, lastSignAt: 1000.1 }} />
    </ul>
  )
  expect(view.container.textContent).toContain('30s')
  await act(async () => {
    vi.advanceTimersByTime(90_000)
  })
  expect(view.container.querySelector('li')?.getAttribute('data-member-status')).toBe('stalled')
  expect(view.container.textContent).toContain('2m00s')
  view.rerender(
    <ul>
      <MemberStatusChip {...props} status={{ kind: 'working', startedAt: 1000.1, lastSignAt: 1120.9 }} />
    </ul>
  )
  expect(view.container.querySelector('li')?.getAttribute('data-member-status')).toBe('working')
  expect(view.container.textContent).toContain('2m00s')
})
