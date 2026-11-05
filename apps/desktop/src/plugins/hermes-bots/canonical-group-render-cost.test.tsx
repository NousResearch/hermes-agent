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
vi.mock('./group-room-status', async importOriginal => {
  const real = await importOriginal<typeof RoomStatus>()

  return { ...real, roomHandoffs: (...args: Parameters<typeof real.roomHandoffs>) => { handoffScans.count++

    return real.roomHandoffs(...args) } }
})
const historyRenders = vi.hoisted(() => ({ count: 0 }))
vi.mock('./canonical-group-history', async importOriginal => {
  const real = await importOriginal<typeof History>()

  return { ...real, CanonicalGroupHistory: (props: Parameters<typeof real.CanonicalGroupHistory>[0]) => { historyRenders.count++

    return real.CanonicalGroupHistory(props) } }
})
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')

  return { ...await pluginSdkMock(gateway.host), atom, useValue: useStore, ConfirmDialog,
    useQuery: (await import('@tanstack/react-query')).useQuery,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: request } }
})

import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const members = [{ member_id: 'ops', profile: 'ops', handle: 'ops', display_name: 'Ops' }]
const binding = { connectionId: 'local', profile: 'team', roomId: 'busy', roomName: 'Room' }

beforeEach(() => { Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined }) })
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear(); vi.useRealTimers(); handoffScans.count = 0; historyRenders.count = 0 })

it('ticks only the working member’s time, never re-deriving the room each second', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  vi.setSystemTime(new Date(1_000_000_000))
  const started = 1_000_000 - 5
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room', members, latest_seq: 1 }, change_events: true,
        driver_status: { working: true, member_activity: [{ member_id: 'ops', status: 'running', started_at: started }] } }
    : method === 'groups.log'
      ? { events: [{ seq: 1, event_id: 'u1', kind: 'message.user', actor: { kind: 'user', id: 'u' }, payload: { text: '@ops go', thread_id: 't' } }] }
      : {})
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(screen.getAllByText(/5s/).length).toBeGreaterThan(0))
  const scans = handoffScans.count
  const renders = historyRenders.count

  for (let second = 1; second <= 5; second++) {
    await act(async () => { vi.advanceTimersByTime(1000) })
  }

  expect(screen.getAllByText(/10s/).length).toBeGreaterThan(0)  // the chip still counts
  expect(handoffScans.count - scans).toBe(0)                    // the room's derivations did not rerun
  expect(historyRenders.count - renders).toBe(0)                // nor did the room (its transcript) re-render
})

it('turns a quiet member stalled on the clock alone, and back to working on its next frame', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
  vi.setSystemTime(new Date(2_000_000_000))
  const now = 2_000_000
  let progressAt = now - 115
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room', members, latest_seq: 0 },
        driver_status: { working: true, member_activity: [{ member_id: 'ops', status: 'running', started_at: now - 200, progress_at: progressAt, tool: 'terminal' }] } }
    : method === 'groups.log' ? { events: [] } : {})
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  const chip = () => view.container.querySelector('[data-member-status]')?.getAttribute('data-member-status')
  await vi.waitFor(() => expect(chip()).toBe('working'))

  const reads = request.mock.calls.length
  await act(async () => { vi.advanceTimersByTime(1000) })  // 116 s quiet
  expect(chip()).toBe('working')

  for (let second = 0; second < 4; second++) {await act(async () => { vi.advanceTimersByTime(1000) })}
  expect(chip()).toBe('stalled')  // 120 s quiet: from the clock, with no new groups.state needed
  expect(screen.getAllByText(/no progress for 2m00s/i).length).toBeGreaterThan(0)
  expect(request.mock.calls.slice(reads).filter(call => call[1] === 'groups.state').length).toBe(0)

  progressAt = now + 5  // the member's next frame arrives with the next read (the real 2 s poll)
  await vi.waitFor(() => expect(chip()).toBe('working'), { timeout: 4000 })
})

it('does not rescan the log for handoffs on each keystroke in the composer', async () => {
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room', members, latest_seq: 1 }, driver_status: {} }
    : method === 'groups.log'
      ? { events: [{ seq: 1, event_id: 'u1', kind: 'message.user', actor: { kind: 'user', id: 'u' }, payload: { text: 'hi', thread_id: 't' } }] }
      : {})
  render(<CanonicalGroupWorkspace binding={binding} />)
  const input = await screen.findByRole('textbox') as HTMLTextAreaElement
  await vi.waitFor(() => expect(input.disabled).toBe(false))
  const scans = handoffScans.count

  for (const text of ['h', 'he', 'hel', 'hell', 'hello']) {
    await act(async () => { fireEvent.change(input, { target: { value: text } }) })
  }

  expect(input.value).toBe('hello')
  expect(handoffScans.count - scans).toBe(0)
})


it('moves the round strip when only groups.state changes (the log stays the same)', async () => {
  let round = 0
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room', members, latest_seq: 1 }, driver_status: { working: true, max_rounds: 3,
        member_activity: [{ member_id: 'ops', status: 'running', thread_id: 't', round_index: round, started_at: 1 }] } }
    : method === 'groups.log'
      ? { events: [{ seq: 1, event_id: 'u1', kind: 'message.user', actor: { kind: 'user', id: 'u' }, payload: { text: 'go', thread_id: 't' } }] }
      : {})
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(screen.getByRole('status').textContent).toContain('Round 1 of 3'))
  round = 1
  await vi.waitFor(() => expect(screen.getByRole('status').textContent).toContain('Round 2 of 3'), { timeout: 4000 })
})


it('does not rescan the log for handoffs when a poll brings the same room back', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    // A fresh object graph every read, as over the wire: same roster, new arrays.
    ? { room: { name: 'Room', members: members.map(member => ({ ...member })), latest_seq: 1 }, driver_status: {} }
    : method === 'groups.log'
      ? { events: [{ seq: 1, event_id: 'u1', kind: 'message.user', actor: { kind: 'user', id: 'u' }, payload: { text: 'hi', thread_id: 't' } }] }
      : {})
  render(<CanonicalGroupWorkspace binding={binding} />)
  await vi.waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.log')).toBe(true))
  await act(async () => { await vi.advanceTimersByTimeAsync(100) })
  const scans = handoffScans.count
  const reads = request.mock.calls.filter(call => call[1] === 'groups.state').length

  await act(async () => { await vi.advanceTimersByTimeAsync(6100) })  // three 2 s polls

  expect(request.mock.calls.filter(call => call[1] === 'groups.state').length - reads).toBeGreaterThanOrEqual(3)
  expect(handoffScans.count - scans).toBe(0)
})
