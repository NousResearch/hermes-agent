import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps, ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())

// A native <select> stands in for the Radix one: same value/onValueChange contract.
vi.mock('@hermes/plugin-sdk', async () => {
  const { en } = await import('@/i18n/en')

  return {
    ...(await (await import('./group-test-utils')).pluginSdkMock({})),
    atom: (await import('nanostores')).atom,
    host: { requestProfile: request },
    randomUuid: () => 'uuid-1',
    useI18n: () => ({ locale: 'en', t: en }),
    Button: (props: ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>,
    MessageTextContent: ({ text }: { text: string }) => <div>{text}</div>,
    Select: ({
      children,
      onValueChange,
      value,
      disabled
    }: {
      children: ReactNode
      onValueChange: (v: string) => void
      value: string
      disabled?: boolean
    }) => (
      <select
        aria-label="routing"
        disabled={disabled}
        onChange={event => onValueChange(event.target.value)}
        value={value}
      >
        {children}
      </select>
    ),
    SelectContent: ({ children }: { children: ReactNode }) => <>{children}</>,
    SelectItem: ({ children, value }: { children: ReactNode; value: string }) => (
      <option value={value}>{children}</option>
    ),
    SelectTrigger: () => null,
    SelectValue: () => null
  }
})
vi.mock('./group-chat-parts', () => ({
  GroupMentionInput: ({
    onChange,
    ...props
  }: Omit<ComponentProps<'textarea'>, 'onChange'> & { onChange: (value: string) => void }) => (
    <textarea
      aria-label={props['aria-label']}
      disabled={props.disabled}
      onChange={event => onChange(event.target.value)}
      value={props.value}
    />
  )
}))
vi.mock('./canonical-group-labels', async () => {
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return {
    useCanonicalGroupLabels: () => ({
      ...CANONICAL_GROUP_LOCALES.en,
      back: 'Back',
      refresh: 'Refresh',
      retry: 'Retry',
      send: 'Send',
      stop: 'Stop',
      download: 'Download',
      discard: 'Discard',
      cancel: 'Cancel',
      save: 'Save'
    })
  }
})

import type { CanonicalGroupEvent } from './canonical-group-history'
import { CanonicalGroupHistory } from './canonical-group-history'
import { CanonicalGroupLeadSetting } from './canonical-group-lead'
import { CanonicalGroupRoutedRow, pausedDiscussions } from './canonical-group-routed'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const members = [
  { member_id: 'm-ada', handle: 'ada', display_name: 'Ada', profile: 'ada' },
  { member_id: 'm-grace', handle: 'grace', profile: 'grace' }
]

const binding = { connectionId: 'local', profile: 'team', roomId: 'room-1' }
const D = 'evt-user-1'
const T = 'thread-1'

const ev = (
  seq: number,
  kind: string,
  payload: Record<string, unknown> = {},
  actor?: CanonicalGroupEvent['actor']
): CanonicalGroupEvent => ({ seq, event_id: `e${seq}`, room_id: binding.roomId, kind, payload, actor })

const userMsg = (seq: number, thread = T) =>
  ev(seq, 'message.user', { text: 'hi', thread_id: thread }, { kind: 'user', id: 'desktop' })
const memberMsg = (seq: number, discussion = D) =>
  ev(
    seq,
    'message.member',
    { text: `reply ${seq}`, discussion_event_id: discussion, thread_id: T },
    { kind: 'member', id: 'm-ada' }
  )
const bounded = (seq: number) =>
  ev(seq, 'room.activity', { status: 'bounded', reason_code: 'max_messages', thread_id: T, discussion_event_id: D })
const continued = (seq: number, grant = 1) =>
  ev(seq, 'room.continued', { discussion_event_id: D, thread_id: T, grant }, { kind: 'gateway' })
const pausedLog = [userMsg(1), memberMsg(2), memberMsg(3), memberMsg(4, 'other'), bounded(5)]

const renderHistory = (events: CanonicalGroupEvent[], pause?: Record<string, unknown>) =>
  render(<CanonicalGroupHistory binding={binding} events={events} members={members} pause={pause} />)

afterEach(() => {
  cleanup()
  request.mockReset()
})

it('names routed members by their room names, never their ids, and skips unknown ids', () => {
  render(
    <CanonicalGroupRoutedRow
      event={ev(2, 'room.routed', {
        discussion_event_id: D,
        thread_id: T,
        member_ids: ['m-ada', 'm-gone', 'm-grace'],
        everyone: false,
        fallback: false
      })}
      members={members}
    />
  )
  expect(screen.getByTestId('room-routed').textContent).toBe('Assigned to Ada, grace')
})

it('says everyone was assigned, and says plainly when the router fell back to everyone', () => {
  const { rerender } = render(
    <CanonicalGroupRoutedRow
      event={ev(2, 'room.routed', { member_ids: [], everyone: true, fallback: false })}
      members={members}
    />
  )
  expect(screen.getByTestId('room-routed').textContent).toBe('Assigned to everyone')
  rerender(
    <CanonicalGroupRoutedRow
      event={ev(2, 'room.routed', { member_ids: [], everyone: true, fallback: true })}
      members={members}
    />
  )
  expect(screen.getByTestId('room-routed').textContent).toBe("Couldn't choose who answers, so everyone will")
})

it('keeps the legacy limit notice when the runtime reports no max_continues', () => {
  renderHistory(pausedLog, { onContinue: vi.fn(), onStop: vi.fn() })
  expect(screen.queryByTestId('room-paused')).toBeNull()
  expect(screen.getByText('Group work limit reached')).toBeTruthy()
})

it("counts only this discussion's member messages before the pause and offers Continue and Stop", () => {
  const onContinue = vi.fn()
  const onStop = vi.fn()
  renderHistory(pausedLog, { maxContinues: 5, onContinue, onStop })
  expect(screen.getByTestId('room-paused').textContent).toContain('Paused after 2 messages')
  fireEvent.click(screen.getByTestId('room-continue'))
  expect(onContinue).toHaveBeenCalledWith(D)
  fireEvent.click(screen.getByTestId('room-paused-stop'))
  expect(onStop).toHaveBeenCalledTimes(1)
})

it("keeps the paused row's Stop usable while a Continue is in flight, and both idle while a Stop is", () => {
  const { rerender } = renderHistory(pausedLog, { busy: true, maxContinues: 5, onContinue: vi.fn(), onStop: vi.fn() })
  expect((screen.getByTestId('room-continue') as HTMLButtonElement).disabled).toBe(true)
  expect((screen.getByTestId('room-paused-stop') as HTMLButtonElement).disabled).toBe(false)
  rerender(
    <CanonicalGroupHistory
      binding={binding}
      events={pausedLog}
      members={members}
      pause={{ maxContinues: 5, onContinue: vi.fn(), onStop: vi.fn(), stopping: true }}
    />
  )
  expect((screen.getByTestId('room-continue') as HTMLButtonElement).disabled).toBe(true)
  expect((screen.getByTestId('room-paused-stop') as HTMLButtonElement).disabled).toBe(true)
})

it.each([
  ['after it was continued', [...pausedLog, continued(6)]],
  ['after a stop', [...pausedLog, ev(6, 'room.stop_requested')]],
  ['after a new user message in the thread', [...pausedLog, userMsg(6)]]
])('drops the buttons %s', (_case, events) => {
  renderHistory(events, { maxContinues: 5, onContinue: vi.fn(), onStop: vi.fn() })
  expect(screen.getByTestId('room-paused')).toBeTruthy()
  expect(screen.queryByTestId('room-continue')).toBeNull()
  expect(screen.queryByTestId('room-paused-stop')).toBeNull()
})

it('shows the continued row and only the latest pause stays actionable', () => {
  renderHistory([...pausedLog, continued(6), memberMsg(7), bounded(8)], {
    maxContinues: 5,
    onContinue: vi.fn(),
    onStop: vi.fn()
  })
  expect(screen.getByTestId('room-continued').textContent).toBe('Continued')
  expect(screen.getAllByTestId('room-paused')).toHaveLength(2)
  expect(screen.getAllByTestId('room-continue')).toHaveLength(1)
  expect(screen.getAllByTestId('room-paused')[1].textContent).toContain('Paused after 3 messages')
})

it('offers no Continue once the discussion used every continue', () => {
  renderHistory([...pausedLog, continued(6), bounded(7)], { maxContinues: 1, onContinue: vi.fn(), onStop: vi.fn() })
  expect(screen.queryByTestId('room-continue')).toBeNull()
})

it('a Stop that landed before a late cap notice still retires the pause', () => {
  const source = { ...userMsg(1), event_id: D }
  renderHistory([source, memberMsg(2), memberMsg(3), ev(4, 'room.stop_requested'), bounded(5)], {
    maxContinues: 5,
    onContinue: vi.fn(),
    onStop: vi.fn()
  })
  expect(screen.getByTestId('room-paused')).toBeTruthy()
  expect(screen.queryByTestId('room-continue')).toBeNull()
})

it('a newer message in the thread that landed before a late cap notice retires the pause', () => {
  const source = { ...userMsg(1), event_id: D }
  renderHistory([source, memberMsg(2), memberMsg(3), userMsg(4), bounded(5)], {
    maxContinues: 5,
    onContinue: vi.fn(),
    onStop: vi.fn()
  })
  expect(screen.getByTestId('room-paused')).toBeTruthy()
  expect(screen.queryByTestId('room-continue')).toBeNull()
})

it("an unrelated thread's user message does not retire the pause", () => {
  renderHistory([...pausedLog, userMsg(6, 'thread-2')], { maxContinues: 5, onContinue: vi.fn(), onStop: vi.fn() })
  expect(screen.getByTestId('room-continue')).toBeTruthy()
})

it('reads every pause of a long log in one pass with the same answers per row', () => {
  const other = (seq: number) =>
    ev(seq, 'room.activity', { status: 'bounded', thread_id: 'thread-2', discussion_event_id: 'evt-2' })
  const log = [
    ...pausedLog,
    continued(6),
    memberMsg(7),
    bounded(8),
    userMsg(9, 'thread-2'),
    memberMsg(10, 'evt-2'),
    other(11),
    continued(12, 2),
    bounded(13)
  ]
  const reads = new Map(pausedDiscussions(log, 5))
  expect([...reads.entries()]).toEqual([
    [5, { count: 2, actionable: false }],
    [8, { count: 3, actionable: false }],
    [11, { count: 1, actionable: true }],
    [13, { count: 3, actionable: true }]
  ])
  // Shuffled input (a merged page) reads the same; a spent Continue budget retires the live pause.
  expect(new Map(pausedDiscussions([...log].reverse(), 5))).toEqual(reads)
  expect(pausedDiscussions(log, 2).get(13)).toEqual({ count: 3, actionable: false })
})

it('offers automatic routing only when the runtime lists it, and saves it with no lead', () => {
  const onChange = vi.fn()
  const draft = { purpose: '', notes: '', routing: 'everyone' as const, lead_member_id: '' }
  const { rerender } = render(<CanonicalGroupLeadSetting draft={draft} members={members} onChange={onChange} />)
  expect(screen.queryByRole('option', { name: 'Pick who answers automatically' })).toBeNull()
  rerender(<CanonicalGroupLeadSetting autoRouting draft={draft} members={members} onChange={onChange} />)
  fireEvent.change(screen.getByLabelText('routing'), { target: { value: 'auto' } })
  expect(onChange).toHaveBeenCalledWith({ ...draft, routing: 'auto', lead_member_id: '' })
  rerender(
    <CanonicalGroupLeadSetting
      autoRouting
      draft={{ ...draft, routing: 'auto' }}
      members={members}
      onChange={onChange}
    />
  )
  expect(screen.getByText(/A quick model call picks who a message is for/)).toBeTruthy()
})

it('Continue in the room calls groups.continue with the paused discussion', async () => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {
      return {
        room: { name: 'Room', members },
        driver_status: { max_continues: 5, routing_modes: ['everyone', 'lead', 'auto'] }
      }
    }

    if (method === 'groups.log') {
      return { events: pausedLog }
    }

    return { continued: true }
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  fireEvent.click(await screen.findByTestId('room-continue'))
  await waitFor(() => expect(request.mock.calls.some(([, method]) => method === 'groups.continue')).toBe(true))
  const call = request.mock.calls.find(([, method]) => method === 'groups.continue')!
  expect(call[2]).toMatchObject({ room_id: binding.roomId, discussion_event_id: D })
})
