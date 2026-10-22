import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const { captureGroupRequests } = await import('./canonical-group-contract-test-utils')

  return { ...await pluginSdkMock(gateway.host), atom, useValue: useStore,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    host: { ...gateway.host, requestProfile: captureGroupRequests(request).request } }
})
import { CANONICAL_GROUP_CAPABILITIES, STANDALONE_GROUP_CAPABILITIES } from './canonical-group-contract-test-utils'
import { registerCanonicalGroup } from './canonical-group-registry'
import { prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { GroupChatWorkspace } from './group-chat-view'
const originalDesktop = window.hermesDesktop
beforeEach(() => { Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined }) })
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear(); window.hermesDesktop = originalDesktop; vi.useRealTimers() })
// What gateway/session_group_controls.py answers for groups.send.
const receipt = (params: { event_id?: string }) => ({ event: {}, client_event_id: params.event_id, accepted: true, driver_started: true })

it('preserves the unsent draft when only the room display name changes', async () => {
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : {})
  const binding = { connectionId: 'local', profile: 'team', roomId: 'rename', roomName: 'Room' }
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Keep this draft' } })
  view.rerender(<CanonicalGroupWorkspace binding={{ ...binding, roomName: 'Renamed room' }} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Keep this draft')

  view.rerender(<CanonicalGroupWorkspace binding={{ ...binding, roomId: 'another-room' }} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('')
})

it('restores a frozen send after remount and retires only its acknowledged exact retry', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'restore' }
  const entry = await prepareCanonicalGroupSend(binding, { text: 'Original', attachments: [{ path: '/owner/image.png', mime_type: 'image/png' }] })
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {throw new Error('lost ACK')}

    return {}
  })
  const first = render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('Original'))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(true)
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByText('lost ACK')
  // The request was counted before it left, so this intent stays uncertain until a receipt.
  expect(await readCanonicalGroupSend(binding)).toEqual({ ...entry, attempts: 1 })
  first.unmount()
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByRole('button', { name: 'Retry' })
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : receipt(params))
  await waitFor(() => expect((screen.getByRole('button', { name: 'Retry' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  const sends = request.mock.calls.filter(c => c[1] === 'groups.send')
  expect(sends).toHaveLength(2)

  for (const call of sends) {
    expect(call[0]).toMatchObject({ connectionId: binding.connectionId, targetProfile: binding.profile })
    expect(call[2]).toEqual({ ...entry.params, profile: binding.profile })
  }
})

it('blocks Send until journal restore and durable preparation complete', async () => {
  let releaseRead!: (value: string) => void
  let releaseWrite!: () => void
  const journal: Record<string, unknown> = {}

  const native = {
    read: vi.fn().mockImplementationOnce(() => new Promise<string>(resolve => { releaseRead = resolve }))
      .mockImplementation(async () => JSON.stringify(journal)),
    compareAndSet: vi.fn(async (key: string, _expected: string | null, value: string | null) => {
      await new Promise<void>(resolve => { releaseWrite = resolve })

      if (value === null) {delete journal[key]} else {journal[key] = JSON.parse(value)}

      return { applied: true, current: value }
    })
  }

  window.hermesDesktop = { preparedSubmissions: native } as unknown as typeof window.hermesDesktop
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : receipt(params))
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'new' }} />)
  await screen.findByText('Room')
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'New text' } })
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true)
  releaseRead('{}')
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'New text' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(native.compareAndSet).toHaveBeenCalled())
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  releaseWrite()
  // The dispatch is recorded durably before the request leaves.
  await waitFor(() => expect(native.compareAndSet).toHaveBeenCalledTimes(2))
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  releaseWrite()
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  await waitFor(() => expect(native.compareAndSet).toHaveBeenCalledTimes(3))
  releaseWrite()
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(''))
})

it('captures exact pending attempts through confirmation and never retargets or retries unknown work', async () => {
  const action = { kind: 'discard', member_id: 'worker', task_id: 'old-task', execution_generation: 7 }
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { pending_actions: [action] } }}

    if (method === 'groups.log') {return { events: [], has_more: false }}

    if (method === 'groups.discard') {throw new Error('stale_attempt')}

    return {}
  })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'ack-room' }} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Discard' }))
  expect(screen.getByText(/Side effects may already have occurred/)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  action.execution_generation = 8
  fireEvent.click(screen.getByRole('button', { name: 'Confirm discard' }))
  await screen.findByText('stale_attempt')
  const call = request.mock.calls.find(c => c[1] === 'groups.discard')!
  expect(call[0]).toMatchObject({ connectionId: 'remote', targetProfile: 'team' })
  expect(call[2]).toEqual({ room_id: 'ack-room', member_id: 'worker', task_id: 'old-task', execution_generation: 7, profile: 'team' })
  expect(request.mock.calls.every(c => c[1].startsWith('groups.'))).toBe(true)
})

it('reads back retry on the same authority and sends only through the group driver', async () => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { pending_actions: [{ kind: 'retry', member_id: 'w', task_id: 't', execution_generation: 2 }] } }}

    if (method === 'groups.log') {return { events: [{ seq: 1, kind: 'message', payload: { text: 'Owner reply' } }], has_more: false }}

    return {}
  })
  const group = registerCanonicalGroup({ connectionId: 'local', profile: 'default' }, { room_id: 'r', name: 'Room', members: [] })
  render(<GroupChatWorkspace group={group} members={[]} origin="key" />)
  fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(request.mock.calls.filter(c => c[1] === 'groups.state').length).toBeGreaterThan(1))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Hello' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  expect(screen.getByText('Owner reply')).toBeTruthy()
  expect(request.mock.calls.every(c => c[1].startsWith('groups.'))).toBe(true)
})

// Review 5475590057 P3: another window won the room's one intent slot after this one restored.
// Send converges on that durable winner (4149282e49) but this window's typed text is kept, not lost.
it('a Send that converges on another window\'s pending intent keeps this window\'s draft', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'two-windows' }
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : receipt(params))
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('Room')
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  const other = await prepareCanonicalGroupSend(binding, { text: 'from the other window', attachments: [] })
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'typed here' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  expect(request.mock.calls.filter(c => c[1] === 'groups.send').map(c => c[2])).toEqual([{ ...other.params, profile: 'team' }])
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('typed here'))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false)
})

it('shows which bots are working, and still says the group is working on a gateway without member activity', async () => {
  const members = [{ member_id: 'ops', profile: 'ops', handle: 'ops', display_name: 'Ops' },
    { member_id: 'qa', profile: 'qa', handle: 'qa', display_name: 'QA' }]

  let driver: Record<string, unknown> = { working: true, member_activity: [
    { member_id: 'qa', status: 'queued' }, { member_id: 'ops', status: 'running' }, { member_id: 'ops', status: 'queued' }] }

  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room', members }, driver_status: driver } : method === 'groups.log' ? { events: [] } : {})
  const view = render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'busy' }} />)
  await waitFor(() => expect(screen.getByRole('status').textContent).toBe('Working: Ops'))
  const phase = (name: string) => screen.getByText(name).closest('li')?.getAttribute('data-phase')
  expect([phase('Ops'), phase('QA')]).toEqual(['working', 'queued'])

  // The same mounted room clears both indicators on its next poll once the turn settles.
  driver = { working: false, member_activity: [] }
  await waitFor(() => expect(screen.queryByRole('status')).toBeNull(), { timeout: 5_000 })
  expect([phase('Ops'), phase('QA')]).toEqual(['idle', 'idle'])

  view.unmount()
  driver = { working: true }
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'older' }} />)
  await waitFor(() => expect(screen.getByRole('status').textContent).toBe('The group is working…'))
})

it.each([
  ['the gateway', CANONICAL_GROUP_CAPABILITIES, true],
  ['the standalone server', STANDALONE_GROUP_CAPABILITIES, false]
])('offers Discard and attachments only where %s lists them', async (_surface, capabilities, offered) => {
  const actions = [{ kind: 'discard', member_id: 'ops', task_id: 'unknown', execution_generation: 2 },
    { kind: 'retry', member_id: 'qa', task_id: 'late', execution_generation: 1 }]

  const events = [{ seq: 1, room_id: 'surfaces', event_id: 'file-event', kind: 'message.member', actor: { kind: 'member', id: 'ops' },
    payload: { attachments: [{ attachment_id: 'att_1', kind: 'file', name: 'notes.txt', mime: 'text/plain' }] } }]

  let answer!: (value: unknown) => void
  request.mockImplementation(async (_route, method) => method === 'groups.capabilities'
    ? new Promise(resolve => { answer = resolve })
    : method === 'groups.state' ? { room: { name: 'Room' }, driver_status: { pending_actions: actions } }
      : method === 'groups.log' ? { events } : {})
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'surfaces' }} />)
  await screen.findByText('notes.txt')
  // Until the gateway says what it offers, no optional control is shown.
  expect(screen.queryByRole('button', { name: 'Discard' })).toBeNull()
  expect(screen.queryByRole('button', { name: 'Attach files' })).toBeNull()

  answer(capabilities)
  await screen.findByRole('button', { name: 'Retry' })
  expect(Boolean(screen.queryByRole('button', { name: 'Discard' }))).toBe(offered)
  expect(Boolean(screen.queryByRole('button', { name: 'Attach files' }))).toBe(offered)
  expect((screen.getByRole('button', { name: 'Download' }) as HTMLButtonElement).disabled).toBe(!offered)
})

it('reads only new log events, and the whole log again when the room authority changes', async () => {
  // The room polls every two seconds; the test moves the clock instead of waiting.
  vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['setTimeout', 'clearTimeout'] })
  const nextPoll = () => act(() => vi.advanceTimersByTimeAsync(2_000))
  const message = (seq: number, text: string) => ({ seq, kind: 'message.member', actor: { kind: 'member', id: 'ops' }, payload: { text } })
  let epoch = 1
  let log = [message(1, 'First reply')]
  request.mockImplementation(async (_route, method, params) => method === 'groups.state'
    ? { room: { name: 'Room', authority_epoch: epoch, latest_seq: log.at(-1)!.seq }, driver_status: {} }
    : method === 'groups.log' ? { events: log.filter(event => event.seq > params.since_seq), has_more: false } : {})
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'long' }} />)
  await screen.findByText('First reply')
  log = [...log, message(2, 'Second reply')]
  await nextPoll()
  await screen.findByText('Second reply')
  const since = () => request.mock.calls.filter(call => call[1] === 'groups.log').map(call => call[2].since_seq)
  expect(since()).toEqual([0, 1])
  expect(screen.getAllByText(/reply$/).map(node => node.textContent)).toEqual(['First reply', 'Second reply'])

  // The new authority's log is as long as the one read: only its epoch says to read it all again.
  epoch = 2
  log = [message(1, 'Reply after a takeover'), message(2, 'Second reply after a takeover')]
  await nextPoll()
  await screen.findByText('Reply after a takeover')
  expect(since().at(-1)).toBe(0)
  expect(screen.queryByText('First reply')).toBeNull()
})

it('says what the room waits on beside the live work, never instead of it', async () => {
  const members = [{ member_id: 'ops', profile: 'ops', handle: 'ops', display_name: 'Ops' },
    { member_id: 'qa', profile: 'qa', handle: 'qa', display_name: 'QA' }]

  let driver: Record<string, unknown> | undefined = {
    running: true, working: true, blocked: true, member_activity: [{ member_id: 'ops', status: 'running' }],
    pending_actions: [
      { kind: 'retry', member_id: 'qa', task_id: 'failed', execution_generation: 1 },
      { kind: 'approval', member_id: 'ops', task_id: 'live', execution_generation: 1, request_id: 'r1', approval: { choices: ['deny'] } }
    ]
  }

  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room', members }, driver_status: driver } : method === 'groups.log' ? { events: [] } : {})
  const view = render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'waiting' }} />)
  await screen.findByText('Blocked · 1 waiting for approval · Needs attention: 1')
  expect(screen.getByRole('status').textContent).toBe('Working: Ops')

  view.unmount()
  // A gateway whose driver is down sends no driver status.
  driver = undefined
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'stopped' }} />)
  await screen.findByText('Group driver unavailable. Update or reconnect the owning gateway.')
})

it('keeps one cancellation intent until Stop is confirmed, and fences again behind a message still sending', async () => {
  let deliver!: (value: unknown) => void
  let stops = 0
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { working: true } }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {return new Promise(resolve => { deliver = resolve })}

    if (method === 'groups.stop') {
      stops += 1

      if (stops === 1) {throw new Error('request timed out: groups.stop')}

      return { cancelled: stops === 2 ? 2 : 0 }
    }

    return params
  })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'busy-send' }} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  const stop = screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement
  fireEvent.click(stop)
  await screen.findByText('request timed out: groups.stop')
  fireEvent.click(stop)
  await screen.findByText('Stop requested')

  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Long task' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
  expect(screen.queryByText('Stop requested')).toBeNull()
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  // The first fence leaves at once; the message on the wire may post after it, so Stop stays open.
  await waitFor(() => expect(stops).toBe(3))
  expect(stop.disabled).toBe(true)
  deliver(receipt(request.mock.calls.find(call => call[1] === 'groups.send')![2]))
  await screen.findByText('No new replies were cancelled.')

  const cancelIds = request.mock.calls.filter(call => call[1] === 'groups.stop').map(call => call[2].cancel_id)
  expect(cancelIds).toHaveLength(4)
  expect(cancelIds[1]).toBe(cancelIds[0])
  expect(new Set(cancelIds.slice(1)).size).toBe(3)
})

// Send outcomes. Seams: `host.requestProfile` is the gateway transport; `window.hermesDesktop.preparedSubmissions`
// is the native journal's IPC (an in-memory compare-and-set owner here), or absent for the browser-storage
// journal. A refusal carries `admitted: false` only when the gateway refused before the user event committed
// (HostedRoomService.send).
const refusal = (reason: string, admitted?: false) =>
  Object.assign(new Error(reason), { code: 4001, data: { reason, ...(admitted === false ? { admitted } : {}) } })

const frame = (message: string, code: number, data?: Record<string, unknown>) => Object.assign(new Error(message), { code, data })

const MAYBE = 'This message may already have been sent. Retry sends the same message; it will not be posted twice.'
const REFUSED = 'The gateway refused this message. Edit it and send again.'
const FINISHED_ELSEWHERE = 'Another Desktop window already finished with this message. Check the conversation before sending it again.'

function room(sendReply: (params: Record<string, unknown>, attempt: number) => unknown) {
  let attempt = 0
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { working: true } }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.stop') {return { cancelled: 0 }}

    return method === 'groups.send' ? sendReply(params, ++attempt) : {}
  })
}

/** The native journal owner: one atomic compare-and-set per call, as electron/prepared-submissions.ts. */
function nativeJournal(beforeMutation?: (call: number, entry: string | null) => Promise<void> | void) {
  const journal: Record<string, unknown> = {}
  let calls = 0
  window.hermesDesktop = { preparedSubmissions: {
    read: async () => JSON.stringify(journal),
    compareAndSet: async (key: string, expected: string | null, entry: string | null) => {
      await beforeMutation?.(++calls, entry)
      const current = Object.hasOwn(journal, key) ? JSON.stringify(journal[key]) : null

      if (current !== (expected === null ? null : JSON.stringify(JSON.parse(expected)))) {return { applied: false, current }}

      if (entry === null) {delete journal[key]} else {journal[key] = JSON.parse(entry)}

      return { applied: true, current: entry }
    }
  } } as unknown as typeof window.hermesDesktop

  return journal
}

const sentIds = () => request.mock.calls.filter(call => call[1] === 'groups.send').map(call => call[2].event_id as string)
const editor = (scope: HTMLElement) => within(scope).getByRole('textbox') as HTMLTextAreaElement

async function openRoom(binding: { connectionId: string; profile: string; roomId: string }) {
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await within(view.container).findByText('Room')

  return view.container
}

async function typeAndSend(scope: HTMLElement, text: string) {
  await waitFor(() => expect(editor(scope).disabled).toBe(false))
  fireEvent.change(editor(scope), { target: { value: text } })
  fireEvent.click(within(scope).getByRole('button', { name: 'Send' }))
}

// AM1: a valid receipt is delivered even when its journal cleanup fails, and that intent is never
// reused for a new message.
it('treats a receipt whose journal cleanup failed as delivered, and sends a new identical message under its own identity', async () => {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  let failedOnce = false

  const journal = nativeJournal((_call, entry) => {
    if (entry === null && !failedOnce) {
      failedOnce = true

      throw new Error('EIO: journal unavailable')
    }
  })

  room(params => receipt(params))
  const scope = await openRoom({ connectionId: 'local', profile: 'team', roomId: 'cleanup-retry' })
  await typeAndSend(scope, 'Continue')
  await waitFor(() => expect(editor(scope).value).toBe(''))
  expect(within(scope).queryByRole('alert')).toBeNull()
  expect(warn).toHaveBeenCalled()
  await typeAndSend(scope, 'Continue')
  await waitFor(() => expect(sentIds()).toHaveLength(2))
  expect(new Set(sentIds()).size).toBe(2)
  await waitFor(() => expect(Object.keys(journal)).toHaveLength(0))
  warn.mockRestore()
})

// AM2: a refused first request proves nothing once another window has dispatched the same intent,
// also for a window opened after the refused one closed.
it('keeps the identity while another window\'s request of the intent may have posted, also after the refused window closed', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'closed-dispatcher' }
  let refuseA!: () => void

  room((_params, attempt) => attempt === 1
    ? new Promise((_resolve, reject) => { refuseA = () => reject(refusal('permission_denied', false)) })
    : attempt === 2 ? Promise.reject(new Error('request timed out: groups.send')) : Promise.reject(refusal('invalid_params', false)))
  const a = render(<CanonicalGroupWorkspace binding={binding} />)
  await within(a.container).findByText('Room')
  await typeAndSend(a.container, 'Please do this once')
  await waitFor(() => expect(sentIds()).toHaveLength(1))
  const b = await openRoom(binding)
  fireEvent.click(await within(b).findByRole('button', { name: 'Retry' }))
  await within(b).findByText('request timed out: groups.send')
  const before = JSON.stringify(await readCanonicalGroupSend(binding))
  a.unmount()
  refuseA()
  // The closed window still settles its refusal in the journal.
  await waitFor(async () => expect(JSON.stringify(await readCanonicalGroupSend(binding))).not.toBe(before))

  const c = await openRoom(binding)
  const retry = await within(c).findByRole('button', { name: 'Retry' }) as HTMLButtonElement
  await waitFor(() => expect(retry.disabled).toBe(false))
  fireEvent.click(retry)
  await within(c).findByText('invalid_params')
  expect(within(c).getByText(MAYBE)).toBeTruthy()
  expect(editor(c).disabled).toBe(true)
  expect(new Set(sentIds()).size).toBe(1)
  expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(sentIds()[0])
})

// AM3: a refused intent's words stay durable until the user replaces them with a new Send.
it('hands a refused converged intent back for editing without losing either window\'s words', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'converged-refusal' }
  room((params, attempt) => attempt === 1 ? Promise.reject(refusal('invalid_params', false)) : receipt(params))
  const scope = await openRoom(binding)
  await waitFor(() => expect(editor(scope).disabled).toBe(false))
  const other = await prepareCanonicalGroupSend(binding, { text: 'Too long for the gateway', attachments: [] })
  await typeAndSend(scope, 'My own words')
  await within(scope).findByText(REFUSED)
  await waitFor(() => expect(editor(scope).disabled).toBe(false))
  expect(editor(scope).value).toBe('Too long for the gateway')
  expect((await readCanonicalGroupSend(binding))?.params.payload.text).toBe('Too long for the gateway')

  fireEvent.change(editor(scope), { target: { value: 'Shorter' } })
  fireEvent.click(within(scope).getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(editor(scope).value).toBe('My own words'))
  const sends = request.mock.calls.filter(call => call[1] === 'groups.send').map(call => call[2])
  expect(sends.map(sent => sent.payload.text)).toEqual(['Too long for the gateway', 'Shorter'])
  expect(sends[0].event_id).toBe(other.params.event_id)
  expect(sends[1].event_id).not.toBe(other.params.event_id)
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()
})

// AM3: also when its room closed before the refusal arrived.
it('keeps a first Send refused after its room closed for the next open, where Retry hands it back', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'closed-mid-send' }
  let refuse!: () => void
  room((_params, attempt) => attempt === 1
    ? new Promise((_resolve, reject) => { refuse = () => reject(refusal('invalid_params', false)) })
    : Promise.reject(refusal('invalid_params', false)))
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await within(view.container).findByText('Room')
  await typeAndSend(view.container, 'Typed before closing')
  await waitFor(() => expect(sentIds()).toHaveLength(1))
  view.unmount()
  refuse()
  await waitFor(async () => expect((await readCanonicalGroupSend(binding))?.attempts).toBe(0))

  const scope = await openRoom(binding)
  const retry = await within(scope).findByRole('button', { name: 'Retry' }) as HTMLButtonElement
  await waitFor(() => expect(retry.disabled).toBe(false))
  fireEvent.click(retry)
  await within(scope).findByText(REFUSED)
  expect(editor(scope).value).toBe('Typed before closing')
  expect((await readCanonicalGroupSend(binding))?.attempts).toBe(0)
})

// Both surfaces Desktop dials: the session gateway (4001 + reason) and `hermes serve`
// (tui_gateway/methods_groups.py: 4111 room errors, 4123 no worker, 5112 anything else).
it.each([
  ['the gateway refuses the payload', refusal('invalid_params', false)],
  ['hermes serve finds the room ended', frame('hosted room not found', 4111, { admitted: false })],
  ['hermes serve has no Group Chat worker', frame('Group Chat worker is unavailable.', 4123, { admitted: false })],
  ['hermes serve refuses the payload', frame('user payload text must be a bounded string', 5112, { admitted: false })]
])('hands the words back for editing when %s before admitting the Send', async (_case, refused) => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'refused-before-admission' }
  room((params, attempt) => attempt === 1 ? Promise.reject(refused) : receipt(params))
  const scope = await openRoom(binding)
  await typeAndSend(scope, 'Ended elsewhere')
  await within(scope).findByText(REFUSED)
  await waitFor(() => expect(editor(scope).disabled).toBe(false))
  expect(editor(scope).value).toBe('Ended elsewhere')
  expect(within(scope).getByRole('button', { name: 'Send' })).toBeTruthy()
  // The words stay journaled until a new Send replaces them.
  expect(await readCanonicalGroupSend(binding)).toMatchObject({ attempts: 0, params: { payload: { text: 'Ended elsewhere' } } })
})

// AM4: the same reasons and codes also follow a committed user event, without `admitted: false`.
it.each([
  ['an invalid_params from the gateway', refusal('invalid_params')],
  ['a room error from hermes serve', frame('hosted room not found', 4111)],
  ['a failure from hermes serve', frame('storage is full', 5112)]
])('keeps the identity of a first Send after %s that does not prove nothing was admitted', async (_case, failed) => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'post-commit' }
  room((params, attempt) => attempt === 1 ? Promise.reject(failed) : receipt(params))
  const scope = await openRoom(binding)
  await typeAndSend(scope, 'Second')
  await within(scope).findByText(MAYBE)
  expect(editor(scope).disabled).toBe(true)
  expect(editor(scope).value).toBe('Second')
  const retry = within(scope).getByRole('button', { name: 'Retry' }) as HTMLButtonElement
  await waitFor(() => expect(retry.disabled).toBe(false))
  fireEvent.click(retry)
  await waitFor(() => expect(sentIds()).toHaveLength(2))
  expect(new Set(sentIds()).size).toBe(1)
})

it.each([
  ['names another intent', (params: Record<string, unknown>) => ({ accepted: true, client_event_id: `${params.event_id}-other` })],
  ['is not accepted', (params: Record<string, unknown>) => ({ accepted: false, client_event_id: params.event_id })],
  ['is empty', () => undefined]
])('does not treat a reply that %s as delivered', async (_case, reply) => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'malformed' }
  room(reply)
  const scope = await openRoom(binding)
  await typeAndSend(scope, 'Needs a real receipt')
  await within(scope).findByText(MAYBE)
  expect(editor(scope).value).toBe('Needs a real receipt')
  expect((await readCanonicalGroupSend(binding))?.params.payload.text).toBe('Needs a real receipt')
})

// AM6: Stop covers a Send already initiated when it was pressed.
it('does not dispatch a Send still preparing when Stop is pressed, and keeps its words', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'stop-while-preparing' }
  let entered!: () => void
  let release!: () => void
  const preparing = new Promise<void>(resolve => { entered = resolve })
  const gate = new Promise<void>(resolve => { release = resolve })
  nativeJournal(async call => { if (call === 1) { entered(); await gate } })
  room(params => receipt(params))
  const scope = await openRoom(binding)
  await typeAndSend(scope, 'Long task')
  await preparing
  const stop = within(scope).getByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.stop')).toBe(true))
  release()
  await within(scope).findByText('Stopped before sending. Your message is still here.')
  await within(scope).findByText('No new replies were cancelled.')
  // Nothing was sent, so the first fence covers it: no second Stop.
  expect(request.mock.calls.filter(call => call[1] === 'groups.stop')).toHaveLength(1)
  expect(sentIds()).toEqual([])
  await waitFor(() => expect(editor(scope).disabled).toBe(false))
  expect(editor(scope).value).toBe('Long task')
  expect((await readCanonicalGroupSend(binding))?.params.payload.text).toBe('Long task')

  fireEvent.click(within(scope).getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(sentIds()).toHaveLength(1))
  await waitFor(() => expect(editor(scope).value).toBe(''))
})

it('fences no second time behind a Send the gateway refused before admitting it', async () => {
  let refuse!: () => void
  room(() => new Promise((_resolve, reject) => { refuse = () => reject(refusal('invalid_params', false)) }))
  const scope = await openRoom({ connectionId: 'local', profile: 'team', roomId: 'stop-refused' })
  await typeAndSend(scope, 'Never admitted')
  await waitFor(() => expect(sentIds()).toHaveLength(1))
  fireEvent.click(within(scope).getByRole('button', { name: 'Stop' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.stop')).toBe(true))
  refuse()
  await within(scope).findByText('No new replies were cancelled.')
  expect(request.mock.calls.filter(call => call[1] === 'groups.stop')).toHaveLength(1)
})

it('never resends an intent another window already finished with', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'finished-elsewhere' }
  await prepareCanonicalGroupSend(binding, { text: 'From before a reload', attachments: [] })
  room(params => receipt(params))
  const scope = await openRoom(binding)
  const retry = await within(scope).findByRole('button', { name: 'Retry' }) as HTMLButtonElement
  await waitFor(() => expect(retry.disabled).toBe(false))
  localStorage.clear() // the other window received its receipt and retired the slot
  fireEvent.click(retry)
  await within(scope).findByText(FINISHED_ELSEWHERE)
  expect(sentIds()).toEqual([])
  // Its words are not offered again as a new message: that could post them twice.
  expect(editor(scope).value).toBe('')
})

it('does not resend words handed back for editing once another window finished with their intent', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'edited-elsewhere' }
  room((params, attempt) => attempt === 1 ? Promise.reject(refusal('invalid_params', false)) : receipt(params))
  const scope = await openRoom(binding)
  await typeAndSend(scope, 'Handed back')
  await within(scope).findByText(REFUSED)
  localStorage.clear() // another window sent the intent again and retired it on its receipt
  await waitFor(() => expect(editor(scope).disabled).toBe(false))
  fireEvent.click(within(scope).getByRole('button', { name: 'Send' }))
  await within(scope).findByText(FINISHED_ELSEWHERE)
  expect(sentIds()).toHaveLength(1)
  expect(editor(scope).value).toBe('Handed back')
  await waitFor(() => expect(editor(scope).disabled).toBe(false))
  fireEvent.click(within(scope).getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(sentIds()).toHaveLength(2))
  expect(new Set(sentIds()).size).toBe(2)
})

it('offers a read-only retry when the Send journal cannot be read, then restores its pending intent', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'unreadable' }
  const entry = await prepareCanonicalGroupSend(binding, { text: 'Saved before the failure', attachments: [] })
  const saved = localStorage.getItem('hermes.desktop.canonicalGroupSends.v1')!
  const compareAndSet = vi.fn()
  const read = vi.fn().mockRejectedValueOnce(new Error('EACCES: journal unreadable')).mockResolvedValue(saved)
  window.hermesDesktop = { preparedSubmissions: { read, compareAndSet } } as unknown as typeof window.hermesDesktop
  room(params => receipt(params))
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText(/Saved messages couldn’t be loaded/)
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: 'Try loading again' }))
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(entry.params.payload.text))
  expect(screen.queryByText(/Saved messages couldn’t be loaded/)).toBeNull()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(compareAndSet).not.toHaveBeenCalled()
})

it('names the Bot behind each pending action and shows what an approval would allow before offering it', async () => {
  const members = [{ member_id: 'ops', profile: 'ops', handle: 'ops', display_name: 'Ops Bot' },
    { member_id: 'qa', profile: 'qa', handle: 'qa', display_name: 'QA Bot' },
    { member_id: 'guard', profile: 'guard', handle: 'guard', display_name: 'Guard Bot' }]

  const actions = [
    { kind: 'approval', member_id: 'ops', task_id: 'clean', execution_generation: 3, request_id: 'r-clean',
      approval: { request_id: 'r-clean', command: 'rm -rf build', description: 'Clear the build folder', choices: ['once', 'deny'] } },
    { kind: 'approval', member_id: 'qa', task_id: 'vague', execution_generation: 1, request_id: 'r-vague',
      approval: { description: 'Use a tool', choices: ['once', 'deny'] } },
    // A plugin `approve` rule (tools/approval.py request_tool_approval): a tool label and the plugin's words.
    { kind: 'approval', member_id: 'guard', task_id: 'pwd', execution_generation: 4, request_id: 'r-pwd',
      approval: { request_id: 'r-pwd', command: '<terminal> (plugin approval rule)', description: 'Approve running pwd', choices: ['once'] } },
    { kind: 'retry', member_id: 'qa', task_id: 'late', execution_generation: 2 }
  ]

  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room', members }, driver_status: { pending_actions: actions } }
    : method === 'groups.log' ? { events: [] } : { approved: true })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'approvals' }} />)

  const clean = (await screen.findByText('rm -rf build')).closest('[data-task-id]') as HTMLElement
  expect(within(clean).getByText('Ops Bot needs your approval')).toBeTruthy()
  expect(within(clean).getByText('Clear the build folder')).toBeTruthy()
  const vague = screen.getByText('QA Bot needs your approval').closest('[data-task-id]') as HTMLElement
  expect(within(vague).getByText('Action details are unavailable. Check again before allowing this.')).toBeTruthy()
  expect(within(vague).queryByRole('button', { name: 'Allow once' })).toBeNull()
  expect(within(vague).getByRole('button', { name: 'Deny' })).toBeTruthy()
  expect(screen.getByText('QA Bot couldn’t start this reply.')).toBeTruthy()
  expect(screen.queryByText('ops')).toBeNull()

  // As in the chat approval card, the plugin's words stand in for its tool label.
  const guarded = screen.getByText('Guard Bot needs your approval').closest('[data-task-id]') as HTMLElement
  expect(within(guarded).getByText('Approve running pwd')).toBeTruthy()
  expect(within(guarded).queryByText(/plugin approval rule/)).toBeNull()
  expect(within(guarded).queryByText('Action details are unavailable. Check again before allowing this.')).toBeNull()
  // Only the choices the gateway offered.
  expect(within(guarded).queryByRole('button', { name: 'Deny' })).toBeNull()

  fireEvent.click(within(clean).getByRole('button', { name: 'Allow once' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.approve')).toBe(true))
  expect(request.mock.calls.find(call => call[1] === 'groups.approve')![2]).toEqual({
    room_id: 'approvals', member_id: 'ops', task_id: 'clean', execution_generation: 3, request_id: 'r-clean', choice: 'once', profile: 'team'
  })

  await waitFor(() => expect((within(guarded).getByRole('button', { name: 'Allow once' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(within(guarded).getByRole('button', { name: 'Allow once' }))
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.approve')).toHaveLength(2))
  expect(request.mock.calls.filter(call => call[1] === 'groups.approve')[1][2]).toEqual({
    room_id: 'approvals', member_id: 'guard', task_id: 'pwd', execution_generation: 4, request_id: 'r-pwd', choice: 'once', profile: 'team'
  })
})
