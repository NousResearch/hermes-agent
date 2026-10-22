import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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
  expect(await readCanonicalGroupSend(binding)).toEqual(entry)
  first.unmount()
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByRole('button', { name: 'Retry' })
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : {})
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
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : {})
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
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  await waitFor(() => expect(native.compareAndSet).toHaveBeenCalledTimes(2))
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
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Room' }, driver_status: {} } : method === 'groups.log' ? { events: [] } : {})
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
