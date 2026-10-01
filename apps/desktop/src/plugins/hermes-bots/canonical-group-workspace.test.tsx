import type * as HermesSdk from '@hermes/plugin-sdk'
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const sdk = await vi.importActual<typeof HermesSdk>('@hermes/plugin-sdk')
  const { pluginSdkMock, createGroupGateway, captureGroupRequests } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { ...sdk, ...await pluginSdkMock(gateway.host), atom, useValue: useStore, MessageTextContent: sdk.MessageTextContent,
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    host: { ...gateway.host, requestProfile: captureGroupRequests(request).request } }
})
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings, $canonicalGroupNames, registerCanonicalGroup } from './canonical-group-registry'
import { prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { GroupChatWorkspace } from './group-chat-view'
import { CANONICAL_GROUP_CAPABILITIES } from './group-test-utils'
const originalDesktop = window.hermesDesktop
const labels = CANONICAL_GROUP_LOCALES.en

const chooseGroupAction = async (name: string) => {
  fireEvent.pointerDown(await screen.findByRole('button', { name: labels.groupActions }), { button: 0, ctrlKey: false })
  fireEvent.click(await screen.findByRole('menuitem', { name }))
}

beforeEach(() => { Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined }) })
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear(); window.hermesDesktop = originalDesktop })

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
  expect((screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement).disabled).toBe(false)
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
    update: vi.fn(async (key: string, value: string | null) => {
      await new Promise<void>(resolve => { releaseWrite = resolve })

      if (value === null) {delete journal[key]} else {journal[key] = JSON.parse(value)}
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
  await waitFor(() => expect(native.update).toHaveBeenCalled())
  expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(false)
  releaseWrite()
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  await waitFor(() => expect(native.update).toHaveBeenCalledTimes(2))
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
  fireEvent.click(await screen.findByRole('button', { name: labels.skipReply }))
  expect(screen.getByText(labels.discardWarning)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull()
  action.execution_generation = 8
  fireEvent.click(screen.getByRole('button', { name: labels.confirmDiscard }))
  await screen.findByText(labels.pendingActionUnconfirmed)
  expect(screen.getByRole('dialog')).toBeTruthy()
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
  render(<GroupChatWorkspace group={group} members={[]} />)
  fireEvent.click(await screen.findByRole('button', { name: labels.retryReply }))
  await waitFor(() => expect(request.mock.calls.filter(c => c[1] === 'groups.state').length).toBeGreaterThan(1))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Hello' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  expect(screen.getByText('Owner reply')).toBeTruthy()
  expect(request.mock.calls.every(c => c[1].startsWith('groups.'))).toBe(true)
})

const settle = async () => {
  for (let turn = 0; turn < 10; turn++) {await act(async () => { await vi.advanceTimersByTimeAsync(0) })}
}

it('reads only new room log entries on each visible poll and restarts on a new authority epoch', async () => {
  vi.useFakeTimers()

  try {
    let epoch = 1
    let log = [{ seq: 1, kind: 'message', payload: { text: 'one' } }, { seq: 2, kind: 'message', payload: { text: 'two' } }]
    request.mockImplementation(async (_route, method, params) => {
      if (method === 'groups.state') {return { room: { name: 'Room', authority_epoch: epoch }, driver_status: {} }}

      if (method === 'groups.log') {return { events: log.filter(event => event.seq > params.since_seq), has_more: false }}

      return {}
    })
    const binding = { connectionId: 'remote', profile: 'team', roomId: 'poll-room' }
    const view = render(<CanonicalGroupWorkspace binding={binding} />)
    const reads = () => request.mock.calls.filter(call => call[1] === 'groups.log').map(call => call[2].since_seq)
    await settle()
    expect(reads()).toEqual([0])
    log = [...log, { seq: 3, kind: 'message', payload: { text: 'three' } }]
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    await settle()
    expect(reads()).toEqual([0, 2])
    expect(screen.getAllByText(/^(one|two|three)$/).map(node => node.textContent)).toEqual(['one', 'two', 'three'])
    expect(request.mock.calls.filter(call => call[1] === 'groups.state')).toHaveLength(2)
    epoch = 2
    log = [{ seq: 1, kind: 'message', payload: { text: 'fresh' } }]
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    await settle()
    expect(reads()).toEqual([0, 2, 0])
    expect(screen.queryByText('one')).toBeNull()
    expect(screen.getByText('fresh')).toBeTruthy()
    view.rerender(<CanonicalGroupWorkspace binding={binding} visible={false} />)
    const before = request.mock.calls.length
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(request.mock.calls).toHaveLength(before)
  } finally {
    vi.useRealTimers()
  }
})

it('keeps Stop available while a Send is pending and reports a request without claiming completion', async () => {
  const stops: Record<string, unknown>[] = []
  let stopResult: () => Promise<unknown> = async () => ({ cancelled: 0 })
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { running: true, working: false } }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {return new Promise(() => {})}

    if (method === 'groups.stop') {
      stops.push(params)

      return stopResult()
    }

    return {}
  })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'stop-room' }} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Long task' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
  const stop = screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await screen.findByText(labels.nothingRunning)
  expect(stops).toEqual([{ room_id: 'stop-room', cancel_id: expect.any(String), profile: 'team' }])

  stopResult = async () => { throw new Error('stop refused') }
  fireEvent.click(stop)
  expect((await screen.findByRole('alert')).textContent).toContain(labels.pendingActionUnconfirmed)
  expect(screen.getByText('stop refused').closest('details')?.open).toBe(false)
})

it('keeps an idle chat quiet while its gateway is alive, even with a draft and file upload', async () => {
  let releaseUpload!: (value: unknown) => void
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Autumn launch' }, driver_status: { running: true, working: false, counts: { completed: 2 } } }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.attachment.upload') {return new Promise(resolve => { releaseUpload = resolve })}

    return {}
  })
  const view = render(<CanonicalGroupWorkspace binding={{ connectionId: 'idle-owner', profile: 'team', roomId: 'idle' }} />)
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(input.disabled).toBe(false))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  fireEvent.change(input, { target: { value: 'Review this file' } })
  fireEvent.change(view.container.querySelector('input[type=file]')!, { target: { files: [new File(['A'], 'notes.txt', { type: 'text/plain' })] } })
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.attachment.upload')).toBe(true))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true)
  await act(async () => releaseUpload({ attachment_id: 'uploaded', kind: 'file', name: 'notes.txt', mime: 'text/plain' }))
  await waitFor(() => expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(false))
  expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
})

it.each([
  ['queued work', { working: false, counts: { queued: 1 } }],
  ['stopping work', { working: false, counts: { stopping: 1 } }],
  ['unresolved reply', { working: false, pending_actions: [{ kind: 'discard', member_id: 'worker', task_id: 'pending', execution_generation: 1 }] }]
])('offers Stop for %s from the current driver receipt', async (_description, driver_status) => {
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Autumn launch' }, driver_status } : method === 'groups.log' ? { events: [] } : {})
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'active-owner', profile: 'team', roomId: _description }} />)
  const stop = await screen.findByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.stop')).toBe(true))
})

it('keeps Stop usable for an unresolved frozen Send even when driver status is unavailable', async () => {
  const binding = { connectionId: 'unconfirmed-owner', profile: 'team', roomId: 'unconfirmed' }
  await prepareCanonicalGroupSend(binding, { text: 'Please review the launch' })
  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Autumn launch' } } : method === 'groups.log' ? { events: [] } : {})
  render(<CanonicalGroupWorkspace binding={binding} />)
  const stop = await screen.findByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.stop')).toBe(true))
  expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(false)
})

it('shows every current participant without a network action and closes the popover when hidden', async () => {
  const members = Array.from({ length: 4 }, (_, index) => ({ member_id: `member-${index}`, profile: `profile-${index}`,
    handle: `handle-${index}`, display_name: index % 2 ? 'Mira Bot' : 'Atlas Bot' }))

  request.mockImplementation(async (_route, method) => method === 'groups.state'
    ? { room: { name: 'Autumn launch', members }, driver_status: {} } : method === 'groups.log' ? { events: [] } : {})
  const binding = { connectionId: 'participant-owner', profile: 'team', roomId: 'participants' }
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  const trigger = await screen.findByRole('button', { name: `${labels.members}: ${labels.memberCount.replace('{count}', '4')}` })
  const reads = request.mock.calls.length
  fireEvent.click(trigger)
  const list = within(await screen.findByRole('list', { name: labels.members }))
  expect(list.getAllByText('Atlas Bot')).toHaveLength(2)
  expect(list.getAllByText('Mira Bot')).toHaveLength(2)
  expect(list.queryByText('handle-3')).toBeNull()
  expect(request.mock.calls).toHaveLength(reads)
  view.rerender(<CanonicalGroupWorkspace binding={binding} visible={false} />)
  expect(screen.queryByRole('list', { name: labels.members })).toBeNull()
})

it('lists an unresolved member beside live work without disabling Stop or polling', async () => {
  vi.useFakeTimers()

  try {
    request.mockImplementation(async (_route, method) => method === 'groups.state'
      ? { room: { name: 'Room' }, driver_status: { running: true, working: true,
        pending_actions: [{ kind: 'retry', member_id: 'alpha', task_id: 't', execution_generation: 1 }] } }
      : method === 'groups.log' ? { events: [] } : {})
    render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'warn' }} />)
    await settle()
    expect(screen.getByText(`${labels.statusWorking} · ${labels.statusAttention.replace('{count}', '1')}`)).toBeTruthy()
    expect(screen.getByText(labels.pendingRetryTitle.replace('{name}', labels.pendingBot))).toBeTruthy()
    expect(screen.queryByText('alpha')).toBeNull()
    expect((screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement).disabled).toBe(false)
    const polls = request.mock.calls.filter(call => call[1] === 'groups.state').length
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    await settle()
    expect(request.mock.calls.filter(call => call[1] === 'groups.state').length).toBeGreaterThan(polls)
  } finally {
    vi.useRealTimers()
  }
})

const refusal = (reason: string) => Object.assign(new Error(reason), { code: 4001, data: { reason } })

it('hands a refused message back for editing and keeps other outcomes for an exact retry', async () => {
  let outcome: () => Promise<unknown> = async () => { throw refusal('invalid_params') }
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {return outcome()}

    return {}
  })
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'outcomes' }
  render(<CanonicalGroupWorkspace binding={binding} />)
  const box = () => screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(box().disabled).toBe(false))
  fireEvent.change(box(), { target: { value: 'Too long' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText(labels.sendRefused)
  await waitFor(() => expect(box().disabled).toBe(false))
  expect(box().value).toBe('Too long')
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()

  outcome = async () => { throw refusal('not_ready') }
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText(labels.sendNotYet)
  const kept = await readCanonicalGroupSend(binding)
  expect(kept?.params.payload.text).toBe('Too long')

  outcome = async () => { throw new Error('socket closed') }
  fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
  await screen.findByText(labels.sendMaybe)
  expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(kept?.params.event_id)
  const ids = request.mock.calls.filter(call => call[1] === 'groups.send').map(call => call[2].event_id)
  expect(ids[0]).not.toBe(ids[1])
  expect(ids[2]).toBe(ids[1])

  outcome = async () => ({ accepted: true, client_event_id: 'someone-else' })
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.send')).toHaveLength(4))
  expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(kept?.params.event_id)

  outcome = async () => ({ accepted: true, client_event_id: kept?.params.event_id, driver_started: true })
  fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  await waitFor(() => expect(box().value).toBe(''))
})

it('returns a failed Send only to its own room when the view switches rooms', async () => {
  let refuse!: () => void
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.send') {return new Promise((_resolve, reject) => { refuse = () => reject(refusal('invalid_params')) })}

    return {}
  })
  const first = { connectionId: 'remote', profile: 'team', roomId: 'room-a' }
  const second = { connectionId: 'remote', profile: 'team', roomId: 'room-b' }
  const view = render(<CanonicalGroupWorkspace binding={first} />)
  const box = () => screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(box().disabled).toBe(false))
  fireEvent.change(box(), { target: { value: 'For room A' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
  view.rerender(<CanonicalGroupWorkspace binding={second} />)
  await waitFor(() => expect(box().disabled).toBe(false))
  refuse()
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)) })
  expect(box().value).toBe('')
  expect(screen.queryByRole('alert')).toBeNull()
  expect((await readCanonicalGroupSend(first))?.params.payload.text).toBe('For room A')
  view.rerender(<CanonicalGroupWorkspace binding={first} />)
  await waitFor(() => expect(box().value).toBe('For room A'))
})

it('renames a gateway room with one event id across retries', async () => {
  let name = 'Old name'

  let renameOutcome: () => Promise<unknown> = async () => { throw new Error('socket closed') }

  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (method === 'groups.state') {return { room: { name }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.rename') {
      const result = await renameOutcome()
      name = params.name

      return result
    }

    return {}
  })
  const group = registerCanonicalGroup({ connectionId: 'rename-owner', profile: 'team' }, { room_id: 'named', name: 'Old name', members: [] })
  render(<GroupChatWorkspace group={group} members={[]} />)
  await chooseGroupAction(labels.rename)
  fireEvent.change(screen.getByRole('textbox', { name: labels.roomName }), { target: { value: 'New name' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect((await screen.findByRole('alert')).textContent).toContain(labels.pendingActionUnconfirmed)
  expect(screen.getByText('socket closed').closest('details')?.open).toBe(false)

  renameOutcome = async () => ({ room: { room_id: 'named', name: 'New name' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await screen.findByRole('heading', { name: 'New name' })
  expect($canonicalGroupNames.get()[group]).toBe('New name')
  const renames = request.mock.calls.filter(call => call[1] === 'groups.rename').map(call => call[2])
  expect(renames).toEqual([
    { room_id: 'named', event_id: expect.any(String), name: 'New name', profile: 'team' },
    { room_id: 'named', event_id: renames[0].event_id, name: 'New name', profile: 'team' }
  ])
})

it.each([
  ['missing receipt', {}],
  ['legacy boolean', { tombstone: true }],
  ['another room', { tombstone: { room_id: 'other-room', disbanded_at: 123, idempotent: false } }],
  ['missing timestamp', { tombstone: { room_id: 'leaving', idempotent: false } }]
])('keeps a room for %s and accepts its canonical Disband receipt', async (_label, initial) => {
  let disbandResult: unknown = initial
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (method === 'groups.state') {return { room: { name: 'Leaving' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.disband') {return disbandResult}

    return {}
  })
  const binding = { connectionId: 'disband-owner', profile: 'team', roomId: 'leaving' }
  const key = registerCanonicalGroup(binding, { room_id: 'leaving', name: 'Leaving', members: [] })
  const onBack = vi.fn()
  render(<GroupChatWorkspace group={key} members={[]} onBack={onBack} />)
  await chooseGroupAction(labels.disband)
  expect(request.mock.calls.some(call => call[1] === 'groups.disband')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: labels.confirmDisband }))
  await screen.findByText(labels.disbandUnconfirmed)
  expect($canonicalGroupBindings.get()[key]).toEqual(binding)
  expect(onBack).not.toHaveBeenCalled()

  disbandResult = { tombstone: { room_id: binding.roomId, disbanded_at: 123, idempotent: false } }
  fireEvent.click(screen.getByRole('button', { name: labels.confirmDisband }))
  await waitFor(() => expect(onBack).toHaveBeenCalledOnce())
  expect($canonicalGroupBindings.get()[key]).toBeUndefined()
  expect(request.mock.calls.filter(call => call[1] === 'groups.disband').map(call => call[2])).toEqual([
    { room_id: 'leaving', cancel_id: expect.any(String), profile: 'team' },
    { room_id: 'leaving', cancel_id: expect.any(String), profile: 'team' }
  ])
})

it('shows shared confirmation progress and prevents duplicate End requests while its receipt is pending', async () => {
  let release!: (value: unknown) => void
  const held = new Promise(resolve => { release = resolve })
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (method === 'groups.state') {return { room: { name: 'Autumn launch' }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.disband') {return held}

    return {}
  })
  const binding = { connectionId: 'end-busy-owner', profile: 'team', roomId: 'ending' }
  const key = registerCanonicalGroup(binding, { room_id: binding.roomId, name: 'Autumn launch', members: [] })
  const onBack = vi.fn()
  render(<GroupChatWorkspace group={key} members={[]} onBack={onBack} />)
  await chooseGroupAction(labels.disband)
  const dialog = within(screen.getByRole('dialog'))
  const confirm = dialog.getByRole('button', { name: labels.confirmDisband }) as HTMLButtonElement
  fireEvent.click(confirm)
  await waitFor(() => expect(confirm.disabled).toBe(true))
  expect((dialog.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(confirm)
  expect(request.mock.calls.filter(call => call[1] === 'groups.disband')).toHaveLength(1)
  expect(onBack).not.toHaveBeenCalled()
  await act(async () => release({ tombstone: { room_id: binding.roomId, disbanded_at: 123 } }))
  await waitFor(() => expect(onBack).toHaveBeenCalledOnce())
})
