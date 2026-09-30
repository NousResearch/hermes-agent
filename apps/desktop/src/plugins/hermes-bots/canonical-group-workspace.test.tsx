import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway, captureGroupRequests } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { ...await pluginSdkMock(gateway.host), atom, useValue: useStore,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    host: { ...gateway.host, requestProfile: captureGroupRequests(request).request } }
})
import { $canonicalGroupBindings, registerCanonicalGroup } from './canonical-group-registry'
import { prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { GroupChatWorkspace } from './group-chat-view'
import { CANONICAL_GROUP_CAPABILITIES } from './group-test-utils'
const originalDesktop = window.hermesDesktop
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
  render(<GroupChatWorkspace group={group} members={[]} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
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

it('keeps Stop available while a Send is pending and reports what it stopped', async () => {
  const stops: Record<string, unknown>[] = []
  let stopResult: () => Promise<unknown> = async () => ({ cancelled: 0 })
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {return { room: { name: 'Room' }, driver_status: { running: true, working: true } }}

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
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Long task' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.send')).toBe(true))
  const stop = screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement
  expect(stop.disabled).toBe(false)
  fireEvent.click(stop)
  await screen.findByText('Nothing was running.')
  expect(stops).toEqual([{ room_id: 'stop-room', cancel_id: expect.any(String), profile: 'team' }])

  stopResult = async () => { throw new Error('stop refused') }
  fireEvent.click(stop)
  expect((await screen.findByRole('alert')).textContent).toBe('stop refused')
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
    expect(screen.getByText('Working · 1 need attention')).toBeTruthy()
    expect(screen.getByText('alpha')).toBeTruthy()
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
  await screen.findByText('The gateway refused this message. Edit it and send again.')
  await waitFor(() => expect(box().disabled).toBe(false))
  expect(box().value).toBe('Too long')
  expect(await readCanonicalGroupSend(binding)).toBeUndefined()

  outcome = async () => { throw refusal('not_ready') }
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText('Not sent yet. Retry sends the same message.')
  const kept = await readCanonicalGroupSend(binding)
  expect(kept?.params.payload.text).toBe('Too long')

  outcome = async () => { throw new Error('socket closed') }
  fireEvent.click(await screen.findByRole('button', { name: 'Retry' }))
  await screen.findByText(/may already have been sent/)
  expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(kept?.params.event_id)
  const ids = request.mock.calls.filter(call => call[1] === 'groups.send').map(call => call[2].event_id)
  expect(ids[0]).not.toBe(ids[1])
  expect(ids[2]).toBe(ids[1])
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
  fireEvent.click(await screen.findByRole('button', { name: 'Rename' }))
  fireEvent.change(screen.getByRole('textbox', { name: 'Room name' }), { target: { value: 'New name' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  expect((await screen.findByRole('alert')).textContent).toBe('socket closed')

  renameOutcome = async () => ({ room: { room_id: 'named', name: 'New name' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await screen.findByRole('heading', { name: 'New name' })
  const renames = request.mock.calls.filter(call => call[1] === 'groups.rename').map(call => call[2])
  expect(renames).toEqual([
    { room_id: 'named', event_id: expect.any(String), name: 'New name', profile: 'team' },
    { room_id: 'named', event_id: renames[0].event_id, name: 'New name', profile: 'team' }
  ])
})

it('disbands a gateway room only after confirmation and keeps it unless the gateway confirms', async () => {
  let disbandResult: unknown = {}
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
  fireEvent.click(await screen.findByRole('button', { name: 'Disband' }))
  expect(request.mock.calls.some(call => call[1] === 'groups.disband')).toBe(false)
  fireEvent.click(screen.getByRole('button', { name: 'Confirm disband' }))
  await screen.findByText('The gateway did not confirm the disband. The room was kept.')
  expect($canonicalGroupBindings.get()[key]).toEqual(binding)
  expect(onBack).not.toHaveBeenCalled()

  disbandResult = { tombstone: true }
  fireEvent.click(screen.getByRole('button', { name: 'Confirm disband' }))
  await waitFor(() => expect(onBack).toHaveBeenCalledOnce())
  expect($canonicalGroupBindings.get()[key]).toBeUndefined()
  expect(request.mock.calls.filter(call => call[1] === 'groups.disband').map(call => call[2])).toEqual([
    { room_id: 'leaving', cancel_id: expect.any(String), profile: 'team' },
    { room_id: 'leaving', cancel_id: expect.any(String), profile: 'team' }
  ])
})
