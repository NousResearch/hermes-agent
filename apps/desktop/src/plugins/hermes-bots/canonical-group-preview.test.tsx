import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
const control = vi.hoisted(() => ({ epoch: (_n: number) => {} }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const activation = atom(0)
  control.epoch = n => activation.set(n)

  return { ...await pluginSdkMock(gateway.host), atom, useValue: useStore,
    gatewayActivationEpoch: () => activation.get(), useGatewayActivationEpoch: () => useStore(activation),
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, state: { connectionId: atom<string | null>('local'), profile: atom('default'), gateway: atom('open') }, requestProfile: request } }
})
import { host } from '@hermes/plugin-sdk'

import { CanonicalGroupPreview } from './canonical-group-preview'
import { prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

afterEach(() => {
  cleanup(); request.mockReset(); localStorage.clear()
  ;(host.state.connectionId as unknown as { set: (value: string) => void }).set('local')
  ;(host.state.profile as unknown as { set: (value: string) => void }).set('default')
  ;(host.state.gateway as unknown as { set: (value: string) => void }).set('open')
  control.epoch(0)
})

it('renders only the owning server first-round selection without changing the draft', async () => {
  const members = [{ member_id: 'one', profile: 'default', handle: 'one' },
    { member_id: 'two', profile: 'two', handle: 'two', display_name: 'Beta' }]

  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room: { room_id: 'room', name: 'Room', members, revision: 1, authority_gateway_id: 'install:x', authority_epoch: 1 }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.capabilities') {return { driver: true, methods: ['groups.preview'] }}

    if (method === 'groups.preview') {return { room_id: 'room', authority_gateway_id: 'install:x', authority_epoch: 1, revision: 1, members: [members[1]] }}

    return {}
  })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'default', roomId: 'room' }} />)
  const input = await screen.findByRole('textbox') as HTMLTextAreaElement
  await screen.findByText('Room')
  fireEvent.change(input, { target: { value: '@two hello' } })
  expect(await screen.findByText(/First-round selected: Beta/)).toBeTruthy()
  const status = screen.getByRole('status')
  expect(status.getAttribute('aria-live')).toBe('polite')
  expect(status.textContent).toContain('may pass or wait')
  expect(input.value).toBe('@two hello')
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(false)
})

const binding = { connectionId: 'local', profile: 'default', roomId: 'room' }

const roster = [{ member_id: 'one', profile: 'default', handle: 'one', display_name: 'Alpha' },
  { member_id: 'two', profile: 'two', handle: 'two', display_name: 'Beta' }]

const room = { room_id: 'room', name: 'Room', members: roster, revision: 1, authority_gateway_id: 'install:x', authority_epoch: 1 }
const reply = (members: typeof roster, revision = 1) => ({ room_id: 'room', authority_gateway_id: 'install:x', authority_epoch: 1, revision, members })

it.each([
  ['hello', roster], ['@unknown', roster], ['@one', [roster[0]]],
  ['@two @one', roster], ['@everyone @two', roster],
  ['@TWO @two', [roster[1]]], ['@Beta', roster]
])('shows the server result for %s without local mention parsing', async (draft, selected) => {
  request.mockImplementation(async (_route, method) => method === 'groups.capabilities'
    ? { driver: true, methods: ['groups.preview'] } : reply(selected))
  render(<CanonicalGroupPreview binding={binding} draft={draft as string} room={room} visible />)
  const expected = (selected as typeof roster).map(member => member.display_name).join(', ')
  expect(await screen.findByText(new RegExp(`First-round selected: ${expected}`))).toBeTruthy()
})

it('drops out-of-order drafts, roster observations and source changes', async () => {
  const deferred: Record<string, (result: ReturnType<typeof reply>) => void> = {}
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return { driver: true, methods: ['groups.preview'] }}

    return new Promise(resolve => { deferred[params.text] = resolve })
  })
  const view = render(<CanonicalGroupPreview binding={binding} draft="old" room={room} visible />)
  await waitFor(() => expect(deferred.old).toBeDefined())
  view.rerender(<CanonicalGroupPreview binding={binding} draft="new" room={room} visible />)
  await waitFor(() => expect(deferred.new).toBeDefined())
  await act(async () => deferred.new(reply([roster[1]])))
  expect(screen.getByText(/First-round selected: Beta/)).toBeTruthy()
  await act(async () => deferred.old(reply([roster[0]])))
  expect(screen.queryByText(/First-round selected: Alpha/)).toBeNull()
  view.rerender(<CanonicalGroupPreview binding={binding} draft="new" room={{ ...room, revision: 2 }} visible />)
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  await act(async () => (host.state.profile as unknown as { set: (value: string) => void }).set('other'))
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  view.unmount()
})

it('does not reuse a previous preview after hide/show, empty draft, or connection switch', async () => {
  request.mockImplementation(async (_route, method) => method === 'groups.capabilities'
    ? { driver: true, methods: ['groups.preview'] } : reply([roster[0]]))
  const view = render(<CanonicalGroupPreview binding={binding} draft="hello" room={room} visible />)
  await screen.findByText(/First-round selected: Alpha/)
  view.rerender(<CanonicalGroupPreview binding={binding} draft="hello" room={room} visible={false} />)
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  view.rerender(<CanonicalGroupPreview binding={binding} draft="hello" room={room} visible />)
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  await screen.findByText(/First-round selected: Alpha/)
  view.rerender(<CanonicalGroupPreview binding={binding} draft="" room={room} visible />)
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  view.rerender(<CanonicalGroupPreview binding={binding} draft="hello" room={room} visible />)
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  await screen.findByText(/First-round selected: Alpha/)
  await act(async () => (host.state.connectionId as unknown as { set: (value: string) => void }).set('other'))
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  await act(async () => (host.state.connectionId as unknown as { set: (value: string) => void }).set('local'))
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  await screen.findByText(/First-round selected: Alpha/)
  await act(async () => control.epoch(1))
  expect(screen.queryByText(/First-round selected/)).toBeNull()
})

it.each(['room', 'profile', 'connection', 'activation', 'visibility', 'draft-aba', 'roster'])('fences pending responses across %s transitions', async scenario => {
  const pending: Array<{ route: unknown; params: Record<string, unknown>; resolve: (value: ReturnType<typeof reply>) => void }> = []
  request.mockImplementation((_route, method, params) => {
    if (method === 'groups.capabilities') {return Promise.resolve({ driver: true, methods: ['groups.preview'] })}

    return new Promise(resolve => pending.push({ route: _route, params, resolve }))
  })
  let activeBinding = binding
  let activeRoom = room
  let draft = 'first'
  const view = render(<CanonicalGroupPreview binding={activeBinding} draft={draft} room={activeRoom} visible />)
  await waitFor(() => expect(pending).toHaveLength(1))
  expect(pending[0].params).toEqual({ room_id: 'room', text: 'first', profile: 'default' })
  expect(pending[0].route).toMatchObject({ connectionId: 'local', profile: 'default', targetProfile: 'default' })

  if (scenario === 'room') {
    activeBinding = { ...binding, roomId: 'other-room' }
    activeRoom = { ...room, room_id: 'other-room' }
    view.rerender(<CanonicalGroupPreview binding={activeBinding} draft={draft} room={activeRoom} visible />)
  } else if (scenario === 'profile' || scenario === 'connection') {
    const state = scenario === 'profile' ? host.state.profile : host.state.connectionId
    await act(async () => (state as unknown as { set: (v: string) => void }).set('away'))
    expect(screen.queryByRole('status')).toBeNull()
    await act(async () => (state as unknown as { set: (v: string) => void }).set(scenario === 'profile' ? 'default' : 'local'))
  } else if (scenario === 'activation') {
    await act(async () => control.epoch(1))
  } else if (scenario === 'visibility') {
    view.rerender(<CanonicalGroupPreview binding={binding} draft={draft} room={room} visible={false} />)
    expect(screen.queryByRole('status')).toBeNull()
    view.rerender(<CanonicalGroupPreview binding={binding} draft={draft} room={room} visible />)
  } else if (scenario === 'draft-aba') {
    view.rerender(<CanonicalGroupPreview binding={binding} draft="middle" room={room} visible />)
    await waitFor(() => expect(pending).toHaveLength(2))
    view.rerender(<CanonicalGroupPreview binding={binding} draft={draft} room={room} visible />)
  } else {
    activeRoom = { ...room, revision: 2, members: [{ ...roster[0], display_name: 'New Alpha' }, roster[1]] }
    view.rerender(<CanonicalGroupPreview binding={binding} draft={draft} room={activeRoom} visible />)
  }

  const expectedCount = scenario === 'draft-aba' ? 3 : 2
  expect(screen.queryByRole('status')).toBeNull()
  await waitFor(() => expect(pending).toHaveLength(expectedCount))
  const current = pending[expectedCount - 1]
  expect(current.params).toEqual({ room_id: activeBinding.roomId, text: draft, profile: 'default' })
  await act(async () => current.resolve({ ...reply([roster[1]], activeRoom.revision), room_id: activeBinding.roomId }))
  expect(screen.getByText(/First-round selected: Beta/)).toBeTruthy()
  await act(async () => pending[0].resolve(reply([roster[0]])))
  expect(screen.getByText(/First-round selected: Beta/)).toBeTruthy()
  expect(screen.queryByText(/First-round selected: Alpha/)).toBeNull()
})

it('does not dispatch an old draft after late capabilities or an unmounted debounce', async () => {
  let release!: (value: unknown) => void
  request.mockImplementation((_route, method) => method === 'groups.capabilities'
    ? new Promise(resolve => { release = resolve }) : Promise.resolve(reply([roster[0]])))
  const view = render(<CanonicalGroupPreview binding={binding} draft="old" room={room} visible />)
  await waitFor(() => expect(release).toBeDefined())
  await act(async () => (host.state.connectionId as unknown as { set: (v: string) => void }).set('away'))
  view.rerender(<CanonicalGroupPreview binding={binding} draft="new" room={room} visible />)
  await act(async () => (host.state.connectionId as unknown as { set: (v: string) => void }).set('local'))
  await act(async () => release({ driver: true, methods: ['groups.preview'] }))
  expect(request.mock.calls.filter(call => call[1] === 'groups.preview')).toHaveLength(0)
  view.unmount()
  await new Promise(resolve => setTimeout(resolve, 350))
  expect(request.mock.calls.filter(call => call[1] === 'groups.preview')).toHaveLength(0)
})

it.each(['send', 'stop', 'once', 'deny', 'restored'])('preserves %s behavior after an actual preview RPC rejection', async scenario => {
  const action = { kind: 'approval', member_id: 'one', task_id: 'task-1', execution_generation: 3, request_id: 'approval-1' }
  const attachment = { attachment_id: 'file-1', kind: 'file', name: 'report.txt', mime: 'text/plain', size: 1 }
  const original = scenario === 'restored' ? await prepareCanonicalGroupSend(binding, { text: 'exact draft Ω', attachments: [attachment] }) : undefined
  let rejectPreview!: (reason: Error) => void
  let sendAck!: (value: unknown) => void
  request.mockImplementation((_route, method) => {
    if (method === 'groups.state') {return Promise.resolve({ room, driver_status: { pending_actions: [action] } })}

    if (method === 'groups.log') {return Promise.resolve({ events: [] })}

    if (method === 'groups.capabilities') {return Promise.resolve({ driver: true, methods: ['groups.preview'] })}

    if (method === 'groups.preview') {return new Promise((_resolve, reject) => { rejectPreview = reject })}

    if (method === 'groups.attachment.upload') {return Promise.resolve(attachment)}

    if (method === 'groups.send') {return new Promise(resolve => { sendAck = resolve })}

    return Promise.resolve({})
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('Room')
  const input = screen.getByRole('textbox') as HTMLTextAreaElement

  if (scenario !== 'restored') {
    await waitFor(() => expect(input.disabled).toBe(false))
    const file = new File(['a'], 'report.txt', { type: 'text/plain' })
    Object.defineProperty(file, 'arrayBuffer', { value: async () => new Uint8Array([97]).buffer })
    fireEvent.paste(input, { clipboardData: { files: [file], types: ['Files'] } })
    await screen.findByText('report.txt')
    fireEvent.change(input, { target: { value: 'exact draft Ω' } })
  } else { await waitFor(() => expect(input.value).toBe('exact draft Ω')) }

  await waitFor(() => expect(rejectPreview).toBeDefined())
  const before = await readCanonicalGroupSend(binding)
  expect(before).toEqual(original)
  await act(async () => rejectPreview(new Error('preview failed')))
  expect(input.value).toBe('exact draft Ω')
  expect(screen.getByText('report.txt')).toBeTruthy()
  expect(await readCanonicalGroupSend(binding)).toEqual(before)
  expect(screen.queryByText(/First-round selected/)).toBeNull()

  if (scenario === 'restored') { expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(original?.params.event_id);

 return }

  if (scenario === 'send') {
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    await waitFor(() => expect(sendAck).toBeDefined())
    const sent = request.mock.calls.find(call => call[1] === 'groups.send')![2]
    expect(sent).toMatchObject({ room_id: 'room', profile: 'default', payload: { text: 'exact draft Ω', attachments: [attachment] }, event_id: expect.any(String) })
    expect((await readCanonicalGroupSend(binding))?.params.event_id).toBe(sent.event_id)
    await act(async () => sendAck({}))
    await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  } else if (scenario === 'stop') {
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }))
    await waitFor(() => expect(request.mock.calls.find(call => call[1] === 'groups.stop')?.[2]).toEqual({ room_id: 'room', profile: 'default', cancel_id: expect.any(String) }))
  } else {
    fireEvent.click(screen.getByRole('button', { name: scenario === 'once' ? 'Allow once' : 'Deny' }))
    await waitFor(() => expect(request.mock.calls.find(call => call[1] === 'groups.approve')?.[2]).toEqual({ room_id: 'room', profile: 'default', member_id: 'one', task_id: 'task-1', execution_generation: 3, request_id: 'approval-1', choice: scenario }))
  }
})

it.each(['unsupported', 'failed', 'offline'])('hides %s preview and preserves the composer', async scenario => {
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.state') {return { room, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.capabilities') {
      if (scenario === 'failed') {throw new Error('temporary')}

      return { driver: true, methods: [] }
    }

    return {}
  })

  if (scenario === 'offline') {(host.state.gateway as unknown as { set: (value: string) => void }).set('closed')}
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText('Room')
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(input.disabled).toBe(false))
  fireEvent.change(input, { target: { value: 'draft remains' } })
  await new Promise(resolve => setTimeout(resolve, 350))
  expect(screen.queryByText(/First-round selected/)).toBeNull()
  expect(input.value).toBe('draft remains')
  expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(false)
  expect(request.mock.calls.some(call => call[1] === 'groups.preview')).toBe(false)
})
