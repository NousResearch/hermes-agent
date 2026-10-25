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
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')

  return { ...await pluginSdkMock(gateway.host), atom, useValue: useStore, ConfirmDialog,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) => CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: request } }
})
import { host } from '@hermes/plugin-sdk'

import { $canonicalGroupBindings, CanonicalGroupList, registerCanonicalGroup } from './canonical-group-registry'
import { prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { $groupChatWorkspace } from './group-chat'
import { GroupChatWorkspace } from './group-chat-view'
const originalDesktop = window.hermesDesktop
beforeEach(() => { Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined }) })
afterEach(() => { cleanup(); request.mockReset(); localStorage.clear(); window.hermesDesktop = originalDesktop; vi.unstubAllGlobals() })

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


const roomReads = (members: unknown[] = [], events: unknown[] = []) => (method: string) => method === 'groups.state'
  ? { room: { name: 'Room', members }, driver_status: {} } : method === 'groups.log' ? { events } : undefined

it('describes only a send restored from the journal as restored, never an ordinary in-flight send', async () => {
  let ack!: () => void
  request.mockImplementation(async (_route, method) => roomReads()(method)
    ?? (method === 'groups.send' ? new Promise(resolve => { ack = () => resolve({}) }) : {}))
  const view = render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'ordinary' }} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Hello' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.some(c => c[1] === 'groups.send')).toBe(true))
  expect(screen.queryByText(/unconfirmed send was restored/)).toBeNull()
  await act(async () => ack())
  view.unmount()

  const binding = { connectionId: 'remote', profile: 'team', roomId: 'restored' }
  await prepareCanonicalGroupSend(binding, { text: 'Left over', attachments: [] })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText(/unconfirmed send was restored/)
})

it('renames the room and disbands it only through ConfirmDialog, which stays open with a failed disband\'s error', async () => {
  const onBack = vi.fn()
  let disbands = 0
  request.mockImplementation(async (_route, method) => roomReads()(method)
    ?? (method === 'groups.disband' && ++disbands === 1 ? Promise.reject(new Error('disband failed')) : {}))
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'controls' }} onBack={onBack} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Rename' }))
  fireEvent.change(screen.getByRole('textbox', { name: 'New name' }), { target: { value: ' Renamed ' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(request.mock.calls.find(c => c[1] === 'groups.rename')?.[2]).toMatchObject({ room_id: 'controls', name: 'Renamed' }))
  const disband = screen.getByRole('button', { name: /disband/i }) as HTMLButtonElement
  await waitFor(() => expect(disband.disabled).toBe(false))
  fireEvent.click(disband)
  expect(request.mock.calls.some(c => c[1] === 'groups.disband')).toBe(false)
  const dialog = await screen.findByRole('dialog')
  fireEvent.click(within(dialog).getByRole('button', { name: /disband/i }))
  await within(dialog).findByText('disband failed')
  expect(onBack).not.toHaveBeenCalled()
  fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: /disband/i }))
  await waitFor(() => expect(onBack).toHaveBeenCalledOnce())
  expect(request.mock.calls.filter(c => c[1] === 'groups.disband').map(c => c[2])).toEqual([
    expect.objectContaining({ room_id: 'controls', cancel_id: expect.any(String) }), expect.objectContaining({ room_id: 'controls' })])
})

it('leaves the in-panel room after a disband even though no tab supplies onBack', async () => {
  request.mockImplementation(async (_route, method) => roomReads()(method) ?? {})
  const group = registerCanonicalGroup({ connectionId: 'local', profile: 'default' }, { room_id: 'panel', name: 'Room', members: [] })
  $groupChatWorkspace.set(group)
  render(<GroupChatWorkspace group={group} members={[]} origin="key" />)
  const disband = await screen.findByRole('button', { name: /disband/i }) as HTMLButtonElement
  await waitFor(() => expect(disband.disabled).toBe(false))
  fireEvent.click(disband)
  fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: /disband/i }))
  await waitFor(() => expect($groupChatWorkspace.get()).toBeNull())
})

it('drops a disbanded room from the sidebar list and its bindings, with the disbanded toast', async () => {
  vi.mocked(host.notify).mockClear()
  let disbanded = false
  const room = { room_id: 'gone', name: 'Doomed', members: [] }
  request.mockImplementation(async (_route, method) => method === 'groups.disband' ? (disbanded = true, {})
    : method === 'groups.capabilities' ? { driver: true } : method === 'groups.list' ? { rooms: disbanded ? [] : [room], next_offset: null }
      : roomReads()(method) ?? {})
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  await screen.findByRole('button', { name: 'Doomed' })
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'gone', roomName: 'Doomed' }
  const key = registerCanonicalGroup(binding, room)
  render(<CanonicalGroupWorkspace binding={binding} onBack={vi.fn()} />)
  const disband = await screen.findByRole('button', { name: /disband/i }) as HTMLButtonElement
  await waitFor(() => expect(disband.disabled).toBe(false))
  fireEvent.click(disband)
  fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: /disband/i }))
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Doomed' })).toBeNull())
  expect($canonicalGroupBindings.get()[key]).toBeUndefined()
  // This fixture's plugin i18n mock answers with the key: the legacy view's `group.disbanded` toast.
  expect(host.notify).toHaveBeenCalledExactlyOnceWith({ kind: 'success', message: 'group.disbanded' })
})

it('@-completes room members with the exact handle the gateway resolves', async () => {
  request.mockImplementation(async (_route, method) => roomReads([{ member_id: 'm1', profile: 'ops', handle: 'ops-2', display_name: 'Ops Bot' }])(method) ?? {})
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'mentions' }} />)
  await screen.findByText('Ops Bot')
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '@' } })
  expect(screen.getByText('@everyone')).toBeTruthy()
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '@op' } })
  fireEvent.mouseDown(screen.getByText('@ops-2'))
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('@ops-2 ')
})

it('uploads pasted and dropped files through the room upload and previews committed images inline', async () => {
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:preview') })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
  const image = { attachment_id: 'img', kind: 'image', name: 'shot.png', mime: 'image/png', size: 1 }
  const large = { ...image, attachment_id: 'large', name: 'large.png', size: 6 * 1024 * 1024 }
  const events = [{ seq: 1, room_id: 'paste', event_id: 'e1', kind: 'message.member', payload: { attachments: [image, large] } }]
  request.mockImplementation(async (_route, method, params) => roomReads([], events)(method)
    ?? (method === 'groups.attachment.upload' ? { attachment_id: `att-${params.name}`, kind: params.kind, name: params.name, mime: params.mime, size: 1 }
      : method === 'groups.attachment.download' ? { ...image, data_base64: 'QQ==' } : {}))
  const observers: IntersectionObserverCallback[] = []
  vi.stubGlobal('IntersectionObserver', class { constructor(callback: IntersectionObserverCallback) { observers.push(callback) } observe() {} disconnect() {} })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'paste' }} />)
  await waitFor(() => expect(observers).toHaveLength(1))
  expect(request.mock.calls.some(c => c[1] === 'groups.attachment.download')).toBe(false)
  act(() => observers[0]([{ isIntersecting: true }] as IntersectionObserverEntry[], {} as IntersectionObserver))
  await waitFor(() => expect(screen.getByRole('img', { name: 'shot.png' }).getAttribute('src')).toBe('blob:preview'))
  expect(request.mock.calls.filter(c => c[1] === 'groups.attachment.download').map(c => c[2].attachment_id)).toEqual(['img'])
  const box = screen.getByRole('textbox')
  await waitFor(() => expect((box as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.paste(box, { clipboardData: { files: [], types: ['text/plain'] } })
  fireEvent.paste(box, { clipboardData: { files: [new File(['a'], 'a.png', { type: 'image/png' })], types: ['Files'] } })
  await screen.findByText('a.png')
  fireEvent.drop(box, { dataTransfer: { files: [new File(['b'], 'b.txt', { type: 'text/plain' })], types: ['Files'] } })
  await screen.findByText('b.txt')
  expect(request.mock.calls.filter(c => c[1] === 'groups.attachment.upload').map(c => [c[2].name, c[2].kind])).toEqual([['a.png', 'image'], ['b.txt', 'file']])
})

it('re-gates previews on the observer after the room is hidden and shown, instead of refetching them all at once', async () => {
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:preview') })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
  const image = { attachment_id: 'img', kind: 'image', name: 'shot.png', mime: 'image/png', size: 1 }
  const events = [{ seq: 1, room_id: 'toggle', event_id: 'e1', kind: 'message.member', payload: { attachments: [image] } }]
  request.mockImplementation(async (_route, method) => roomReads([], events)(method)
    ?? (method === 'groups.attachment.download' ? { ...image, data_base64: 'QQ==' } : {}))
  const observers: IntersectionObserverCallback[] = []
  vi.stubGlobal('IntersectionObserver', class { constructor(callback: IntersectionObserverCallback) { observers.push(callback) } observe() {} disconnect() {} })
  const downloads = () => request.mock.calls.filter(c => c[1] === 'groups.attachment.download').length
  const binding = { connectionId: 'local', profile: 'team', roomId: 'toggle' }
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect(observers).toHaveLength(1))
  act(() => observers[0]([{ isIntersecting: true }] as IntersectionObserverEntry[], {} as IntersectionObserver))
  await screen.findByRole('img', { name: 'shot.png' })
  view.rerender(<CanonicalGroupWorkspace binding={binding} visible={false} />)
  view.rerender(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect(observers).toHaveLength(2))
  expect(downloads()).toBe(1)
  act(() => observers[1]([{ isIntersecting: true }] as IntersectionObserverEntry[], {} as IntersectionObserver))
  await waitFor(() => expect(downloads()).toBe(2))
})

it('keeps automatic preview downloads bounded and never requests one file twice at once', async () => {
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:preview') })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
  const image = (n: number) => ({ attachment_id: `img-${n}`, kind: 'image', name: `shot-${n}.png`, mime: 'image/png', size: 4 * 1024 * 1024 })
  const events = Array.from({ length: 20 }, (_, n) => ({ seq: n + 1, room_id: 'bound', event_id: `e${n}`, kind: 'message.member', payload: { attachments: [image(n)] } }))
  const held: Array<() => void> = []
  request.mockImplementation(async (_route, method) => roomReads([], events)(method)
    ?? (method === 'groups.attachment.download' ? new Promise(resolve => { held.push(() => resolve({ ...image(0), data_base64: 'QQ==' })) }) : {}))
  const observers: IntersectionObserverCallback[] = []
  vi.stubGlobal('IntersectionObserver', class { constructor(callback: IntersectionObserverCallback) { observers.push(callback) } observe() {} disconnect() {} })

  const seen = (callback: IntersectionObserverCallback, isIntersecting: boolean) =>
    act(() => callback([{ isIntersecting }] as IntersectionObserverEntry[], {} as IntersectionObserver))

  const downloads = () => request.mock.calls.filter(c => c[1] === 'groups.attachment.download').map(c => c[2].attachment_id)
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'bound' }
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect(observers).toHaveLength(20))

  // A fast scroll: every preview enters and leaves the viewport before any download finishes.
  for (const callback of [...observers]) { seen(callback, true); seen(callback, false) }
  expect(downloads().length).toBeGreaterThan(0)
  expect(downloads().length).toBeLessThanOrEqual(2)
  // Hidden and shown again while a download is pending: the same file is not requested twice.
  view.rerender(<CanonicalGroupWorkspace binding={binding} visible={false} />)
  view.rerender(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect(observers).toHaveLength(40))
  seen(observers[20], true)
  await act(async () => { await Promise.resolve() })
  expect(downloads().filter(id => id === 'img-0')).toHaveLength(1)
  // The shared request still paints the re-shown preview.
  await act(async () => { held.splice(0).forEach(release => release()) })
  await screen.findByRole('img', { name: 'shot-0.png' })
})

it('starts waiting previews in order as downloads succeed or fail, never more than two at once', async () => {
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:preview') })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
  const image = (n: number) => ({ attachment_id: `img-${n}`, kind: 'image', name: `shot-${n}.png`, mime: 'image/png', size: 1 })
  const events = Array.from({ length: 6 }, (_, n) => ({ seq: n + 1, room_id: 'drain', event_id: `e${n}`, kind: 'message.member', payload: { attachments: [image(n)] } }))
  const held = new Map<string, (ok: boolean) => void>()
  let running = 0
  let peak = 0
  request.mockImplementation(async (_route, method, params) => roomReads([], events)(method)
    ?? (method === 'groups.attachment.download' ? new Promise((resolve, reject) => {
      peak = Math.max(peak, ++running)
      held.set(params.attachment_id, ok => {
        running--

        if (ok) {resolve({ ...image(Number(params.attachment_id.slice(4))), data_base64: 'QQ==' })} else {reject(new Error('gone'))}
      })
    }) : {}))
  const observers: IntersectionObserverCallback[] = []
  vi.stubGlobal('IntersectionObserver', class { constructor(callback: IntersectionObserverCallback) { observers.push(callback) } observe() {} disconnect() {} })
  const downloads = () => request.mock.calls.filter(c => c[1] === 'groups.attachment.download').map(c => c[2].attachment_id)
  const settle = (outcomes: Record<string, boolean>) => act(async () => { Object.entries(outcomes).forEach(([id, ok]) => held.get(id)?.(ok)) })
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'remote', profile: 'team', roomId: 'drain' }} />)
  await waitFor(() => expect(observers).toHaveLength(6))

  // Every preview stays in view, so four wait behind the first two.
  for (const callback of observers) {act(() => callback([{ isIntersecting: true }] as IntersectionObserverEntry[], {} as IntersectionObserver))}
  await waitFor(() => expect(downloads()).toEqual(['img-0', 'img-1']))
  // A finished and a failed download each hand their slot to the next waiting preview.
  await settle({ 'img-0': true, 'img-1': false })
  await waitFor(() => expect(downloads()).toEqual(['img-0', 'img-1', 'img-2', 'img-3']))
  await settle({ 'img-3': false, 'img-2': true })
  await waitFor(() => expect(downloads()).toEqual(['img-0', 'img-1', 'img-2', 'img-3', 'img-4', 'img-5']))
  await settle({ 'img-4': true, 'img-5': true })
  await screen.findByRole('img', { name: 'shot-5.png' })
  expect(peak).toBe(2)
})

it('uploads a file pasted while another upload runs once that one finishes, instead of dropping it', async () => {
  let release!: () => void
  const receipt = (params: Record<string, string>) => ({ attachment_id: `att-${params.name}`, kind: params.kind, name: params.name, mime: params.mime, size: 1 })
  request.mockImplementation(async (_route, method, params) => roomReads()(method) ?? (method !== 'groups.attachment.upload' ? {}
    : params.name === 'first.png' ? new Promise(resolve => { release = () => resolve(receipt(params)) }) : receipt(params)))
  const uploads = () => request.mock.calls.filter(c => c[1] === 'groups.attachment.upload').map(c => c[2].name)
  render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'team', roomId: 'queue' }} />)
  const box = screen.getByRole('textbox')
  await waitFor(() => expect((box as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.paste(box, { clipboardData: { files: [new File(['a'], 'first.png', { type: 'image/png' })], types: ['Files'] } })
  await waitFor(() => expect(uploads()).toEqual(['first.png']))
  fireEvent.paste(box, { clipboardData: { files: [new File(['b'], 'second.txt', { type: 'text/plain' })], types: ['Files'] } })
  await act(async () => release())
  await screen.findByText('second.txt')
  expect(screen.getByText('first.png')).toBeTruthy()
  expect(uploads()).toEqual(['first.png', 'second.txt'])
})
