import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps, ReactNode } from 'react'
import type * as HermesSdk from '@hermes/plugin-sdk'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const { request, routes } = vi.hoisted(() => ({
  request: vi.fn(),
  routes: { value: [] as Array<Record<string, unknown>> }
}))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')

  return { ...sdk, ...await pluginSdkMock(gateway.host), atom, useValue: useStore,
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => (key: string) => key === 'group.checkAgain'
      ? 'Check again'
      : CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Codicon: () => <span />,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>,
    host: { ...gateway.host, profileRoutes: async () => routes.value, requestProfile: request } }

})
import { $canonicalGroupBindings, registerCanonicalGroup } from './canonical-group-registry'
import { prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { $groupChats } from './group-chat'
import { GroupChatWorkspace } from './group-chat-view'
import { scriptedStorage } from './group-test-utils'
import { startHostedRoomRuntime, stopHostedRoomRuntime } from './hosted-room-runtime'

const originalDesktop = window.hermesDesktop
beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
  routes.value = []
  $canonicalGroupBindings.set({})
  $groupChats.set({})
})
afterEach(() => {
  stopHostedRoomRuntime()
  cleanup()
  request.mockReset()
  localStorage.clear()
  window.hermesDesktop = originalDesktop
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

it('renders mixed-gateway recovery in the registered canonical workspace and runs the real bounded check', async () => {
  routes.value = [
    { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' },
    { connectionId: 'gateway-b', mode: 'remote', profile: 'default', targetProfile: 'default' }
  ]
  const members = [
    { handle: 'research', member_id: 'research', profile: 'research' },
    {
      display_name: 'Remote Builder',
      handle: 'builder',
      member_id: 'builder',
      profile: 'builder',
      target: { installation_id: 'install:peer', kind: 'peer', peer_id: 'install:peer' }
    }
  ]
  const serverRoom = {
    authority_epoch: 1,
    authority_gateway_id: 'install:home',
    latest_seq: 0,
    members,
    name: 'Release',
    room_id: 'room-1'
  }
  let upgraded = false
  let holdUpgrade = false
  let upgradeStarted!: () => void
  let releaseUpgrade!: () => void
  const upgradeRequest = new Promise<void>(resolve => { upgradeStarted = resolve })
  const upgradeResponse = new Promise<Record<string, unknown>>(resolve => {
    releaseUpgrade = () => resolve({
      authority_gateway_id: 'install:peer',
      driver: true,
      methods: ['groups.peer.revoke_exact'],
      persistent_process: true
    })
  })

  request.mockImplementation(async (route, method) => {
    const connectionId = String(route?.connectionId || '')

    if (method === 'groups.capabilities') {
      if (connectionId === 'gateway-b') {
        if (!upgraded) {throw Object.assign(new Error('Method not found'), { code: -32601 })}

        if (holdUpgrade) {upgradeStarted(); return upgradeResponse}

        return {
          authority_gateway_id: 'install:peer',
          driver: true,
          methods: ['groups.peer.revoke_exact'],
          persistent_process: true
        }
      }

      return {
        authority_gateway_id: 'install:home',
        driver: true,
        features: ['peer_route_grant_fingerprint'],
        persistent_process: true
      }
    }

    if (method === 'groups.list') {return connectionId === 'gateway-b' ? { rooms: [] } : { rooms: [serverRoom] }}
    if (method === 'groups.state') {return { driver_status: { peer_routes: [{ member_id: 'builder', status: 'needs_reauthorization' }] }, room: serverRoom }}
    if (method === 'groups.log') {return { events: [], has_more: false, latest_seq: 0 }}
    throw new Error(`Unexpected method: ${method}`)
  })

  await startHostedRoomRuntime(scriptedStorage(new Map()).storage)
  const key = registerCanonicalGroup(
    { connectionId: 'gateway-a', profile: 'default' },
    { room_id: 'room-1', name: 'Release', members }
  )
  render(<GroupChatWorkspace group={key} members={[]} />)

  await screen.findByText('Update this device to keep this Group Chat running.')
  const check = screen.getByRole('button', { name: 'Check again' }) as HTMLButtonElement
  upgraded = true
  holdUpgrade = true
  fireEvent.click(check)
  await upgradeRequest
  expect(check.getAttribute('aria-busy')).toBe('true')
  expect(check.disabled).toBe(true)
  expect(request.mock.calls.filter(call => call[1] === 'groups.send')).toEqual([])

  releaseUpgrade()
  await waitFor(() => expect(screen.getByText('Reconnect Remote Builder to continue this Group Chat.')).toBeTruthy())
  expect(screen.queryByRole('button', { name: 'Check again' })).toBeNull()
  expect(request.mock.calls.filter(call => call[1] === 'groups.capabilities' && call[0]?.connectionId === 'gateway-b')).toHaveLength(2)
})
