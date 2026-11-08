import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { closeoutCommand } from '@/lib/session-closeout-command'
import { $connection, setActiveSessionId, setSelectedStoredSessionId, setSessionOwnerHint, setSessions } from '@/store/session'
import { $sessionTiles, clearAllSessionStates, setSessionTileDelegate } from '@/store/session-states'
import { broadcastSessionRestored } from '@/store/session-sync'
import { makeSessionInfo } from '@/test/session-info'

import { actRender, Harness, type HarnessHandle } from './index-test-utils'

const { archive, request, profileRequest } = vi.hoisted(() => ({ archive: vi.fn(), request: vi.fn(), profileRequest: vi.fn() }))
vi.mock('@/store/gateway', async original => ({ ...(await original<object>()), requestGatewayForAgent: request, requestGatewayForProfile: profileRequest }))
vi.mock('@/hermes', async original => ({
  ...(await original<object>()),
  getLatestSessionMessages: vi.fn(async () => ({ messages: [] }))
}))

const receipt = {
  closeout_id: 'closeout',
  status: 'preserved',
  archived: true,
  settled: false,
  old: { logical_session_id: 'old', stored_session_id: 'old', runtime_id: 'rt-abc123', profile: 'work' },
  receipt_id: 'durable',
  destinations: { memory: 'pending', tasks: 'pending' },
  next_create_request_id: 'new-stable'
}

beforeEach(() => {
  archive.mockReset().mockImplementation(async (_stored, options) => {
    options?.onApplied?.()

    return receipt
  })
  request.mockReset().mockResolvedValue({ session_id: 'fresh-runtime', stored_session_id: 'fresh-stored' })
  $connection.set({
    mode: 'remote',
    connectionId: 'cloud-a',
    profile: 'work',
    wsUrl: 'wss://synthetic.invalid/api/ws'
  } as never)
  setSessionOwnerHint('old', { connectionId: 'cloud-a', profile: 'work' })
  setSelectedStoredSessionId('old')
  setSessionTileDelegate({ archiveSession: archive } as never)
})
afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $sessionTiles.set([])
  setSelectedStoredSessionId(null)
  setActiveSessionId(null)
  $connection.set(null)
  setSessions([])
})

async function mount() {
  let handle!: HarnessHandle
  await actRender(
    <Harness
      onReady={value => {
        handle = value
      }}
      refreshSessions={async () => {}}
      requestGateway={request}
      storedSessionId="old"
    />
  )

  return handle
}

it('explicit /new preserves once then creates an independent chat on the captured owner using the durable create id', async () => {
  const handle = await mount()
  await handle.submitText('/new')
  expect(archive).toHaveBeenCalledWith('old', expect.objectContaining({ intent: 'new' }))
  expect(request).toHaveBeenCalledWith(
    'cloud-a',
    'work',
    'session.create',
    expect.objectContaining({ request_id: 'new-stable', source: 'desktop', profile: 'work' })
  )
  expect($sessionTiles.get().some(tile => tile.storedSessionId === 'fresh-stored')).toBe(true)
})

it('creates and titles on captured bare profile A after focus moves to B without an ambient connection change', async () => {
  setSessions([makeSessionInfo({ id: 'profile-old', profile: 'profile-A' })])
  setSelectedStoredSessionId('profile-old')
  $sessionTiles.set([{ storedSessionId: 'profile-old', runtimeId: 'profile-runtime', ownerProfile: 'profile-A' }])
  let finish!: (value: unknown) => void
  archive.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  profileRequest.mockReset().mockResolvedValue({ session_id: 'profile-fresh-runtime', stored_session_id: 'profile-fresh' })
  const handle = await mount()
  const closing = handle.submitText('/new Title', { storedSessionId: 'profile-old' })
  await vi.waitFor(() => expect(finish).toBeTypeOf('function'))
  setSelectedStoredSessionId('B')
  finish({ ...receipt, old: { logical_session_id: 'profile-old', stored_session_id: 'profile-old', runtime_id: 'profile-runtime', profile: 'profile-A' } })
  expect(await closing).toBe(true)
  expect(profileRequest).toHaveBeenCalledWith('profile-A', 'session.create', expect.objectContaining({ request_id: 'new-stable' }), undefined, undefined)
  expect(profileRequest).toHaveBeenCalledWith('profile-A', 'session.title', expect.objectContaining({ title: 'Title' }), undefined, undefined)
  expect(request).not.toHaveBeenCalled()
  expect($sessionTiles.get().some(tile => tile.storedSessionId === 'profile-fresh')).toBe(false)
})

it.each(['archive', '/ARCHIVE', ' Archive this ', 'archive the chat'])(
  'only an exact archive invocation enters preservation: %s',
  async text => {
    const handle = await mount()
    await handle.submitText(text)
    expect(archive).toHaveBeenCalledWith('old', expect.objectContaining({ intent: 'archive' }))
    expect(request.mock.calls.some(call => call.includes('prompt.submit'))).toBe(false)
  }
)

it('a failed preservation leaves the current conversation and never creates a fresh chat', async () => {
  archive.mockResolvedValue(undefined)
  const handle = await mount()
  expect(await handle.submitText('/new')).toBe(false)
  expect(request.mock.calls.some(call => call.includes('session.create'))).toBe(false)
})

it('an explicit restore makes a delayed /new receipt harmless instead of creating or hiding a chat', async () => {
  let finish!: (value: unknown) => void
  archive.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const handle = await mount()
  const closing = handle.submitText('/new')
  await vi.waitFor(() => expect(finish).toBeTypeOf('function'))
  broadcastSessionRestored('old', 'cloud-a', 'work')
  finish(receipt)
  expect(await closing).toBe(false)
  expect(request).not.toHaveBeenCalled()
  expect(archive).toHaveBeenCalledTimes(1)
  expect($sessionTiles.get().some(tile => tile.storedSessionId === 'fresh-stored')).toBe(false)
})

it.each([false, true])('restoring while a new title is pending keeps its late completion harmless (reject=%s)', async reject => {
  let finish!: () => void
  request.mockImplementation(async (_connection, _profile, method) => {
    if (method === 'session.title') {
      await new Promise<void>((resolve, fail) => { finish = () => reject ? fail(new Error('title refused')) : resolve() })
    }

    return { session_id: 'fresh-runtime', stored_session_id: 'fresh-stored' }
  })
  const handle = await mount()
  const closing = handle.submitText('/new Title')
  await vi.waitFor(() => expect(finish).toBeTypeOf('function'))
  broadcastSessionRestored('old', 'cloud-a', 'work')
  finish()
  expect(await closing).toBe(false)
  expect(archive).toHaveBeenCalledTimes(1)
  expect($sessionTiles.get().some(tile => tile.storedSessionId === 'fresh-stored')).toBe(false)
})

it.each([
  '/reset',
  '/clear',
  'Please archive this',
  'How does /new work?',
  '"archive"',
  "'/new'",
  '```\n/archive\n```'
])('ordinary mentions, quotes and existing reset commands do not trigger closeout: %s', text => {
  expect(closeoutCommand(text)).toBeNull()
})

it('a user moving to another chat during review cannot have that chat replaced by a stale completion', async () => {
  let finish!: (value: unknown) => void
  archive.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  const handle = await mount()
  const closing = handle.submitText('/new')
  await vi.waitFor(() => expect(finish).toBeTypeOf('function'))
  setSelectedStoredSessionId('another-chat')
  finish(receipt)
  await closing
  expect($sessionTiles.get().some(tile => tile.storedSessionId === 'fresh-stored')).toBe(false)
})

it('Bot /new creates an untitled sidechat on the same exact owner, never adopts Bot Chat', async () => {
  $sessionTiles.set([
    {
      storedSessionId: 'old',
      runtimeId: 'rt-abc123',
      ownerRoute: { connectionId: 'cloud-a', profile: 'work' },
      workspaceMode: 'bots',
      workspaceOwnerKey: 'cloud-a/work'
    }
  ])
  const handle = await mount()
  await handle.submitText('/new Bot Chat')
  const create = request.mock.calls.find(call => call.includes('session.create'))
  expect(create?.[3]).not.toHaveProperty('title')
  expect($sessionTiles.get().find(tile => tile.storedSessionId === 'fresh-stored')).toMatchObject({
    workspaceMode: 'bots',
    workspaceOwnerKey: 'cloud-a/work',
    ownerRoute: { connectionId: 'cloud-a', profile: 'work' }
  })
})

it('never reuses an archived logical owner returned under a new physical segment', async () => {
  request.mockResolvedValue({ session_id: 'old', stored_session_id: 'new-physical-segment' })
  const handle = await mount()
  expect(await handle.submitText('/new')).toBe(false)
  expect($sessionTiles.get().some(tile => tile.storedSessionId === 'new-physical-segment')).toBe(false)
})

it('an optional title is validated only after independent creation, so an existing title cannot adopt another chat', async () => {
  const handle = await mount()
  await handle.submitText('/new Existing title')
  const create = request.mock.calls.find(call => call.includes('session.create'))
  expect(create?.[3]).not.toHaveProperty('title')
  expect(request).toHaveBeenCalledWith('cloud-a', 'work', 'session.title', {
    session_id: 'fresh-runtime',
    profile: 'work',
    title: 'Existing title'
  })
})

it('compression rotating the displayed stored segment does not cancel the same focused conversation closeout', async () => {
  let finish!: (value: unknown) => void
  archive.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  setActiveSessionId('rt-abc123')
  const handle = await mount()
  const closing = handle.submitText('/new')
  await vi.waitFor(() => expect(finish).toBeTypeOf('function'))
  setSelectedStoredSessionId('compressed-tip')
  finish(receipt)
  await closing
  expect($sessionTiles.get().some(tile => tile.storedSessionId === 'fresh-stored')).toBe(true)
})
