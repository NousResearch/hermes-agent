import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import * as registryModule from './canonical-group-registry'
import * as workspaceModule from './canonical-group-workspace'
import type { CanonicalGroupBinding } from './canonical-groups'
import * as chatModule from './group-chat'
import { scriptedStorage } from './group-test-utils'
import * as sharedModule from './shared'
import * as adoptionModule from './shipped-group-adoption'

const runtime = vi.hoisted(() => ({
  activation: 1,
  authority: 'install:owner-a',
  connectionId: 'owner-a',
  gateway: 'open',
  handler: async (_route: unknown, _method: string, _params: Record<string, unknown>): Promise<unknown> => ({}),
  profile: 'default',
  routeGeneration: 1,
  routeListeners: new Set<(event: { connectionId: string; profile: string; state: string }) => void>(),
  routes: [
    { connectionId: 'owner-a', mode: 'local', profile: 'default', targetProfile: 'default' },
    { connectionId: 'remote-b', mode: 'remote', profile: 'default', targetProfile: 'default' }
  ] as Array<{ connectionId: string; mode: 'local' | 'remote'; profile: string; targetProfile: string }>
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()
  const { pluginSdkMock } = await import('./group-test-utils')

  const sdk = await pluginSdkMock({
    ...original.host,
    activeConnectionId: () => runtime.connectionId,
    retainProfileSocket: () => () => undefined,
    onProfileRouteState: (listener: (event: { connectionId: string; profile: string; state: string }) => void) => {
      runtime.routeListeners.add(listener)

      return () => { runtime.routeListeners.delete(listener) }
    },
    acquireProfileRoute: async (route: any) => {
      const generation = runtime.routeGeneration
      let released = false

      const assertCurrent = () => {
        if (released || generation !== runtime.routeGeneration) {
          throw new Error('Hermes gateway route lease expired')
        }
      }

      return {
        generation,
        route: { ...route },
        assertCurrent,
        release: () => {
          released = true
        },
        request: async (method: string, params: Record<string, unknown>) => {
          assertCurrent()
          const result = await runtime.handler(route, method, params)
          assertCurrent()

          return result
        }
      }
    },
    profileRoutes: async () => runtime.routes,
    requestProfile: (route: unknown, method: string, params: Record<string, unknown>) =>
      runtime.handler(route, method, params),
    state: {
      connectionId: { get: () => runtime.connectionId, listen: () => () => undefined },
      gateway: { get: () => runtime.gateway, listen: () => () => undefined },
      profile: { get: () => runtime.profile, listen: () => () => undefined }
    }
  })

  const { useStore } = await import('@nanostores/react')

  return {
    ...original,
    ...sdk,
    Button: (props: React.ComponentProps<'button'>) => <button {...props} />,
    Codicon: () => <span />,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>,
    gatewayActivationEpoch: () => runtime.activation,
    useI18n: () => ({
      locale: 'en',
      t: {
        common: { back: 'Back', cancel: 'Cancel', refresh: 'Refresh', retry: 'Retry', send: 'Send' },
        composer: { queueLostDiscard: 'Discard', stop: 'Stop' },
        fileMenu: { download: 'Download' }
      }
    }),
    usePluginI18n: () => (key: string) => {
      if (key === 'group.checkAgain') {
        return 'Check again'
      }

      const value = CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en]

      return value ?? key
    },
    useValue: useStore
  }
})

interface ImportRequest {
  room_id: string
  name: string
  source_id: string
  members: Array<{
    source_member_id: string
    name: string
    profile: string
    handle: string
    remote_source: boolean
    active: boolean
    connection_id?: string
    connection_label?: string
  }>
  history: Array<{
    source_entry_id: string
    at_ms: number
    author_kind: 'member' | 'user'
    author_name: string
    member_source_id?: string
    text: string
    thread_id: string
    attachments?: Array<{ kind: string; name: string; data: string }>
  }>
  held_work: Array<{
    source_work_id: string
    at_ms: number
    state: 'uncertain'
    description: string
    member_source_id?: string
  }>
}

interface ServerMember {
  member_id: string
  profile: string
  handle: string
  display_name: string
  membership: { state: 'active' | 'former' | 'retiring' }
  availability: { state: 'authorization_required' | 'ready' | 'retired'; reason?: string }
  source: { remote_source: boolean; source_member_id: string; connection_id?: string; connection_label?: string }
}

async function releasedRecord() {
  const { durableGroupChatMembers } = await import('./group-membership')

  const members = durableGroupChatMembers([
    {
      connectionId: 'owner-a',
      connectionLabel: 'This Mac',
      display_name: 'Reviewer',
      handle: 'alpha',
      name: 'alpha',
      route: { connectionId: 'owner-a', mode: 'local', profile: 'alpha', targetProfile: 'alpha' },
      targetProfile: 'alpha'
    },
    {
      connectionId: 'owner-a',
      connectionLabel: 'This Mac',
      display_name: 'Reviewer',
      handle: 'beta',
      name: 'beta',
      route: { connectionId: 'owner-a', mode: 'local', profile: 'beta', targetProfile: 'beta' },
      targetProfile: 'beta'
    },
    {
      connectionId: 'remote-b',
      connectionLabel: 'Workshop Mac',
      display_name: 'Remote Builder',
      handle: 'builder',
      name: 'builder',
      route: { connectionId: 'remote-b', mode: 'remote', profile: 'builder', targetProfile: 'builder' },
      targetProfile: 'builder'
    }
  ])

  return {
    Release: {
      log: [
        {
          at: 1_700_000_000,
          from: { kind: 'user', name: 'You' },
          text: 'Keep this shipped history',
          images: [{ name: 'photo.png', data: 'data:image/png;base64,UE5H' }]
        },
        { at: 1_700_000_001, from: { kind: 'member', name: 'alpha' }, text: 'Alpha result' },
        { at: 1_700_000_002, from: { kind: 'member', name: 'beta' }, text: 'Beta result' },
        { at: 1_700_000_003, from: { kind: 'member', name: 'departed' }, text: 'Former result' }
      ],
      members,
      sessions: { builder: 'session-before-upgrade' },
      stranded: { builder: { before: 3, thread: 'legacy' } },
      watermarks: {}
    }
  }
}

function serverMember(member: ImportRequest['members'][number]): ServerMember {
  const former = member.active === false
  const remote = member.remote_source

  return {
    member_id: `server:${member.source_member_id}`,
    profile: member.profile,
    handle: member.handle,
    display_name: member.name,
    membership: { state: former ? 'former' : 'active' },
    availability: former
      ? { state: 'retired', reason: 'former_member' }
      : remote
        ? { state: 'authorization_required', reason: 'remote_execution_not_authorized' }
        : { state: 'ready' },
    source: {
      source_member_id: member.source_member_id,
      remote_source: remote,
      ...(member.connection_id ? { connection_id: member.connection_id } : {}),
      ...(member.connection_label ? { connection_label: member.connection_label } : {})
    }
  }
}

function backend({ failAfterFirstCommit = false }: { failAfterFirstCommit?: boolean } = {}) {
  const calls: Array<{ route: any; method: string; params: Record<string, unknown> }> = []
  const imports: ImportRequest[] = []
  let committed: ImportRequest | null = null
  let firstCommitFailed = false
  let releaseImport: null | (() => void) = null
  let holdImport = false
  let retiringMember = ''

  const room = () => {
    if (!committed) {
      throw new Error('room not imported')
    }

    const members = committed.members.map(serverMember).map(member =>
      member.display_name === retiringMember
        ? {
            ...member,
            membership: { state: 'retiring' as const },
            availability: { state: 'authorization_required' as const, reason: 'member_retirement_pending' }
          }
        : member
    )

    return {
      room_id: committed.room_id,
      name: committed.name,
      members,
      authority_gateway_id: runtime.authority,
      authority_epoch: 1,
      revision: 1,
      created_at: 1,
      updated_at: 1
    }
  }

  const events = () => {
    if (!committed) {
      return []
    }

    const ids = new Map(committed.members.map(member => [member.source_member_id, `server:${member.source_member_id}`]))

    const imported = committed.history.map((entry, index) => ({
      room_id: committed!.room_id,
      seq: index + 1,
      event_id: `history:${index}`,
      kind: 'history.imported',
      actor: { kind: 'system', id: 'desktop-history-import' },
      authority_epoch: 1,
      payload: {
        author: {
          kind: entry.author_kind,
          name: entry.author_name,
          ...(entry.member_source_id ? { member_id: ids.get(entry.member_source_id) } : {})
        },
        text: entry.text,
        thread_id: entry.thread_id,
        ...(entry.attachments?.length
          ? {
              attachments: entry.attachments.map((attachment, attachmentIndex) => ({
                attachment_id: `att_${String(index * 10 + attachmentIndex).padStart(32, '0')}`,
                kind: attachment.kind,
                name: attachment.name,
                mime: attachment.data.slice(5, attachment.data.indexOf(';')),
                size: 3
              }))
            }
          : {})
      },
      created_at: entry.at_ms / 1000
    }))

    const held = committed.held_work.map((work, index) => ({
      room_id: committed!.room_id,
      seq: imported.length + index + 1,
      event_id: `held:${index}`,
      kind: 'history.held',
      actor: { kind: 'system', id: 'desktop-history-import' },
      authority_epoch: 1,
      payload: {
        state: 'uncertain',
        description: work.description,
        action: 'review_before_retry'
      },
      created_at: work.at_ms / 1000
    }))

    return [...imported, ...held]
  }

  runtime.handler = async (route, method, params) => {
    calls.push({ route, method, params: structuredClone(params) })

    if (method === 'groups.capabilities') {
      return {
        authority_gateway_id: runtime.authority,
        driver: true,
        methods: [
          'groups.capabilities',
          'groups.import_history',
          'groups.member.resolve',
          'groups.state',
          'groups.log',
          'groups.send'
        ],
        persistent_process: true
      }
    }

    if (method === 'groups.import_history') {
      const request = structuredClone(params) as unknown as ImportRequest
      imports.push(request)

      if (holdImport) {
        await new Promise<void>(resolve => {
          releaseImport = resolve
        })
      }

      if (!committed) {
        committed = request
      } else {
        expect(request).toEqual(committed)
      }

      if (failAfterFirstCommit && !firstCommitFailed) {
        firstCommitFailed = true
        throw new Error('Connection closed after server commit')
      }

      const members = room().members as ServerMember[]

      return {
        room: { ...room(), idempotent: imports.length > 1 },
        source_id: request.source_id,
        imported_history: request.history.length,
        held_work: request.held_work.length,
        held_members: members.filter(member => member.availability.state === 'authorization_required').length,
        retired_members: members.filter(member => member.availability.state === 'retired').length,
        idempotent: imports.length > 1
      }
    }

    if (method === 'groups.state') {
      return { room: room(), driver_status: { pending_actions: [] } }
    }

    if (method === 'groups.send') {
      return { event: { event_id: params.event_id, payload: params.payload } }
    }

    if (method === 'groups.log') {
      return { events: events(), cursor: events().length, latest_seq: events().length, has_more: false }
    }

    if (method === 'groups.attachment.list') {
      return {
        room_id: room().room_id,
        authority: { gateway_id: runtime.authority, epoch: 1 },
        snapshot_seq: events().length,
        items: [],
        has_more: false,
        next_cursor: null
      }
    }

    if (method === 'groups.attachment.download') {
      return {
        attachment_id: params.attachment_id,
        event_id: params.event_id,
        kind: 'image',
        name: 'photo.png',
        mime: 'image/png',
        size: 3,
        data_base64: 'UE5H'
      }
    }

    if (method === 'groups.member.resolve') {
      return {
        room: room(),
        member: room().members.find(member => member.member_id === params.member_id),
        action: params.action,
        changed: false
      }
    }

    throw new Error(`Unexpected method ${method}`)
  }

  return {
    calls,
    imports,
    setHold(value: boolean) {
      holdImport = value
    },
    setRetiring(name: string) {
      retiringMember = name
    },
    release() {
      releaseImport?.()
      releaseImport = null
    },
    committed: () => committed
  }
}

async function modules() {
  // Compile the real module graph during collection, not in the first state-reset hook.
  return {
    adoption: adoptionModule,
    chat: chatModule,
    registry: registryModule,
    shared: sharedModule,
    workspace: workspaceModule
  }
}

async function coldHydrate(storage: Map<string, unknown>) {
  const loaded = await modules()
  loaded.shared.setPluginCtx(scriptedStorage(storage))
  loaded.chat.setGroupChatSyncDisposed(false)
  loaded.chat.$groupChats.set(loaded.chat.hydrateGroupChatRooms(storage.get('group-chats')))
  await loaded.chat.activateClassicGroupAuthorities()
  loaded.chat.stopGroupChatServerSync()

  return loaded
}

beforeEach(async () => {
  runtime.routeListeners.clear()
  runtime.activation += 1
  runtime.authority = 'install:owner-a'
  runtime.connectionId = 'owner-a'
  runtime.gateway = 'open'
  runtime.profile = 'default'
  runtime.routeGeneration += 1
  runtime.routes = [
    { connectionId: 'owner-a', mode: 'local', profile: 'default', targetProfile: 'default' },
    { connectionId: 'remote-b', mode: 'remote', profile: 'default', targetProfile: 'default' }
  ]
  localStorage.clear()
  Element.prototype.scrollIntoView = vi.fn()
  URL.createObjectURL = vi.fn(() => 'blob:history')
  URL.revokeObjectURL = vi.fn()
  const loaded = await modules()
  loaded.adoption.stopShippedGroupAdoption()
  loaded.registry.$canonicalGroupBindings.set({})
  loaded.chat.$groupChats.set({})
}, 30_000) // Cold SDK transforms can exceed the default hook budget on a loaded runner.

afterEach(async () => {
  cleanup()
  vi.restoreAllMocks()
  const loaded = await modules()
  loaded.adoption.stopShippedGroupAdoption()
  loaded.registry.$canonicalGroupBindings.set({})
  loaded.chat.$groupChats.set({})
})

describe('automatic shipped Group Chat adoption', () => {
  it.each(['remote', 'scoped', 'missing-owner'])('keeps a hydrated %s orphan intact instead of adopting a same-named local replacement', async shape => {
    const transport = backend()
    const record = await releasedRecord()
    Object.assign(record.Release, { roomId: `r${'1'.repeat(32)}` })
    // A valid local builder must not lend its owner to the unresolved builder.
    record.Release.members.push({ name: 'builder', handle: 'local-builder', connectionId: 'owner-a', remoteSource: true } as any)
    record.Release.members.push({
      name: 'builder', handle: 'lost-builder',
      ...(shape === 'remote' ? { remoteSource: true } : {}),
      ...(shape === 'scoped' ? { sourceScoped: true } : {}),
      ...(shape === 'missing-owner' ? { sourceMissing: true, connectionId: 'owner-a' } : {})
    } as any)
    const storage = new Map<string, unknown>([['group-chats', record]])
    const loaded = await coldHydrate(storage)
    const { annotateOrphanedGroupChatMembers } = await import('./hygiene')
    const annotated = annotateOrphanedGroupChatMembers(loaded.chat.$groupChats.get(), new Set(['owner-a', 'remote-b']))
    loaded.chat.$groupChats.set(annotated.rooms)
    const original = structuredClone(annotated.rooms.Release)
    expect(original.members!.at(-1)?.sourceMissing).toBe(true)
    const route = { connectionId: 'owner-a', profile: 'default' }
    const room = { room_id: original.roomId!, name: 'Release', members: [] }
    const oldAlias = loaded.registry.registerCanonicalGroup(route, room)
    const oldBinding = loaded.registry.$canonicalGroupBindings.get()[oldAlias]
    expect(oldBinding.isCurrent?.()).toBe(true)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    expect(oldBinding.isCurrent?.()).toBe(false)
    expect(loaded.registry.registerCanonicalGroup(route, room)).toBe('Release')
    const { canonicalGroupRequest } = await import('./canonical-groups')

    for (const method of ['groups.send', 'groups.member.resolve', 'groups.member.remove', 'groups.disband']) {
      await expect(canonicalGroupRequest(oldBinding, method, { room_id: room.room_id }))
        .rejects.toThrow('no longer current')
    }

    const view = await import('./group-chat-view')

    for (const group of [oldAlias, 'Release']) {
      render(<view.GroupChatWorkspace group={group} members={[]} />)
      expect(screen.queryByRole('textbox')).toBeNull()
      expect(screen.getByRole('log').textContent).toContain('Keep this shipped history')
      expect(screen.getByRole('status').textContent).toContain('unresolved source owner')
      cleanup()
    }

    expect(transport.imports).toHaveLength(0)
    expect(transport.calls.some(call => ['groups.send', 'groups.member.resolve'].includes(call.method))).toBe(false)
    expect(loaded.registry.$canonicalGroupBindings.get()).toEqual({})
    const retained = (storage.get('group-chats') as Record<string, any>).Release
    expect(retained.members).toEqual(original.members)
    expect(retained.log).toEqual(original.log)
    expect(retained.stranded).toEqual(original.stranded)
    expect(retained.shippedAdoption.state).not.toBe('adopted')
  })

  it.each(['waiting', 'prepared', 'conflict'] as const)('hydrated %s checkpoints block discovery and already-mounted aliases', async stage => {
    const transport = backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const original = loaded.chat.$groupChats.get().Release
    const built = await loaded.adoption.buildShippedGroupImport('Release', original, 'owner-a')
    const route = { connectionId: 'owner-a', profile: 'default' }
    const room = { room_id: built.request.room_id, name: 'Release', members: [] }
    const alias = loaded.registry.registerCanonicalGroup(route, room)
    const binding = loaded.registry.$canonicalGroupBindings.get()[alias]
    const view = await import('./group-chat-view')
    const mounted = render(<view.GroupChatWorkspace group={alias} members={[]} />)

    const checkpoint = {
      version: 1 as const, state: stage === 'prepared' ? 'prepared' as const : 'waiting' as const,
      sourceId: built.request.source_id, roomId: room.room_id, requestHash: built.requestHash,
      ...(stage === 'prepared' ? { route: { ...route, authorityGatewayId: runtime.authority } } : {}),
      ...(stage === 'conflict' ? { issue: { kind: 'conflict' as const, message: 'Original owner unresolved' } } : {})
    }

    // Hydration can publish ownership without running a checkpoint writer.
    await act(async () => {
      loaded.chat.$groupChats.set({ Release: { ...original, shippedAdoption: checkpoint } })
    })
    expect(binding.isCurrent?.()).toBe(false)
    expect(loaded.registry.registerCanonicalGroup(route, room)).toBe('Release')
    mounted.rerender(<view.GroupChatWorkspace group={alias} members={[]} />)
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.getByRole('log').textContent).toContain('Keep this shipped history')
    const { canonicalGroupRequest } = await import('./canonical-groups')
    await expect(canonicalGroupRequest(binding, 'groups.send', { room_id: room.room_id })).rejects.toThrow('no longer current')
    const unrelated = loaded.registry.registerCanonicalGroup(route, { ...room, room_id: 'unrelated-room' })
    expect(loaded.registry.$canonicalGroupBindings.get()[unrelated].isCurrent?.()).toBe(true)
    const otherRoute = { ...route, profile: 'other-profile' }
    const other = loaded.registry.registerCanonicalGroup(otherRoute, room)

    if (stage === 'prepared') {
      expect(loaded.registry.$canonicalGroupBindings.get()[other].isCurrent?.()).toBe(true)
    } else {
      expect(other).toBe('Release')
    }

    expect(transport.calls.some(call => ['groups.send', 'groups.member.resolve'].includes(call.method))).toBe(false)
  })

  it.each([false, true])('quarantines aliases before checkpoint storage settles, including rollback: %s', async fail => {
    const transport = backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const original = loaded.chat.$groupChats.get().Release
    const built = await loaded.adoption.buildShippedGroupImport('Release', original, 'owner-a')
    const route = { connectionId: 'owner-a', profile: 'default' }
    const room = { room_id: built.request.room_id, name: 'Release', members: [] }
    const alias = loaded.registry.registerCanonicalGroup(route, room)
    const binding = loaded.registry.$canonicalGroupBindings.get()[alias]
    const ctx = scriptedStorage(storage)
    const write = ctx.storage.set
    let release!: () => void
    const blocked = new Promise<void>(resolve => { release = resolve })

    const waitingWrite = vi.spyOn(ctx.storage, 'set').mockImplementation(async (key, value) => {
      if (key === 'group-chats' && (value as Record<string, any>).Release?.shippedAdoption?.state === 'waiting') {
        await blocked

        if (fail) { throw new Error('checkpoint could not be saved') }
      }

      return write(key, value)
    })

    const adoption = loaded.adoption.adoptShippedGroupChats(ctx.storage)

    try {
      await waitFor(() => expect(waitingWrite).toHaveBeenCalled())
      expect(binding.isCurrent?.()).toBe(false)
      expect(loaded.registry.$canonicalGroupBindings.get()[alias]).toBeUndefined()
      expect(loaded.registry.registerCanonicalGroup(route, room)).toBe('Release')
      expect(transport.imports).toHaveLength(0)
    } finally {
      release()
      await adoption
    }

    expect(binding.isCurrent?.()).toBe(false)

    if (fail) {
      expect(transport.imports).toHaveLength(0)
      expect(loaded.chat.$groupChats.get().Release.log).toEqual(original.log)
      expect(loaded.chat.$groupChats.get().Release.shippedAdoption).toBeUndefined()
    } else {
      expect(loaded.registry.$canonicalGroupBindings.get().Release.isCurrent?.()).toBe(true)
    }
  })

  it.each([false, true])('canonical discovery reuses the adopted Send lease, refusing replacement before retry: %s', async replace => {
    const transport = backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const built = await loaded.adoption.buildShippedGroupImport('Release', loaded.chat.$groupChats.get().Release, 'owner-a')

    const earlierAlias = loaded.registry.registerCanonicalGroup(
      { connectionId: 'owner-a', profile: 'default' }, { room_id: built.request.room_id, name: 'Release', members: [] })

    const earlierBinding = loaded.registry.$canonicalGroupBindings.get()[earlierAlias]
    expect(earlierBinding.isCurrent?.()).toBe(true)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    expect(earlierBinding.isCurrent?.()).toBe(false)
    const binding = loaded.registry.$canonicalGroupBindings.get().Release!
    const journal = await import('./canonical-group-send')
    const prepared = await journal.prepareCanonicalGroupSend(binding, { text: 'private owner A intent' })
    const retained = localStorage.getItem('hermes.desktop.canonicalGroupSends.v1')
    const route = { connectionId: binding.connectionId, profile: binding.profile }
    const room = { room_id: binding.roomId, name: 'Release', members: [] }
    const alias = loaded.registry.registerCanonicalGroup(route, room)
    expect(loaded.registry.$canonicalGroupBindings.get()[alias]).toBe(binding)
    render(<loaded.workspace.CanonicalGroupWorkspace binding={loaded.registry.$canonicalGroupBindings.get()[alias]} />)
    await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('private owner A intent'))
    const send = screen.getByRole('button', { name: 'Retry' })
    await waitFor(() => expect((send as HTMLButtonElement).disabled).toBe(false))

    if (replace) {
      runtime.authority = 'install:replacement'
      runtime.routeGeneration += 1
    }

    fireEvent.click(send)

    if (replace) {
      expect(transport.calls.filter(call => call.method === 'groups.send')).toHaveLength(0)
      expect(localStorage.getItem('hermes.desktop.canonicalGroupSends.v1')).toBe(retained)
      loaded.registry.revokeStaleAdoptedCanonicalGroups()
      expect(loaded.registry.registerCanonicalGroup(route, room)).toBe('Release')
      expect(loaded.registry.$canonicalGroupBindings.get()).toEqual({})
      cleanup()
      const view = await import('./group-chat-view')
      render(<view.GroupChatWorkspace group="Release" members={[]} />)
      expect(screen.queryByRole('textbox')).toBeNull()
      expect(screen.getByText(CANONICAL_GROUP_LOCALES.en.upgradeChecking)).toBeTruthy()
    } else {
      await waitFor(() => expect(transport.calls.filter(call => call.method === 'groups.send')).toHaveLength(1))
      expect(transport.calls.find(call => call.method === 'groups.send')?.params).toMatchObject(prepared.params)
      await waitFor(async () => expect(await journal.readCanonicalGroupSend(binding)).toBeUndefined())
    }
  })

  it('cold-hydrates a released record, retries the exact committed request, and mounts retained history in the canonical consumer', async () => {
    const transport = backend({ failAfterFirstCommit: true })
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    let loaded = await coldHydrate(storage)

    await Promise.all([
      loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage),
      loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    ])

    expect(transport.imports).toHaveLength(1)
    const interrupted = storage.get('group-chats') as Record<string, any>
    expect(interrupted.Release.shippedAdoption.state).toBe('prepared')
    expect(interrupted.Release.log).toHaveLength(4)
    expect(interrupted.Release.shippedAdoption.issue.kind).toBe('offline')

    // A renderer restart has only the durable record. The server already committed,
    // so the client must retry byte-for-byte on the same owner instead of minting.
    loaded.adoption.stopShippedGroupAdoption()
    loaded.registry.$canonicalGroupBindings.set({})
    loaded.chat.$groupChats.set({})
    loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)

    expect(transport.imports).toHaveLength(2)
    expect(transport.imports[1]).toEqual(transport.imports[0])
    const request = transport.imports[0]
    expect(request.source_id).toMatch(/^hermes\.plugin\.hermes-bots\.group-chats:/)
    expect(request.history.map(entry => entry.text)).toEqual([
      'Keep this shipped history',
      'Alpha result',
      'Beta result',
      'Former result'
    ])
    expect(request.history.map(entry => entry.thread_id)).toEqual(['legacy', 'legacy', 'legacy', 'legacy'])
    expect(request.history[0].attachments).toEqual([
      { kind: 'image', name: 'photo.png', data: 'data:image/png;base64,UE5H' }
    ])
    expect(request.members.filter(member => member.name === 'Reviewer')).toHaveLength(2)
    expect(
      new Set(request.members.filter(member => member.name === 'Reviewer').map(member => member.source_member_id)).size
    ).toBe(2)
    expect(request.members.find(member => member.name === 'Remote Builder')).toMatchObject({
      active: true,
      connection_id: 'remote-b',
      connection_label: 'Workshop Mac',
      remote_source: true
    })
    expect(
      request.members.filter(member => member.name === 'Reviewer').every(member => member.remote_source === false)
    ).toBe(true)
    expect(request.members.find(member => member.name === 'departed')).toMatchObject({ active: false })
    expect(request.held_work).toHaveLength(1)

    const adopted = storage.get('group-chats') as Record<string, any>
    expect(adopted.Release.shippedAdoption).toMatchObject({
      state: 'adopted',
      sourceId: request.source_id,
      roomId: request.room_id,
      route: { connectionId: 'owner-a', profile: 'default', authorityGatewayId: 'install:owner-a' }
    })
    expect(adopted.Release.log).toHaveLength(4)
    expect(loaded.registry.$canonicalGroupBindings.get().Release).toMatchObject({
      connectionId: 'owner-a',
      profile: 'default',
      roomId: request.room_id
    })
    expect(loaded.registry.$canonicalGroupBindings.get().Release.adoptionOwner).toMatchObject({
      authorityGatewayId: 'install:owner-a',
      requestHash: adopted.Release.shippedAdoption.requestHash,
      sourceId: adopted.Release.shippedAdoption.sourceId
    })

    // A later cold launch restores the same canonical binding without importing again.
    loaded.adoption.stopShippedGroupAdoption()
    loaded.registry.$canonicalGroupBindings.set({})
    loaded.chat.$groupChats.set(loaded.chat.hydrateGroupChatRooms(storage.get('group-chats')))
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    expect(transport.imports).toHaveLength(2)

    const binding = loaded.registry.$canonicalGroupBindings.get().Release as CanonicalGroupBinding
    render(<loaded.workspace.CanonicalGroupWorkspace binding={binding} />)
    const log = await screen.findByRole('log')
    const text = log.textContent || ''
    expect(text.indexOf('Keep this shipped history')).toBeLessThan(text.indexOf('Alpha result'))
    expect(text.indexOf('Alpha result')).toBeLessThan(text.indexOf('Beta result'))
    expect(text.indexOf('Beta result')).toBeLessThan(text.indexOf('Former result'))
    expect(within(log).getByText('photo.png')).toBeTruthy()
    expect(within(log).getByText(/held for review/i)).toBeTruthy()
    const memberList = await screen.findByRole('region', { name: 'Members' })
    expect(within(memberList).getAllByText('Reviewer')).toHaveLength(2)
    const remoteMember = within(memberList).getByText('Remote Builder').closest('div')!
    expect(remoteMember.textContent).toContain('Authorization required')
    expect(within(memberList).getByText('departed').closest('div')?.textContent).toContain('Former member')

    fireEvent.click(within(remoteMember).getByRole('button', { name: 'Check again' }))
    await waitFor(() =>
      expect(
        transport.calls.some(call => call.method === 'groups.member.resolve' && call.params.action === 'refresh')
      ).toBe(true)
    )
    expect(transport.imports).toHaveLength(2)
    expect(transport.calls.some(call => call.method === 'groups.send')).toBe(false)

    fireEvent.click(within(log).getByRole('button', { name: 'Download' }))
    await waitFor(() => expect(transport.calls.some(call => call.method === 'groups.attachment.download')).toBe(true))
    expect(transport.calls.some(call => call.method === 'groups.send')).toBe(false)
  })

  it('coalesces overlapping adoption and leaves a late success unacknowledged after the foreground source moves', async () => {
    const transport = backend()
    transport.setHold(true)
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const storageApi = scriptedStorage(storage).storage
    const first = loaded.adoption.adoptShippedGroupChats(storageApi)
    const overlap = loaded.adoption.adoptShippedGroupChats(storageApi)

    await waitFor(() => expect(transport.imports).toHaveLength(1))
    runtime.connectionId = 'remote-b'
    runtime.activation += 1
    transport.release()
    await Promise.all([first, overlap])

    expect(transport.imports).toHaveLength(1)
    expect((storage.get('group-chats') as Record<string, any>).Release.shippedAdoption.state).toBe('prepared')
    expect(loaded.registry.$canonicalGroupBindings.get()).toEqual({})

    runtime.connectionId = 'owner-a'
    transport.setHold(false)
    loaded.adoption.stopShippedGroupAdoption()
    loaded.chat.$groupChats.set(loaded.chat.hydrateGroupChatRooms(storage.get('group-chats')))
    await loaded.adoption.adoptShippedGroupChats(storageApi)
    expect(transport.imports).toHaveLength(2)
    expect(transport.imports[1]).toEqual(transport.imports[0])
    expect((storage.get('group-chats') as Record<string, any>).Release.shippedAdoption.state).toBe('adopted')
  })

  it('rejects a same-connection route ABA after import response and recovers only through a fresh lease', async () => {
    const transport = backend()
    transport.setHold(true)
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const storageApi = scriptedStorage(storage).storage
    const importing = loaded.adoption.adoptShippedGroupChats(storageApi)

    await waitFor(() => expect(transport.imports).toHaveLength(1))
    runtime.routeGeneration += 1
    runtime.routeGeneration += 1
    transport.release()
    await importing

    expect((storage.get('group-chats') as Record<string, any>).Release.shippedAdoption).toMatchObject({
      state: 'prepared',
      issue: { kind: 'offline' }
    })
    expect(loaded.registry.$canonicalGroupBindings.get()).toEqual({})

    transport.setHold(false)
    await loaded.adoption.adoptShippedGroupChats(storageApi)
    expect(transport.imports).toHaveLength(2)
    expect(transport.imports[1]).toEqual(transport.imports[0])
    expect((storage.get('group-chats') as Record<string, any>).Release.shippedAdoption.state).toBe('adopted')
    expect(loaded.registry.$canonicalGroupBindings.get().Release?.isCurrent?.()).toBe(true)
  })

  it('rechecks route ownership after durable acknowledgement before binding', async () => {
    const transport = backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const storageApi = scriptedStorage(storage).storage
    const originalSet = storageApi.set.bind(storageApi)
    let releaseAck!: () => void

    const ackHeld = new Promise<void>(resolve => {
      releaseAck = resolve
    })

    let ackStarted = false

    storageApi.set = async (key, value) => {
      await originalSet(key, value)
      const rooms = value as Record<string, any>

      if (key === 'group-chats' && rooms.Release?.shippedAdoption?.state === 'adopted') {
        ackStarted = true
        await ackHeld
      }
    }

    const adopting = loaded.adoption.adoptShippedGroupChats(storageApi)
    await waitFor(() => expect(ackStarted).toBe(true))
    expect(transport.imports).toHaveLength(1)
    runtime.routeGeneration += 1
    releaseAck()
    await adopting

    expect((storage.get('group-chats') as Record<string, any>).Release.shippedAdoption.state).toBe('adopted')
    expect(loaded.registry.$canonicalGroupBindings.get()).toEqual({})

    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    expect(transport.imports).toHaveLength(1)
    expect(loaded.registry.$canonicalGroupBindings.get().Release?.isCurrent?.()).toBe(true)
  })

  it('fences a same-name local room replacement while the old import is in flight', async () => {
    const transport = backend()
    transport.setHold(true)
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const storageApi = scriptedStorage(storage).storage
    const importing = loaded.adoption.adoptShippedGroupChats(storageApi)

    await waitFor(() => expect(transport.imports).toHaveLength(1))

    const replacement = {
      ...loaded.chat.$groupChats.get().Release,
      desktopAuthorityHash: 'f'.repeat(64),
      desktopAuthorityToken: 'authority:replacement',
      log: [{ at: 1_900_000_000_000, from: { kind: 'user' as const, name: 'You' }, text: 'Replacement room' }],
      shippedAdoption: undefined
    }

    const rooms = { ...loaded.chat.$groupChats.get(), Release: replacement }
    loaded.chat.$groupChats.set(rooms)
    await loaded.chat.persistGroupChatRoomsRequired(rooms, storageApi)
    transport.release()
    await importing

    expect(loaded.registry.$canonicalGroupBindings.get()).toEqual({})
    expect((storage.get('group-chats') as Record<string, any>).Release.log[0].text).toBe('Replacement room')
    expect((storage.get('group-chats') as Record<string, any>).Release.shippedAdoption).toBeUndefined()
  })

  it.each([
    [
      'missing import capability',
      { authority_gateway_id: 'install:owner-a', driver: true, methods: ['groups.state'] },
      'update-required'
    ],
    ['expired authentication', Object.assign(new Error('Unauthorized'), { status: 401 }), 'auth'],
    ['offline owner', new Error('Socket closed'), 'offline']
  ] as const)('reports %s accurately without starting import or dispatch', async (_label, outcome, issueKind) => {
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    const calls: string[] = []

    runtime.handler = async (_route, method) => {
      calls.push(method)

      if (method !== 'groups.capabilities') {
        throw new Error(`Unexpected method ${method}`)
      }

      if (outcome instanceof Error) {
        throw outcome
      }

      return outcome
    }

    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    const checkpoint = (storage.get('group-chats') as Record<string, any>).Release.shippedAdoption

    expect(checkpoint).toMatchObject({ state: 'waiting', issue: { kind: issueKind } })
    expect(calls).toEqual(['groups.capabilities'])
  })

  it('never crosses an owner installation and does not guess among ambiguous historical routes', async () => {
    const transport = backend({ failAfterFirstCommit: true })
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    let loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    expect(transport.imports).toHaveLength(1)

    loaded.adoption.stopShippedGroupAdoption()
    loaded.chat.$groupChats.set(loaded.chat.hydrateGroupChatRooms(storage.get('group-chats')))
    runtime.authority = 'install:replacement'
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)

    expect(transport.imports).toHaveLength(1)
    const replaced = (storage.get('group-chats') as Record<string, any>).Release.shippedAdoption
    expect(replaced.route).toEqual({
      connectionId: 'owner-a',
      profile: 'default',
      authorityGatewayId: 'install:owner-a'
    })
    expect(replaced.issue.kind).toBe('owner-replaced')
    expect(transport.calls.filter(call => call.route?.connectionId === 'remote-b')).toEqual([])

    const ambiguousStorage = new Map<string, unknown>([
      [
        'group-chats',
        {
          Ambiguous: {
            log: [{ at: 1, from: { kind: 'user', name: 'You' }, text: 'Keep me' }],
            members: [{ name: 'alpha' }, { name: 'beta' }],
            watermarks: {}
          }
        }
      ]
    ])

    runtime.routes = [
      { connectionId: 'remote-one', mode: 'remote', profile: 'default', targetProfile: 'default' },
      { connectionId: 'remote-two', mode: 'remote', profile: 'default', targetProfile: 'default' }
    ]
    const ambiguousTransport = backend()
    loaded = await coldHydrate(ambiguousStorage)
    const before = ambiguousTransport.calls.length
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(ambiguousStorage).storage)
    const ambiguous = (ambiguousStorage.get('group-chats') as Record<string, any>).Ambiguous.shippedAdoption

    expect(ambiguousTransport.calls).toHaveLength(before)
    expect(ambiguous).toMatchObject({ state: 'waiting', issue: { kind: 'owner-ambiguous' } })
    expect(ambiguous.sourceId).toMatch(/^hermes\.plugin\.hermes-bots\.group-chats:/)
    expect(ambiguous.requestHash).toMatch(/^[0-9a-f]{64}$/)
    expect((ambiguousStorage.get('group-chats') as Record<string, any>).Ambiguous.log).toHaveLength(1)

    const view = await import('./group-chat-view')
    render(
      <view.GroupChatWorkspace group="Ambiguous" members={loaded.chat.$groupChats.get().Ambiguous.members || []} />
    )
    expect(screen.queryByRole('textbox')).toBeNull()
    const ownerButtons = await screen.findAllByRole('button', { name: /^Use Device / })
    fireEvent.click(ownerButtons[1])
    await waitFor(() =>
      expect((ambiguousStorage.get('group-chats') as Record<string, any>).Ambiguous.shippedAdoption.state).toBe(
        'adopted'
      )
    )
    expect((ambiguousStorage.get('group-chats') as Record<string, any>).Ambiguous.shippedAdoption).toMatchObject({
      ownerSelection: 'explicit',
      route: { connectionId: 'remote-two', profile: 'default' }
    })
    expect(ambiguousTransport.imports.at(-1)?.members.every(member => member.remote_source === false)).toBe(true)
    expect(loaded.registry.$canonicalGroupBindings.get().Ambiguous?.isCurrent?.()).toBe(true)
  })

  it('preserves another adopted checkpoint through disband and cold reload', async () => {
    backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    const adopted = structuredClone((storage.get('group-chats') as Record<string, any>).Release.shippedAdoption)

    const rooms = {
      ...loaded.chat.$groupChats.get(),
      Other: { log: [], members: [], sessions: {}, watermarks: {}, roomId: 'other-room' }
    }

    loaded.chat.$groupChats.set(rooms)
    await loaded.chat.persistGroupChatRoomsRequired(rooms, scriptedStorage(storage).storage)
    const view = await import('./group-chat-view')
    await view.disbandGroupChat('Other', [])

    loaded.adoption.stopShippedGroupAdoption()
    loaded.chat.$groupChats.set(loaded.chat.hydrateGroupChatRooms(storage.get('group-chats')))
    expect(loaded.chat.$groupChats.get().Release.shippedAdoption).toEqual(adopted)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    expect(loaded.registry.$canonicalGroupBindings.get().Release?.isCurrent?.()).toBe(true)
  })

  it('revokes adopted bindings on rename, same-name reuse, and lifecycle disposal', async () => {
    backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    const original = loaded.registry.$canonicalGroupBindings.get().Release
    expect(original?.isCurrent?.()).toBe(true)

    const view = await import('./group-chat-view')
    await view.renameGroupChat('Release', 'Renamed', [], { hostedAlreadyRenamed: true })
    expect(original?.isCurrent?.()).toBe(false)
    expect(loaded.registry.$canonicalGroupBindings.get().Release).toBeUndefined()

    expect(loaded.registry.$canonicalGroupBindings.get().Renamed?.isCurrent?.()).toBe(true)

    loaded.chat.$groupChats.set({
      ...loaded.chat.$groupChats.get(),
      Release: {
        log: [{ at: 2, from: { kind: 'user', name: 'You' }, text: 'New incarnation' }],
        members: [],
        watermarks: {}
      }
    })
    render(<view.GroupChatWorkspace group="Release" members={[]} />)
    expect(screen.getByText('New incarnation')).toBeTruthy()

    loaded.adoption.stopShippedGroupAdoption()
    expect(loaded.registry.$canonicalGroupBindings.get()).toEqual({})
  })

  it.each([false, true])('sends through the live adopted binding and retains a late ACK after route replacement: %s', async expire => {
    const transport = backend()
    const handler = runtime.handler
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })

    runtime.handler = async (route, method, params) => {
      const result = await handler(route, method, params)

      if (expire && method === 'groups.send') { await held }

      return result
    }

    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    const binding = loaded.registry.$canonicalGroupBindings.get().Release!
    render(<loaded.workspace.CanonicalGroupWorkspace binding={binding} />)
    fireEvent.change(await screen.findByRole('textbox'), { target: { value: 'new user work' } })
    const button = screen.getByRole('button', { name: 'Send' })
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(button)
    await waitFor(() => expect(transport.calls.filter(call => call.method === 'groups.send')).toHaveLength(1))
    expect(transport.calls.find(call => call.method === 'groups.send')?.params.payload).toMatchObject({ text: 'new user work' })
    const journal = await import('./canonical-group-send')

    if (expire) {
      runtime.routeGeneration += 1
      release()
      await waitFor(() => expect(binding.isCurrent?.()).toBe(false))
      await expect(journal.readCanonicalGroupSend(binding)).rejects.toThrow(/owner/i)
      const pending = Object.values(JSON.parse(localStorage.getItem('hermes.desktop.canonicalGroupSends.v1')!))[0] as any
      expect(pending.params.payload.text).toBe('new user work')
      expect(pending.binding).not.toHaveProperty('routeOwner')
    } else {
      await waitFor(async () => expect(await journal.readCanonicalGroupSend(binding)).toBeUndefined())
    }
  })

  it('keeps an adopted room read-only while its binding is absent, then restores on its background route open', async () => {
    const transport = backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    const old = loaded.registry.$canonicalGroupBindings.get().Release!
    loaded.chat.$groupChats.set({ ...loaded.chat.$groupChats.get(), Pending: (await releasedRecord()).Release as ReturnType<typeof loaded.chat.$groupChats.get>[string] })
    runtime.routeGeneration += 1

    for (const listener of runtime.routeListeners) { listener({ connectionId: 'owner-a', profile: 'default', state: 'closed' }) }
    expect(old.isCurrent?.()).toBe(false)
    expect(loaded.registry.$canonicalGroupBindings.get().Release).toBeUndefined()
    const view = await import('./group-chat-view')
    render(<view.GroupChatWorkspace group="Release" members={[]} />)
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(screen.getByText(CANONICAL_GROUP_LOCALES.en.upgradeChecking)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()

    for (const listener of runtime.routeListeners) { listener({ connectionId: 'owner-a', profile: 'default', state: 'open' }) }
    await waitFor(() => expect(loaded.registry.$canonicalGroupBindings.get().Release?.isCurrent?.()).toBe(true))
    expect(transport.imports).toHaveLength(1)
    expect(loaded.chat.$groupChats.get().Pending.shippedAdoption).toBeUndefined()
  })

  it.each(['rename', 'disband'])('restores the exact adopted binding after a failed %s', async operation => {
    backend()
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    const original = loaded.registry.$canonicalGroupBindings.get().Release!
    const hosted = await import('./hosted-room-runtime')
    const view = await import('./group-chat-view')

    if (operation === 'rename') {
      vi.spyOn(hosted, 'renameHostedGroupChat').mockRejectedValueOnce(new Error('offline'))
      expect(await view.renameGroupChat('Release', 'Renamed', [])).toBeNull()
    } else {
      vi.spyOn(hosted, 'disbandHostedGroupChat').mockResolvedValueOnce(false)
      await expect(view.disbandGroupChat('Release', [])).rejects.toThrow()
    }

    expect(original.isCurrent?.()).toBe(false)
    const restored = loaded.registry.$canonicalGroupBindings.get().Release!
    expect(restored.isCurrent?.()).toBe(true)
    expect(restored.roomId).toBe(original.roomId)
    expect(restored.bindingGeneration).not.toBe(original.bindingGeneration)
  })

  it('renders durable retirement as removal pending and retries retire for the exact server member', async () => {
    const transport = backend()
    transport.setRetiring('Remote Builder')
    const storage = new Map<string, unknown>([['group-chats', await releasedRecord()]])
    const loaded = await coldHydrate(storage)
    await loaded.adoption.adoptShippedGroupChats(scriptedStorage(storage).storage)
    const binding = loaded.registry.$canonicalGroupBindings.get().Release as CanonicalGroupBinding
    render(<loaded.workspace.CanonicalGroupWorkspace binding={binding} />)

    const pending = await screen.findByText('Removal pending. This Bot cannot receive new work.')
    const row = pending.closest('div')!
    expect(within(row).queryByRole('button', { name: 'Re-add' })).toBeNull()
    expect(within(row).queryByRole('button', { name: 'Check again' })).toBeNull()
    fireEvent.click(within(row).getByRole('button', { name: 'Retry removal' }))
    await waitFor(() =>
      expect(
        transport.calls.some(
          call =>
            call.method === 'groups.member.resolve' &&
            call.params.action === 'retire' &&
            String(call.params.member_id).startsWith('server:member:2:')
        )
      ).toBe(true)
    )
  })
})
