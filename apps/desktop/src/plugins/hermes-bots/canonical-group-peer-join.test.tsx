import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import type { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as DataModule from './data'

const { request, union, notify, language } = vi.hoisted(() => ({
  language: { value: 'en' },
  request: vi.fn(),
  notify: vi.fn(),
  union: { profiles: [] as unknown[] }
}))
vi.mock('@hermes/plugin-sdk', async () => {
  const { atom: createAtom } = await import('nanostores')
  const { useStore } = await import('@nanostores/react')
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { captureGroupRequests } = await import('./canonical-group-contract-test-utils')
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')

  return {
    ...(await pluginSdkMock(gateway.host)),
    atom: createAtom,
    useValue: useStore,
    ConfirmDialog,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => translateBotsIn(language.value as keyof typeof CANONICAL_GROUP_LOCALES),
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    host: {
      ...gateway.host,
      state: { ...(gateway.host.state as object), gateway: createAtom('open') },
      notify,
      requestProfile: captureGroupRequests(request).request
    }
  }
})
vi.mock('./data', async importOriginal => ({
  ...(await importOriginal<typeof DataModule>()),
  cachedUnionRoster: () => union
}))

import {
  answerGroupRequests,
  DISBAND_FAILURES,
  PEER_HOME as HOME,
  PEER_MINI as MINI,
  PEER_REGISTERED,
  peerGroupCapabilities,
  peerInvitation,
  peerRosterRow,
  strictGroupCreation
} from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $peerJoinErrors, peerJoinKey, peerRenewalSettled, renewPeers } from './canonical-group-peers'
import { $canonicalGroupBindings, registerCanonicalGroup } from './canonical-group-registry'
import { readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import { $groupChats, updateGroupChat } from './group-chat'
import { scriptedStorage } from './group-test-utils'
import { translateBotsIn } from './i18n-test-helper'
import { setPluginCtx } from './shared'

const words = CANONICAL_GROUP_LOCALES.en
const say = (text: string) => text.replace('{name}', 'Hermes · Mac Mini')
const binding = { connectionId: 'local', profile: 'default', roomId: 'linked' }
const pinned = {
  kind: 'peer',
  peer_id: 'install:3f2a0000000000000000000000000000',
  installation_id: 'install:3f2a0000000000000000000000000000',
  profile: 'default',
  capability_digest: MINI.room_link.catalog.catalog_digest
}
const mini = peerRosterRow('mini', 'Mac Mini')

const room = {
  room_id: 'linked',
  name: 'Linked',
  authority_gateway_id: 'install:a18b0000000000000000000000000000',
  authority_epoch: 1,
  latest_seq: 0,
  members: [
    { member_id: 'default', profile: 'default', handle: 'hermes', target: { kind: 'local', profile: 'default' } },
    { member_id: 'coder', profile: 'coder', handle: 'coder', target: { kind: 'local', profile: 'coder' } },
    {
      member_id: 'hermes-mac-mini',
      profile: 'default',
      handle: 'hermes-mac-mini',
      display_name: 'Hermes · Mac Mini',
      target: pinned
    }
  ]
}

type Handler = (method: string, params: Record<string, any>) => unknown

/** The home (`local`) and the Mac Mini's gateway (`mini`), each answering as `HOME` / `MINI` unless its
 *  handler does; `offline` never answers. The home reports `routes`, and the room as ended once `ended`. */
function gateways({
  home,
  peer,
  elsewhere = {},
  routes = []
}: {
  home?: Handler
  peer?: Handler
  elsewhere?: Record<string, Handler>
  routes?: unknown[]
}) {
  const state = { ended: false, routes }
  const { create } = strictGroupCreation()

  answerGroupRequests(request, {
    ...elsewhere,
    offline: () => new Promise(() => undefined),
    mini: (method, params) =>
      peer?.(method, params) ??
      (
        {
          'groups.capabilities': MINI,
          'groups.peer.invite': peerInvitation(),
          'groups.peer.revoke': { revoked: true }
        } as Record<string, unknown>
      )[method],
    local: (method, params) =>
      home?.(method, params) ??
      (method === 'groups.create'
        ? create(params)
        : (
            {
              'groups.capabilities': HOME,
              'groups.log': { events: [] },
              'groups.peer.register': PEER_REGISTERED,
              'groups.state': {
                room: state.ended ? { ...room, disbanded_at: 2 } : room,
                driver_status: { running: true, peer_routes: state.routes }
              }
            } as Record<string, unknown>
          )[method])
  })

  return state
}

const refusal = (reason: string) => Object.assign(new Error(reason), { code: 4001, data: { reason } })
const status = () => screen.getByText(say(words.peerNotJoined)).closest('[role="status"]') as HTMLElement
const steps = () =>
  request.mock.calls
    .map(([route, method]) => [route?.connectionId, method])
    .filter(([, method]) => method.startsWith('groups.peer'))
const SEATED = [
  ['mini', 'groups.peer.invite'],
  ['local', 'groups.peer.register']
]

async function retry() {
  await screen.findByText(say(words.peerNotJoined))
  await act(async () => {
    fireEvent.click(within(status()).getByRole('button', { name: 'Retry' }))
  })
}

beforeEach(() => {
  setPluginCtx(scriptedStorage(new Map()))
  language.value = 'en'
  ;(host.state.gateway as ReturnType<typeof atom<string>>).set('open')
  union.profiles = [mini]
  $groupChats.set({})
  $peerJoinErrors.set({})
  $canonicalGroupBindings.set({})
})
afterEach(() => {
  setPluginCtx(null)
  cleanup()
  request.mockReset()
  notify.mockReset()
  vi.restoreAllMocks()
})

it.each(['missing', 'unavailable'])(
  'retries a %s peer through its matching connection without waiting on another offline computer',
  async status => {
    union.profiles = [peerRosterRow('offline', 'Old laptop'), mini]

    const state = gateways({
      routes: status === 'missing' ? [] : [{ room_id: 'linked', member_id: 'hermes-mac-mini', status }],
      home: (method, params) =>
        method === 'groups.peer.register'
          ? ((state.routes = [{ room_id: 'linked', member_id: params.member_id, status: 'ready' }]), PEER_REGISTERED)
          : undefined
    })

    render(<CanonicalGroupWorkspace binding={binding} />)
    const text = say(status === 'missing' ? words.peerNotJoined : words.peerAway)
    const row = (await screen.findByText(text)).closest('[role="status"]') as HTMLElement
    const badge = status === 'missing' ? words.peerNotJoinedBadge : words.peerOfflineBadge
    expect(within(screen.getByRole('list', { name: words.members })).getByText(badge)).toBeTruthy()
    expect(within(row).getByRole('button', { name: say(words.peerStartWithout) })).toBeTruthy()
    await act(async () => {
      fireEvent.click(within(row).getByRole('button', { name: 'Retry' }))
    })
    await waitFor(() => expect(screen.queryByText(text)).toBeNull())
    expect(screen.queryByText(badge)).toBeNull()
    expect(steps()).toEqual(SEATED)
    expect(request.mock.calls.find(call => call[1] === 'groups.peer.invite')![2]).toEqual({
      profile: 'default',
      room_id: 'linked',
      member_id: 'hermes-mac-mini',
      home_install_id: 'install:a18b0000000000000000000000000000',
      authority_gateway_id: 'install:a18b0000000000000000000000000000',
      authority_epoch: 1,
      ttl_seconds: 86400,
      status_ttl_seconds: 2592000
    })
  }
)

it.each([
  // An ordinary failure never revokes: revocation is scope-wide, and would end a live room's working grant.
  [
    'the home cannot reach it',
    { home: (method: string) => (method === 'groups.peer.register' ? refusal('peer_unreachable') : undefined) },
    words.peerJoinUnreachable,
    SEATED
  ],
  [
    'its gateway lends another catalog now',
    {
      peer: (method: string) =>
        method === 'groups.capabilities'
          ? {
              ...MINI,
              room_link: { ...MINI.room_link, catalog: { ...MINI.room_link.catalog, catalog_digest: 'd'.repeat(64) } }
            }
          : undefined
    },
    words.peerJoinChanged,
    []
  ],
  [
    'no connection here reaches it',
    {
      peer: (method: string) => (method === 'groups.capabilities' ? peerGroupCapabilities('install:other') : undefined)
    },
    words.peerJoinConnect,
    []
  ]
])('says why a Retry did not seat the Bot when %s', async (_case, handlers, why, asked) => {
  gateways(handlers)
  render(<CanonicalGroupWorkspace binding={binding} />)
  await retry()
  await waitFor(() => expect(within(status()).getByText(say(why))).toBeTruthy())
  expect(steps()).toEqual(asked)
})

it("says when a joined Bot's access ended, and offers Retry", async () => {
  gateways({ routes: [{ room_id: 'linked', member_id: 'hermes-mac-mini', status: 'needs_reauthorization' }] })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const row = (await screen.findByText(say(words.peerRejoin))).closest('[role="status"]') as HTMLElement
  expect(within(row).getByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(screen.getByText(words.peerRejoinBadge)).toBeTruthy()
  expect(screen.queryByText(words.peerNotJoinedBadge)).toBeNull()
})

const later = (answer: unknown) => new Promise(resolve => setTimeout(() => resolve(answer), 0))
const oneDay = { ...MINI, features: MINI.features.filter(feature => feature !== 'peer_renewal_horizon') }
const nonpersistent = {
  ...MINI,
  room_link: {
    ...MINI.room_link,
    catalog: { ...MINI.room_link.catalog, persistent_process: false, catalog_digest: 'e'.repeat(64) }
  }
}

it.each([
  ['grants the longer renewal horizon', oneDay, MINI, 2592000],
  ['can host it, when another connection to that computer cannot', nonpersistent, oneDay, undefined]
])('seats the Bot through the connection that %s', async (_case, first, chosen, horizon) => {
  union.profiles = [peerRosterRow('first', 'Mac Mini (first)'), mini]
  gateways({
    elsewhere: { first: method => (method === 'groups.capabilities' ? first : undefined) },
    peer: method => (method === 'groups.capabilities' ? later(chosen) : undefined)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await retry()
  await waitFor(() => expect(steps()).toEqual(SEATED))
  expect(request.mock.calls.find(call => call[1] === 'groups.peer.invite')![2].status_ttl_seconds).toBe(horizon)
})

it('stops waiting for a silent connection once another can seat the Bot', async () => {
  union.profiles = [peerRosterRow('first', 'Mac Mini (first)'), mini]
  gateways({
    elsewhere: { first: method => (method === 'groups.capabilities' ? oneDay : peerInvitation()) },
    peer: method => (method === 'groups.capabilities' ? new Promise(() => undefined) : undefined)
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await retry()
  await waitFor(
    () =>
      expect(steps()).toEqual([
        ['first', 'groups.peer.invite'],
        ['local', 'groups.peer.register']
      ]),
    { timeout: 4_000 }
  )
})

it.each(['capabilities', 'invite'] as const)('does not leave a grant when the room ends during %s', async step => {
  const state = gateways({
    home: method => (method === 'groups.peer.register' ? refusal('invalid_params') : undefined),
    peer: method =>
      method === `groups.peer.${step}` || method === `groups.${step}` ? ((state.ended = true), undefined) : undefined
  })

  render(<CanonicalGroupWorkspace binding={binding} />)
  await retry()
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.state').length).toBeGreaterThan(1))
  await waitFor(() => expect(steps()).toEqual(step === 'invite' ? [...SEATED, ['mini', 'groups.peer.revoke']] : []))

  if (step === 'invite') {
    expect(request.mock.calls.find(call => call[1] === 'groups.peer.revoke')![2].grant).toBe('grant')
  }
})

it.each(['confirmed', 'offline', 'unconfirmed', 'stop pending'])(
  'starts the group again and keeps the original until Disband is confirmed (%s)',
  async outcome => {
    const onReplaced = vi.fn()
    const failure = DISBAND_FAILURES.find(({ name }) => name === outcome)
    const confirmed = outcome === 'confirmed' || outcome === 'offline'

    if (outcome === 'offline') {
      union.profiles = []
    }
    gateways({
      routes: outcome === 'offline' ? [{ room_id: 'linked', member_id: 'hermes-mac-mini', status: 'unavailable' }] : [],
      home: method =>
        method === 'groups.disband'
          ? confirmed
            ? { tombstone: { room_id: 'linked', disbanded_at: 1 } }
            : (failure?.error ?? {})
          : undefined
    })
    updateGroupChat(
      'Linked',
      record => ({ ...record, roomId: binding.roomId, engine: { kind: 'gateway', ...binding } }),
      { sync: false }
    )
    const original = structuredClone($groupChats.get().Linked)
    registerCanonicalGroup({ connectionId: 'local', profile: 'default' }, room)
    render(<CanonicalGroupWorkspace binding={binding} onReplaced={onReplaced} />)
    const row = (await screen.findByText(say(outcome === 'offline' ? words.peerAway : words.peerNotJoined))).closest(
      '[role="status"]'
    ) as HTMLElement

    if (outcome === 'offline') {
      expect(within(row).queryByRole('button', { name: 'Retry' })).toBeNull()
    }
    // The control says what it does: a new group without that Bot, earlier messages not copied.
    fireEvent.click(within(row).getByRole('button', { name: 'Start new group chat without Hermes · Mac Mini' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText(words.peerStartWithoutConfirm)).toBeTruthy()
    await act(async () => {
      fireEvent.click(within(dialog).getByRole('button', { name: words.startGatewayGroup }))
    })
    await waitFor(() => expect(onReplaced).toHaveBeenCalledOnce())

    const create = request.mock.calls.find(call => call[1] === 'groups.create')![2]
    expect(create).toEqual({
      profile: 'default',
      room_id: create.room_id,
      name: 'Linked 2',
      members: room.members.slice(0, 2)
    })
    expect(
      request.mock.calls
        .map(call => call[1])
        .filter(method => method === 'groups.create' || method === 'groups.disband')
    ).toEqual(['groups.create', 'groups.disband'])
    expect(
      Object.values($canonicalGroupBindings.get())
        .map(entry => entry.roomId)
        .sort()
    ).toEqual((confirmed ? [create.room_id] : ['linked', create.room_id]).sort())

    if (confirmed) {
      expect($groupChats.get().Linked).toBeUndefined()
    } else {
      expect($groupChats.get().Linked).toEqual(original)
      expect(notify).toHaveBeenCalledExactlyOnceWith({
        kind: 'error',
        message: words.peerStartWithoutKept.replace('{reason}', words[failure?.detail ?? 'disbandUnconfirmed'])
      })
    }

    expect($groupChats.get()[create.name]?.engine).toEqual({ kind: 'gateway', ...binding, roomId: create.room_id })
  }
)

it('shows the badge but no setup controls on a gateway that cannot register Bots of other gateways', async () => {
  gateways({
    home: method =>
      method === 'groups.capabilities'
        ? { ...HOME, methods: HOME.methods.filter(name => !name.startsWith('groups.peer')) }
        : undefined
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await screen.findByText(words.peerNotJoinedBadge)
  await act(async () => {})
  expect(screen.queryByText(words.peerNotJoined.replace('{name}', 'Hermes · Mac Mini'))).toBeNull()
})

const calls = (method: string) => request.mock.calls.filter(call => call[1] === method)

async function startWithout(name: string) {
  fireEvent.click(await screen.findByRole('button', { name: words.peerStartWithout.replace('{name}', name) }))
  await act(async () => {
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: words.startGatewayGroup }))
  })
}

it.each(['desktop', 'gateway'])(
  'retries a lost replacement create without taking over a colliding %s record',
  async engine => {
    const { rooms, create } = strictGroupCreation()
    let lost = true
    gateways({
      home: (method, params) => {
        if (method === 'groups.disband') {
          return refusal('room_stop_pending')
        }

        if (method !== 'groups.create') {
          return undefined
        }
        const answer = create(params)

        if (lost) {
          lost = false
          throw new Error('reply lost')
        }

        return answer
      }
    })
    updateGroupChat(room.name, record => ({ ...record, engine: { kind: 'gateway', ...binding } }), { sync: false })
    const onReplaced = vi.fn()
    render(<CanonicalGroupWorkspace binding={binding} onReplaced={onReplaced} />)

    await startWithout('Hermes · Mac Mini')
    expect(await screen.findByText('reply lost')).toBeTruthy()
    const payload = calls('groups.create')[0][2]
    updateGroupChat(
      payload.name,
      record => ({
        ...record,
        roomId: 'unrelated',
        log: [{ at: 1, id: 'history', from: { kind: 'user', name: 'You' }, text: 'Keep this history' }],
        engine: engine === 'desktop' ? 'desktop' : { kind: 'gateway', ...binding, roomId: 'unrelated' }
      }),
      { sync: false }
    )
    const other = structuredClone($groupChats.get()[payload.name])
    await act(async () => {
      fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: words.startGatewayGroup }))
    })
    expect(onReplaced).toHaveBeenCalledOnce()
    expect(calls('groups.create')[1][2]).toEqual(payload)
    expect(rooms.size).toBe(1)
    expect($groupChats.get()[payload.name]).toEqual(other)
    const created = Object.values($groupChats.get()).find(record => record.roomId === payload.room_id)
    expect(created?.engine).toEqual({ kind: 'gateway', ...binding, roomId: payload.room_id })
    expect(created?.log).toEqual([])
    await startWithout('Hermes · Mac Mini')
    expect(rooms.size).toBe(2)
    expect(calls('groups.create')[2][2].room_id).not.toBe(payload.room_id)
    expect($groupChats.get()[payload.name]).toEqual(other)
  }
)

it.each(['missing', 'refused'])(
  'keeps A+B+C when removing B cannot seat C (%s), with Retry in the new group',
  async failure => {
    const labels = words

    const original = {
      ...room,
      room_id: `replace-${failure}`,
      members: [
        room.members[0],
        { ...room.members[1], display_name: 'B', target: { ...pinned, installation_id: 'install:b' } },
        { ...room.members[2], display_name: 'C' }
      ]
    }

    const own = { ...binding, roomId: original.room_id }
    const { rooms: created, create } = strictGroupCreation()
    let recovered = false
    union.profiles = failure === 'missing' ? [] : [mini]
    gateways({
      home: (method, params) => {
        if (method === 'groups.state') {
          return {
            room: created.get(params.room_id) ?? original,
            driver_status: {
              running: true,
              peer_routes: [{ member_id: 'hermes-mac-mini', status: recovered ? 'ready' : 'unavailable' }]
            }
          }
        }

        if (method === 'groups.create') {
          return create(params)
        }

        if (method === 'groups.peer.register' && !recovered) {
          return refusal('peer_unreachable')
        }

        if (method === 'groups.disband') {
          return { tombstone: { room_id: original.room_id, disbanded_at: 1 } }
        }
      }
    })
    updateGroupChat(original.name, record => ({ ...record, roomId: own.roomId, engine: { kind: 'gateway', ...own } }), {
      sync: false
    })
    const before = structuredClone($groupChats.get()[original.name])
    const originalKey = registerCanonicalGroup(own, original)
    const onReplaced = vi.fn()
    const view = render(<CanonicalGroupWorkspace binding={own} onReplaced={onReplaced} />)

    await startWithout('B')
    await waitFor(() => expect(onReplaced).toHaveBeenCalledOnce())
    expect(calls('groups.disband')).toHaveLength(0)
    expect($groupChats.get()[original.name]).toEqual(before)
    expect($canonicalGroupBindings.get()[originalKey]).toMatchObject(own)
    const reason = (failure === 'missing' ? labels.peerJoinConnect : labels.peerJoinUnreachable).replace('{name}', 'C')
    expect(notify).toHaveBeenCalledWith({
      kind: 'error',
      message: labels.peerStartWithoutKept.replace('{reason}', reason + ' ' + labels.peerStartWithoutRetry)
    })
    expect(created.size).toBe(1)
    const next = [...created.values()][0]
    expect(next.members.map(member => member.member_id)).toEqual(['default', 'hermes-mac-mini'])
    expect($groupChats.get()[next.name].engine).toEqual({ kind: 'gateway', ...binding, roomId: next.room_id })
    union.profiles = [mini]
    view.unmount()
    const replacementView = render(<CanonicalGroupWorkspace binding={{ ...own, roomId: next.room_id }} />)
    const retryBot = await screen.findByRole('button', { name: 'Retry' })
    recovered = true
    await act(async () => {
      fireEvent.click(retryBot)
    })
    await waitFor(() => expect(calls('groups.peer.register').at(-1)?.[2].room_id).toBe(next.room_id))
    expect(calls('groups.disband')).toHaveLength(0)
    replacementView.unmount()
    render(<CanonicalGroupWorkspace binding={own} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Disband' }))
    await act(async () => {
      fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Disband' }))
    })
    expect(created.size).toBe(1)
    expect(calls('groups.disband')).toHaveLength(1)
    expect($canonicalGroupBindings.get()[originalKey]).toBeUndefined()
  }
)

it.each(['ready', 'unreadable', 'another room'])(
  'checks the new group after a lost registration reply (%s)',
  async outcome => {
    const original = {
      ...room,
      members: [
        room.members[0],
        { ...room.members[1], display_name: 'B', target: { ...pinned, installation_id: 'install:b' } },
        { ...room.members[2], display_name: 'C' }
      ]
    }

    const { rooms, create } = strictGroupCreation()
    let registered = false
    gateways({
      home: (method, params) => {
        if (method === 'groups.create') {
          return create(params)
        }

        if (method === 'groups.state') {
          if (registered && outcome === 'unreadable') {
            throw new Error('read lost')
          }

          return {
            room: registered && outcome === 'another room' ? original : (rooms.get(params.room_id) ?? original),
            driver_status: {
              running: true,
              peer_routes: registered ? [{ member_id: 'hermes-mac-mini', status: 'ready' }] : []
            }
          }
        }

        if (method === 'groups.peer.register') {
          registered = true
          throw new Error('reply lost')
        }

        if (method === 'groups.disband') {
          return { tombstone: { room_id: original.room_id, disbanded_at: 1 } }
        }
      }
    })
    const onReplaced = vi.fn()
    render(<CanonicalGroupWorkspace binding={binding} onReplaced={onReplaced} />)
    await startWithout('B')
    expect(onReplaced).toHaveBeenCalledOnce()
    const next = [...rooms.values()][0]
    expect(calls('groups.state').filter(call => call[2].room_id === next.room_id)).toHaveLength(2)
    expect(calls('groups.disband')).toHaveLength(outcome === 'ready' ? 1 : 0)

    if (outcome === 'ready') {
      expect(notify).not.toHaveBeenCalled()
    } else {
      expect(notify).toHaveBeenCalledWith({
        kind: 'error',
        message: words.peerStartWithoutKept.replace(
          '{reason}',
          'Could not confirm whether C joined. Check the group chat when its computer is reachable.'
        )
      })
    }
  }
)

it.each(['acknowledgement', 'refusal'])(
  'replays an uncertain replacement through reconnect, then clears it on %s',
  async outcome => {
    const { rooms, create } = strictGroupCreation()
    let lost = true
    let refuse = outcome === 'refusal'
    gateways({
      home: (method, params) => {
        if (method === 'groups.disband') {
          return refusal('room_stop_pending')
        }

        if (method !== 'groups.create') {
          return undefined
        }

        if (!lost && refuse) {
          refuse = false

          return refusal('invalid_params')
        }

        const answer = create(params)

        if (lost) {
          lost = false
          throw new Error('reply lost')
        }

        return answer
      }
    })
    const onReplaced = vi.fn()
    render(<CanonicalGroupWorkspace binding={binding} onReplaced={onReplaced} />)
    await startWithout('Hermes · Mac Mini')
    expect(await screen.findByText('reply lost')).toBeTruthy()
    const payload = calls('groups.create')[0][2]
    await act(async () => {
      ;(host.state.gateway as ReturnType<typeof atom<string>>).set('closed')
    })
    expect(screen.queryByRole('dialog')).toBeNull()
    await act(async () => {
      ;(host.state.gateway as ReturnType<typeof atom<string>>).set('open')
    })
    await startWithout('Hermes · Mac Mini')
    expect(calls('groups.create')[1][2]).toEqual(payload)
    expect(rooms.size).toBe(1)
    expect(onReplaced).toHaveBeenCalledTimes(outcome === 'refusal' ? 0 : 1)

    if (outcome === 'refusal') {
      await act(async () => {
        fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: words.startGatewayGroup }))
      })
    } else {
      await startWithout('Hermes · Mac Mini')
    }

    expect(calls('groups.create')[2][2].room_id).not.toBe(payload.room_id)
    expect(rooms.size).toBe(2)
  }
)

const joined = [{ room_id: 'linked', member_id: 'hermes-mac-mini', status: 'ready' }]
const sent = (params: Record<string, any>) => ({
  event: {},
  client_event_id: params.event_id,
  accepted: true,
  driver_started: true
})
const textbox = () => screen.getByRole('textbox') as HTMLTextAreaElement

/** A registration on the home that answers once the test releases it. */
function heldRegistration() {
  let release!: () => void
  const held = new Promise<void>(resolve => {
    release = resolve
  })

  return { release, register: () => held.then(() => PEER_REGISTERED) }
}

async function send(text: string) {
  await waitFor(() => expect(textbox().disabled).toBe(false))
  fireEvent.change(textbox(), { target: { value: text } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
}

it('renews a Bot that is due when a message is sent: the send is journaled first, leaves after the renewal, and the seat is fresh', async () => {
  union.profiles = []
  const { release, register } = heldRegistration()
  gateways({
    routes: joined,
    home: (method, params) =>
      method === 'groups.peer.register' ? register() : method === 'groups.send' ? sent(params) : undefined
  })
  const sending = { ...binding, roomId: 'renew-on-send' }
  render(<CanonicalGroupWorkspace binding={sending} />)
  // Its connection had not loaded when the room opened: nothing was renewed yet.
  await waitFor(() => expect(textbox().disabled).toBe(false))
  expect(calls('groups.peer.invite')).toHaveLength(0)
  union.profiles = [mini]
  await send('Keep this across a restart')
  // Journaled before it waits: a Desktop that quits now keeps it.
  await waitFor(async () =>
    expect((await readCanonicalGroupSend(sending))?.params.payload.text).toBe('Keep this across a restart')
  )
  expect(calls('groups.send')).toHaveLength(0)
  await act(async () => {
    release()
  })
  await waitFor(() => expect(calls('groups.send')).toHaveLength(1))
  expect(
    request.mock.calls.map(call => call[1]).filter(method => ['groups.peer.register', 'groups.send'].includes(method))
  ).toEqual(['groups.peer.register', 'groups.send'])
  // Fresh now: renewing again asks for nothing.
  renewPeers(sending, room, joined, [mini])
  await peerRenewalSettled(sending)
  expect(steps()).toEqual(SEATED)
})

it("waits for a renewal as long as the home may take to probe the Bot's gateway", async () => {
  vi.useFakeTimers()

  try {
    const { release, register } = heldRegistration()
    gateways({ home: method => (method === 'groups.peer.register' ? register() : undefined) })
    const probing = { ...binding, roomId: 'slow-probe' }
    renewPeers(probing, room, joined, [mini])
    await vi.waitFor(() => expect(calls('groups.peer.register')).toHaveLength(1))
    let waited = false
    void peerRenewalSettled(probing).then(() => {
      waited = true
    })
    // The home's probe of the member's gateway may take up to 30 s.
    await vi.advanceTimersByTimeAsync(31_000)
    expect(waited).toBe(false)
    release()
    await vi.advanceTimersByTimeAsync(0)
    expect(waited).toBe(true)
  } finally {
    vi.useRealTimers()
  }
})

it.each([
  // Nothing is attempted, so nothing is noted against the Bot.
  ['has not loaded yet', [], undefined],
  ["has other computers' Bots, not yet the Mac Mini's", [peerRosterRow('studio', 'Studio')], 'peer_connection_missing']
])('renews a Bot as soon as its connection loads, when the roster %s', async (_case, first, noted) => {
  gateways({
    elsewhere: {
      studio: method => (method === 'groups.capabilities' ? peerGroupCapabilities('install:studio') : undefined)
    }
  })
  const loading = { ...binding, roomId: `roster-${first.length}` }
  renewPeers(loading, room, joined, first)
  await peerRenewalSettled(loading)
  expect(
    ($peerJoinErrors.get()[peerJoinKey(loading, 'hermes-mac-mini')] as { reason?: string } | undefined)?.reason
  ).toBe(noted)
  renewPeers(loading, room, joined, [...first, mini])
  await waitFor(() => expect(steps()).toEqual(SEATED))
})

it.each(['Stop', 'close'] as const)('holds a journaled Send when %s interrupts renewal', async action => {
  const { release, register } = heldRegistration()
  gateways({
    routes: joined,
    home: (method, params) =>
      method === 'groups.peer.register'
        ? register()
        : method === 'groups.send'
          ? sent(params)
          : method === 'groups.stop'
            ? { cancelled: 0 }
            : undefined
  })
  const interrupted = { ...binding, roomId: `interrupt-renewal-${action}` }
  const view = render(<CanonicalGroupWorkspace binding={interrupted} />)
  await waitFor(() => expect(calls('groups.peer.register')).toHaveLength(1))
  await send('Keep these words')
  await waitFor(async () =>
    expect((await readCanonicalGroupSend(interrupted))?.params.payload.text).toBe('Keep these words')
  )

  if (action === 'close') {
    view.unmount()
  } else {
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: words.stopGroup }))
    })
  }
  await act(async () => {
    release()
    await peerRenewalSettled(interrupted)
  })
  expect(calls('groups.send')).toHaveLength(0)
  expect((await readCanonicalGroupSend(interrupted))?.params.payload.text).toBe('Keep these words')

  if (action === 'Stop') {
    expect(textbox().value).toBe('Keep these words')
  }
})

it('retries an unconfirmed renewal after five minutes even while the old route is ready', async () => {
  let now = Date.now()
  vi.spyOn(Date, 'now').mockImplementation(() => now)
  let lost = false
  const renewing = { ...binding, roomId: 'unconfirmed-renewal' }
  const current = { ...room, room_id: renewing.roomId }

  const state = gateways({
    routes: joined,
    home: method => {
      if (method === 'groups.state') {
        return { room: current, driver_status: { peer_routes: state.routes } }
      }

      if (method === 'groups.peer.register' && lost) {
        throw new Error('reply lost before registration')
      }
    }
  })

  const renew = async () => {
    renewPeers(renewing, current, state.routes, [mini])
    await peerRenewalSettled(renewing)
  }

  for (const [elapsed, status, unconfirmed, count] of [
    [0, 'ready', false, 1],
    [20 * 60 * 60 * 1000, 'ready', true, 2],
    [5 * 60 * 1000 - 1, 'ready', true, 2],
    [1, 'ready', true, 3],
    [5 * 60 * 60 * 1000, 'needs_reauthorization', false, 4],
    [0, 'ready', false, 4]
  ] as const) {
    now += elapsed
    lost = unconfirmed
    state.routes = [{ ...joined[0], status }]
    await renew()
    expect(calls('groups.peer.register')).toHaveLength(count)
  }
})

it.each(['create', 'disband'])(
  'keeps a newer replacement request when an older %s finishes after remount',
  async heldStep => {
    const { rooms, create } = strictGroupCreation()
    const pendingCreates: (() => void)[] = []
    const pendingDisbands: (() => void)[] = []
    let loseNewReply = false
    gateways({
      home: (method, params) => {
        if (method === 'groups.create') {
          const answer = create(params)

          if (loseNewReply) {
            loseNewReply = false
            throw new Error('new reply lost')
          }

          if (calls('groups.create').length <= 2) {
            return new Promise(resolve => pendingCreates.push(() => resolve(answer)))
          }

          return answer
        }

        if (method === 'groups.disband') {
          if (heldStep === 'disband' && calls('groups.disband').length <= 2) {
            return new Promise(resolve => pendingDisbands.push(() => resolve({})))
          }

          return {}
        }
      }
    })
    const onReplaced = vi.fn()
    const first = render(<CanonicalGroupWorkspace binding={binding} onReplaced={onReplaced} />)
    await startWithout('Hermes · Mac Mini')
    first.unmount()
    render(<CanonicalGroupWorkspace binding={binding} onReplaced={onReplaced} />)
    await startWithout('Hermes · Mac Mini')
    expect(calls('groups.create')[1][2]).toEqual(calls('groups.create')[0][2])
    await act(async () => pendingCreates[1]())

    if (heldStep === 'disband') {
      await act(async () => pendingCreates[0]())
      await act(async () => pendingDisbands[0]())
    }

    expect(onReplaced).toHaveBeenCalledTimes(1)
    loseNewReply = true
    await startWithout('Hermes · Mac Mini')
    expect(await screen.findByText('new reply lost')).toBeTruthy()
    const newer = calls('groups.create')[2][2]
    await act(async () => (heldStep === 'disband' ? pendingDisbands[1]() : pendingCreates[0]()))
    await act(async () => {
      fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: words.startGatewayGroup }))
    })
    expect(calls('groups.create')[3][2]).toEqual(newer)
    expect(rooms.size).toBe(2)
  }
)

it('offers Start again without only where the gateway lists groups.create', async () => {
  gateways({
    home: method =>
      method === 'groups.capabilities'
        ? { ...HOME, methods: HOME.methods.filter(name => name !== 'groups.create') }
        : undefined
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  const row = (await screen.findByText(say(words.peerNotJoined))).closest('[role="status"]') as HTMLElement
  expect(within(row).getByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(within(row).queryByRole('button', { name: say(words.peerStartWithout) })).toBeNull()
})

it('keeps a replacement that finishes after its view closed registered, without opening it', async () => {
  let finish!: () => void
  const held = new Promise<void>(resolve => {
    finish = resolve
  })
  const { rooms, create } = strictGroupCreation()
  gateways({
    home: (method, params) =>
      method === 'groups.create'
        ? held.then(() => create(params))
        : method === 'groups.disband'
          ? { tombstone: { room_id: 'linked', disbanded_at: 1 } }
          : undefined
  })
  const onReplaced = vi.fn()
  const view = render(<CanonicalGroupWorkspace binding={binding} onReplaced={onReplaced} />)
  await startWithout('Hermes · Mac Mini')
  await waitFor(() => expect(calls('groups.create')).toHaveLength(1))
  // The person closes the group chat (or the socket resets) while the new group is being made.
  view.unmount()
  await act(async () => {
    finish()
    await held
  })
  await waitFor(() => expect(calls('groups.disband')).toHaveLength(1))
  await act(async () => {})
  const [next] = [...rooms.values()]
  expect(Object.values($canonicalGroupBindings.get()).map(entry => entry.roomId)).toContain(next.room_id)
  expect(onReplaced).not.toHaveBeenCalled()
})
