import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GroupRow } from './bot-row'
import { $groupChats } from './group-chat'
import { GroupChatWorkspace } from './group-chat-view'
import { scriptedStorage } from './group-test-utils'
import { translateBots } from './i18n-test-helper'
import type { GroupChat, GroupMember } from './types'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, usePluginI18n: () => translateBots }
})

const GROUP = 'Project group'
const noop = () => undefined
const scrollDescriptor = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'scrollIntoView')

const members: GroupMember[] = [
  { name: 'planner', sourceReachable: true },
  { name: 'builder', sourceReachable: true },
  { name: 'ux', sourceMissing: true },
  { name: 'reviewer', sourceMissing: true }
]

function setRoom(hosted: boolean, extra: Partial<GroupChat> = {}) {
  $groupChats.set({
    [GROUP]: {
      roomId: 'project-room',
      hosted: hosted ? 'authority' : null,
      hostedStatus: hosted ? { state: 'idle', label: 'Ready' } : undefined,
      members,
      log: [],
      watermarks: {},
      running: false,
      ...extra
    }
  })
}

function row(selected = members) {
  return <GroupRow active={false} group={GROUP} members={selected} needsYou={false} onDisband={noop} onOpen={noop} />
}

async function withHostFailure(state: 'offline' | 'unsupported', check: (room: GroupChat) => void) {
  const runtime = await import('./hosted-room-runtime')

  const routes = vi
    .spyOn(host, 'profileRoutes')
    .mockResolvedValue([{ connectionId: 'host-route', mode: 'remote', profile: 'default', targetProfile: 'default' }])

  const request = vi.spyOn(host, 'requestProfile').mockImplementation(async (_route, method) => {
    if (method !== 'groups.capabilities') {
      throw new Error(`Unexpected RPC: ${method}`)
    }

    if (state === 'offline') {
      throw new Error('Connection failed')
    }

    throw Object.assign(new Error('groups.capabilities method not found'), { code: -32601 })
  })

  try {
    await act(async () => runtime.startHostedRoomRuntime(scriptedStorage(new Map()).storage))
    const room = $groupChats.get()[GROUP]
    expect(room.hostedStatus?.state).toBe(state)
    check(room)
  } finally {
    runtime.stopHostedRoomRuntime()
    runtime.$hostedRoomCapabilities.set({})
    routes.mockRestore()
    request.mockRestore()
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  $groupChats.set({})
})

afterEach(() => {
  cleanup()
  $groupChats.set({})

  if (scrollDescriptor) {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', scrollDescriptor)
  } else {
    Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView')
  }
})

describe('group availability follows its execution mode', () => {
  it('does not label a hosted group partly unavailable from Desktop peer connections', () => {
    setRoom(true)
    render(row())
    expect(screen.getByRole('button', { name: 'Project group, 4 bots' })).toBeTruthy()
    expect(screen.queryByLabelText('2 of 4 available')).toBeNull()
  })

  it('keeps direct-connection availability on classic group rows', () => {
    setRoom(false)
    render(row())
    expect(screen.getByRole('button', { name: 'Project group, 4 bots, 2 of 4 available' })).toBeTruthy()
    expect(screen.getByLabelText('2 of 4 available', { exact: true })).toBeTruthy()
  })

  it('does not dim a hosted group when none of its members has a direct Desktop connection', () => {
    const remote = members.map(member => ({ ...member, sourceReachable: false, sourceMissing: true }))
    setRoom(true, { members: remote })
    const { container } = render(row(remote))
    expect(screen.getByRole('button', { name: 'Project group, 4 bots' })).toBeTruthy()
    expect(container.querySelector('.opacity-60')).toBeNull()
    expect(screen.queryByLabelText('0 of 4 available')).toBeNull()
  })

  it('updates the row when the group changes from classic to hosted', () => {
    setRoom(false)
    render(row())
    expect(screen.getByLabelText('2 of 4 available', { exact: true })).toBeTruthy()
    act(() => setRoom(true))
    expect(screen.getByRole('button', { name: 'Project group, 4 bots' })).toBeTruthy()
    expect(screen.queryByLabelText('2 of 4 available')).toBeNull()
  })

  it.each(['offline', 'unsupported'] as const)('shows membership without replacing runtime %s errors', async state => {
    setRoom(true, { hostedConnectionId: 'host-route' })
    render(<GroupChatWorkspace group={GROUP} members={members} />)
    await withHostFailure(state, room => {
      expect(screen.queryByLabelText('2 of 4 available')).toBeNull()
      expect(screen.getByLabelText('4 bots', { exact: true })).toBeTruthy()
      expect(screen.getAllByText(room.hostedStatus!.label).length).toBeGreaterThan(0)
    })
  })

  it('retains classic availability in the chat header', () => {
    setRoom(false)
    render(<GroupChatWorkspace group={GROUP} members={members} />)
    expect(screen.getByLabelText('2 of 4 available', { exact: true })).toBeTruthy()
  })

  for (const state of ['offline', 'unsupported'] as const) {
    it.each([null, 'data:image/png;base64,iVBORw0KGgo='])(
      'keeps runtime ' + state + ' visible with image=%s',
      async image => {
        setRoom(true, { hostedConnectionId: 'host-route', image })
        const { container } = render(row())
        await withHostFailure(state, room => {
          expect(
            screen.getByRole('button', { name: `Project group, 4 bots, ${room.hostedStatus!.label}` })
          ).toBeTruthy()
          expect(screen.getByLabelText(room.hostedStatus!.label, { exact: true })).toBeTruthy()
          expect(container.querySelector(image ? 'img.grayscale.opacity-60' : '.opacity-60')).toBeTruthy()
          expect(screen.queryByLabelText('2 of 4 available')).toBeNull()
        })
      }
    )
  }

  it('keeps all-unavailable classic groups visibly degraded', () => {
    const missing = members.map(member => ({ ...member, sourceMissing: true }))
    setRoom(false, { members: missing })
    const { container } = render(row(missing))
    expect(screen.getByLabelText('0 of 4 available', { exact: true })).toBeTruthy()
    expect(container.querySelector('.opacity-60')).toBeTruthy()
  })
})
