import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { CANONICAL_GROUP_CAPABILITIES, localGatewaySdkMock } from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'
import type { CanonicalRoomMember } from './canonical-groups'
import { $botMeta, $lastRoster } from './data'
import { RoomBotAvatar } from './group-room-avatar'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async importOriginal =>
  localGatewaySdkMock(await importOriginal<typeof HermesSdk>(), request, () => (key: string) =>
    CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key))

const localPhoto = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB'
const remotePhoto = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAAC'
const member: CanonicalRoomMember = { member_id: 'writer', profile: 'default', handle: 'writer' }

const peer: CanonicalRoomMember = { ...member, member_id: 'peer-writer', target: {
  kind: 'peer', installation_id: 'install:peer-machine', profile: 'default'
} }

beforeEach(() => { $botMeta.set({}); $lastRoster.set([]) })
afterEach(() => { cleanup(); $botMeta.set({}); $lastRoster.set([]); request.mockReset(); localStorage.clear() })

it('renders the stored connection-scoped photo without needing a roster fetch', () => {
  $botMeta.set({ 'local::default': { image: localPhoto, custom: true } })
  const { container } = render(<RoomBotAvatar connectionId="local" member={member} name="default" />)
  expect(container.querySelector('img')?.getAttribute('src')).toBe(localPhoto)
})

it('uses the room connection rather than another connection with the same profile', () => {
  $botMeta.set({ 'local::default': { image: localPhoto }, 'remote::default': { image: remotePhoto } })
  const { container } = render(<RoomBotAvatar connectionId="remote" member={member} name="default" />)
  expect(container.querySelector('img')?.getAttribute('src')).toBe(remotePhoto)
})

it('matches a peer by installation and target profile, not its room-local profile or handle', () => {
  $botMeta.set({ 'local::default': { image: localPhoto }, 'mini::alias': { image: remotePhoto } })
  $lastRoster.set([{ name: 'alias', installId: 'peer-machine', sourceScoped: true,
    route: { connectionId: 'mini', profile: 'alias', targetProfile: 'default', mode: 'remote' } }])
  const { container } = render(<RoomBotAvatar connectionId="local" member={peer} name="default" />)
  expect(container.querySelector('img')?.getAttribute('src')).toBe(remotePhoto)
})

it.each(['missing', 'wrong installation', 'wrong profile', 'removed owner'])('does not borrow a same-name photo for a peer with %s', state => {
  $botMeta.set({ default: { image: localPhoto }, 'local::default': { image: localPhoto }, 'mini::default': { image: remotePhoto } })

  if (state !== 'missing') {
    $lastRoster.set([{ name: 'default', installId: state === 'wrong installation' ? 'other-machine' : 'peer-machine',
      sourceScoped: true, connectionId: state === 'removed owner' ? undefined : 'mini',
      targetProfile: state === 'wrong profile' ? 'reviewer' : 'default' }])
  }

  const { container } = render(<RoomBotAvatar connectionId="local" member={peer} name="default" />)
  expect(container.querySelector('img')).toBeNull()
  expect(container.querySelector('svg')).toBeTruthy()
})

it('repaints an open room when a stored photo changes', () => {
  $botMeta.set({ 'local::default': { image: localPhoto } })
  const { container } = render(<RoomBotAvatar connectionId="local" member={member} name="default" />)
  expect(container.querySelector('img')?.getAttribute('src')).toBe(localPhoto)
  act(() => { $botMeta.set({ 'local::default': { image: remotePhoto } }) })
  expect(container.querySelector('img')?.getAttribute('src')).toBe(remotePhoto)
})

it('keeps an unconfigured member on a generated face', () => {
  const { container } = render(<RoomBotAvatar connectionId="local" member={member} name="default" />)
  expect(container.querySelector('img')).toBeNull()
  expect(container.querySelector('svg')).toBeTruthy()
})

it('resolves an alias on the room connection by its target profile', () => {
  $botMeta.set({ 'remote::alias': { image: remotePhoto } })
  $lastRoster.set([{ name: 'alias', sourceScoped: true,
    route: { connectionId: 'remote', profile: 'alias', targetProfile: 'default', mode: 'remote' } }])
  const { container } = render(<RoomBotAvatar connectionId="remote" member={member} name="default" />)
  expect(container.querySelector('img')?.getAttribute('src')).toBe(remotePhoto)
})

it('passes the saved photo through the real room header, message, typing, pending card and details', async () => {
  $botMeta.set({ 'local::default': { image: localPhoto, custom: true } })
  const room = { room_id: 'avatar-test', name: 'Avatar room', authority_epoch: 1, latest_seq: 1, members: [member] }
  request.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {return CANONICAL_GROUP_CAPABILITIES}

    if (method === 'groups.state') {
      return { room, driver_status: { working: true, member_activity: [{ member_id: member.member_id, status: 'running' }],
        pending_actions: [{ kind: 'retry', member_id: member.member_id, task_id: 'avatar-task', execution_generation: 1 }] } }
    }

    if (method === 'groups.log') {
      return { events: [{ seq: 1, kind: 'message.member', actor: { kind: 'member', id: member.member_id }, payload: { text: 'Avatar message' } }] }
    }

    return {}
  })
  const { container } = render(<CanonicalGroupWorkspace binding={{ connectionId: 'local', profile: 'default', roomId: room.room_id }} />)
  await screen.findByText('Avatar message')
  const photoIn = (selector: string) => container.querySelector(`${selector} img`)?.getAttribute('src')
  expect(photoIn('ul')).toBe(localPhoto)
  expect(photoIn('[data-pending-action]')).toBe(localPhoto)
  const log = screen.getByRole('log')
  expect([...log.querySelectorAll('img')].filter(img => img.getAttribute('src') === localPhoto).length).toBe(3)
  fireEvent.click(screen.getByRole('button', { name: CANONICAL_GROUP_LOCALES.en.groupDetails }))
  await waitFor(() => expect(photoIn('aside')).toBe(localPhoto))
  expect(request.mock.calls.some(([, method]) => method === 'groups.send')).toBe(false)
})
