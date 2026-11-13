import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setSessionOwnerResolver } from '@/api/client'
import { $connection } from '@/store/session'

import { downloadGatewayMediaFile, resolveMediaPlaybackSrc } from './media'

const original = window.hermesDesktop
const stream = vi.fn(async () => 'hermes-media://clip')
const save = vi.fn(async () => ({ saved: false }))

beforeEach(() => {
  stream.mockClear()
  save.mockClear()
  window.hermesDesktop = { getGatewayFileStreamUrl: stream, saveGatewayFile: save } as never
  $connection.set({ connectionId: 'gateway-B', mode: 'remote', profile: 'foreground' } as never)
  setApiRequestConnection('gateway-B')
  setSessionOwnerResolver(() => ({ connectionId: 'gateway-B', profile: 'profile-B' }))
})

afterEach(() => {
  window.hermesDesktop = original
  $connection.set(null)
  setApiRequestConnection(null)
  setSessionOwnerResolver(null)
})

it.each([
  { connectionId: 'gateway-A', profile: 'profile-A' },
  { connectionId: 'gateway-A' },
  { profile: 'profile-A' }
])('preserves the explicit playback owner %j without borrowing a conflicting session route', async owner => {
  await resolveMediaPlaybackSrc('/workspace/clip.mp4', owner, 'shared-session')
  expect(stream).toHaveBeenCalledWith({
    ...owner, path: '/workspace/clip.mp4', ...(owner.profile ? { sessionId: 'shared-session' } : {})
  })
})

it('still resolves a playback session without an explicit owner', async () => {
  await resolveMediaPlaybackSrc('/workspace/clip.mp4', undefined, 'shared-session')
  expect(stream).toHaveBeenCalledWith({
    connectionId: 'gateway-B', profile: 'profile-B', path: '/workspace/clip.mp4', sessionId: 'shared-session'
  })
})

it('keeps an explicit legacy download profile over a conflicting session resolver', async () => {
  await downloadGatewayMediaFile('/workspace/note.md', { profile: 'profile-A', sessionId: 'shared-session' })
  expect(save).toHaveBeenCalledWith({
    profile: 'profile-A', path: '/workspace/note.md', sessionId: 'shared-session', suggestedName: 'note.md'
  })
})

it('does not send an unknown session to the foreground download owner', async () => {
  setSessionOwnerResolver(() => null)
  await downloadGatewayMediaFile('/workspace/note.md', { sessionId: 'unknown' })
  expect(save).toHaveBeenCalledWith({
    connectionId: 'gateway-B', profile: 'foreground', path: '/workspace/note.md', suggestedName: 'note.md'
  })
})
