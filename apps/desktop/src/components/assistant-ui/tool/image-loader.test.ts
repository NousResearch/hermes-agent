import { afterEach, expect, it, vi } from 'vitest'

import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { readToolImage } from './image-loader'

it.each([
  [
    { id: 'legacy', profile: 'profile-a' },
    { id: 'legacy', profile: 'profile-b' }
  ],
  [
    { id: 'legacy', connection_id: 'host-a', profile: 'profile-a' },
    { id: 'legacy', profile: 'profile-b' }
  ]
])('rejects ambiguous legacy profile owners without reading a file: %j', async (first, second) => {
  $sessions.set([first, second] as never)
  const api = vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  await expect(readToolImage('./shot.png', { sessionId: 'legacy', runtimeId: 'unknown' })).rejects.toThrow('ambiguous')
  expect(api).not.toHaveBeenCalled()
})

it('preserves a unique bare profile route without inventing a connection id', async () => {
  $sessions.set([
    { id: 'legacy', profile: 'profile-a' },
    { id: 'legacy', profile: 'profile-a' }
  ] as never)
  const api = vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  await expect(readToolImage('./shot.png', { sessionId: 'legacy', runtimeId: 'unknown' })).resolves.toBe(IMAGE)
  expect(api.mock.calls[0][0].profile).toBe('profile-a')
  expect(api.mock.calls[0][0].connectionId).toBeUndefined()
})

const IMAGE = 'data:image/png;base64,AAAA'
afterEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $sessionTiles.set([])
  _resetSessionOwnerHintsForTests()
  vi.restoreAllMocks()
})

it('uses an exact tile owner for duplicate stored ids and never guesses an ambiguous owner', async () => {
  const rows = [
    { id: 'cloned', connection_id: 'host-a', profile: 'default' },
    { id: 'cloned', connection_id: 'host-b', profile: 'default' }
  ]

  $sessions.set(rows as never)
  const api = vi.fn().mockResolvedValue({ dataUrl: IMAGE })
  window.hermesDesktop = { api } as unknown as typeof window.hermesDesktop
  await expect(readToolImage('./shot.png', { sessionId: 'cloned', runtimeId: 'unknown' })).rejects.toThrow('ambiguous')
  expect(api).not.toHaveBeenCalled()
  $sessionTiles.set([
    { storedSessionId: 'cloned', runtimeId: 'runtime-b', ownerRoute: { connectionId: 'host-b', profile: 'default' } }
  ] as never)
  await expect(readToolImage('./shot.png', { sessionId: 'cloned', runtimeId: 'runtime-b' })).resolves.toBe(IMAGE)
  expect(api).toHaveBeenLastCalledWith(
    expect.objectContaining({
      connectionId: 'host-b',
      path: '/api/fs/read-data-url?path=.%2Fshot.png&session_id=cloned'
    })
  )
})

it('prefers a proven unique owner hint and rejects a non-image response without local fallback', async () => {
  $sessions.set([{ id: 'stored', connection_id: 'stale-host', profile: 'default' }] as never)
  setSessionOwnerHint('stored', { connectionId: 'actual-host', profile: 'artist' })
  $sessionTiles.set([
    {
      storedSessionId: 'stored',
      runtimeId: 'different-runtime',
      ownerRoute: { connectionId: 'stale-host', profile: 'default' }
    }
  ] as never)
  const api = vi.fn().mockResolvedValue({ dataUrl: 'data:text/plain;base64,AAAA' })
  const local = vi.fn()
  window.hermesDesktop = { api, readFileDataUrl: local } as unknown as typeof window.hermesDesktop
  await expect(readToolImage('./shot.png', { sessionId: 'stored', runtimeId: 'runtime' })).rejects.toThrow(
    'Not an image'
  )
  expect(api).toHaveBeenLastCalledWith(expect.objectContaining({ connectionId: 'actual-host', profile: 'artist' }))
  expect(local).not.toHaveBeenCalled()
})
