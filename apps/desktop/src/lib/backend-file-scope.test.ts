import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile, setSessionOwnerResolver } from '@/api/client'
import { $connection } from '@/store/session'

import { readDesktopFileDataUrl, readDesktopFileText, writeDesktopFileText } from './desktop-fs'
import { normalizeOrLocalPreviewTarget } from './local-preview'

const original = window.hermesDesktop
const sessionId = 'shared-durable-id'
const owner = { connectionId: 'gateway-A', profile: 'profile-A' }
const scope = { backendOwned: true, owner, sessionId }
const api = vi.fn(async () => ({ dataUrl: 'data:image/png;base64,QQ==', text: 'A', path: '/workspace/note.md' }))

beforeEach(() => {
  api.mockClear()
  window.hermesDesktop = { api } as never
  $connection.set({ connectionId: 'gateway-B', mode: 'remote', profile: 'profile-B' } as never)
  setApiRequestConnection('gateway-B')
  setApiRequestProfile('profile-B')
  setSessionOwnerResolver(() => ({ connectionId: 'gateway-B', profile: 'profile-B' }))
})

afterEach(() => {
  window.hermesDesktop = original
  $connection.set(null)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  setSessionOwnerResolver(null)
})

it.each(['gateway-A', 'local'])('keeps the complete captured %s owner authoritative over an equal session ID elsewhere', async connectionId => {
  const captured = { ...scope, owner: { ...owner, connectionId } }
  await readDesktopFileDataUrl('/workspace/result.png', captured)
  await readDesktopFileText('/workspace/note.md', captured)
  await normalizeOrLocalPreviewTarget('/workspace/note.md', '/workspace', captured)
  await writeDesktopFileText('/workspace/note.md', 'edited', captured)

  expect(api).toHaveBeenCalledTimes(4)

  for (const [request] of api.mock.calls as unknown as [{ connectionId: string; profile: string; path: string; body?: { session_id: string } }][]) {
    expect(request).toMatchObject({ connectionId, profile: 'profile-A' })
    expect(request.body?.session_id ?? new URL(request.path, 'http://backend').searchParams.get('session_id')).toBe(sessionId)
  }
})

it('does not combine a partial explicit connection with another gateway owner profile or session', async () => {
  await readDesktopFileText('/workspace/note.md', { ...scope, owner: { connectionId: 'gateway-A' } })
  expect(api).toHaveBeenCalledWith({ connectionId: 'gateway-A', path: '/api/fs/read-text?path=%2Fworkspace%2Fnote.md' })
})

it('keeps a profile-only explicit owner instead of borrowing another session owner', async () => {
  setSessionOwnerResolver(() => ({ connectionId: 'gateway-C', profile: 'profile-B' }))
  const captured = { ...scope, owner: { profile: 'profile-A' } }
  await readDesktopFileDataUrl('/workspace/result.png', captured)
  await readDesktopFileText('/workspace/note.md', captured)
  await normalizeOrLocalPreviewTarget('/workspace/note.md', '/workspace', captured)
  await writeDesktopFileText('/workspace/note.md', 'edited', captured)
  expect(api).toHaveBeenCalledTimes(4)
  for (const [request] of api.mock.calls as unknown as [{ connectionId?: string; profile: string }][]) {
    expect(request.profile).toBe('profile-A')
    expect(request.connectionId).toBe('gateway-B')
  }
})

it('retains resolver fallback when no explicit owner is supplied', async () => {
  await readDesktopFileText('/workspace/note.md', { backendOwned: true, sessionId })
  expect(api).toHaveBeenCalledWith({ connectionId: 'gateway-B', profile: 'profile-B', path: `/api/fs/read-text?path=%2Fworkspace%2Fnote.md&session_id=${sessionId}` })
})
