import { webcrypto } from 'node:crypto'

import { afterEach, expect, it, vi } from 'vitest'

import { host } from '@/sdk'

// @ts-expect-error The bundled plugin is a plain JavaScript SDK consumer.
import { openRealmViewer } from '../../../../plugins/hermes-realms/desktop/plugin.js'

interface KeepAliveInput {
  onKeepAlive: () => Promise<void>
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it.each(['watch', 'popout'])('renews %s only through the originally captured authenticated owner', async target => {
  vi.stubGlobal('crypto', webcrypto)
  const token = 'a'.repeat(43)

  const rest = vi.fn(async (path: string) =>
    path.endsWith('/watch') ? { url: 'http://127.0.0.1:1234/realms/r-fixture/view#ticket=' + token } : { renewed: true }
  )

  const preview = vi.spyOn(host, 'openPreview').mockResolvedValue(true)
  const popup = vi.fn(async () => true)

  const session = {
    connectionId: 'local',
    profile: 'original',
    storedSessionId: 'history',
    runtimeSessionId: 'runtime'
  }

  const original = { ...session }
  const realm = { id: 'r-fixture', state: 'live', stored_session_id: 'history', runtime_session_id: 'runtime' }
  let enabled = true
  await openRealmViewer(
    { rest, os: { openViewer: popup } },
    session,
    realm,
    target,
    () => true,
    () => enabled
  )
  const calls = target === 'watch' ? preview.mock.calls : popup.mock.calls
  const input = calls[0][0] as unknown as KeepAliveInput
  session.profile = 'other'
  await input.onKeepAlive()
  expect(rest).toHaveBeenLastCalledWith('/realms/r-fixture/renew', {
    method: 'POST',
    scope: original,
    body: { stored_session_id: 'history', runtime_session_id: 'runtime', viewer_token: token }
  })
  enabled = false
  await expect(input.onKeepAlive()).rejects.toThrow()
  expect(rest).toHaveBeenCalledTimes(2)
})
