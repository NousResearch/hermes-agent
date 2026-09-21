import { afterEach, expect, it, vi } from 'vitest'

import { closeSecondaryGateways, setPrimaryGateway, setPrimaryGatewayConnection } from '@/store/gateway'
import { $newChatProfile, $newChatRoute, captureNewChatSource, resolveNewChatOwnerRoute } from '@/store/profile'
import { $connection, setSessions } from '@/store/session'
import { $sessionTiles, isSessionRemote } from '@/store/session-states'

import { uploadComposerAttachment } from '.'

afterEach(() => {
  $sessionTiles.set([])
  setSessions([])
  $connection.set(null)
  setPrimaryGateway(null)
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  captureNewChatSource(null)
  vi.unstubAllGlobals()
})

it.each([
  ['ssh-source', 'remote'],
  ['local', 'remote'],
  ['local', 'local'],
  ['unresolved-source', null]
] as const)(
  'stages images and files using the %s owner (%s), including restored routes',
  async (connectionId, mode) => {
    closeSecondaryGateways()
    setPrimaryGateway({} as never)
    setPrimaryGatewayConnection({ connectionId, ...(mode ? { mode } : {}) })
    $connection.set({ connectionId: 'foreground', mode: mode === 'local' ? 'remote' : 'local' } as never)
    $newChatProfile.set('default')
    captureNewChatSource(connectionId)
    const ownerRoute = resolveNewChatOwnerRoute()!
    const dataUrl = 'data:application/octet-stream;base64,aGVybWVz'
    vi.stubGlobal('hermesDesktop', { readFileDataUrlForAttach: vi.fn(async () => dataUrl) })

    const request = vi.fn(async () => ({ attached: true, path: '/staged/attachment', ref_text: 'attachment' }))

    for (const restored of [false, true]) {
      $sessionTiles.set(restored ? [] : [{ storedSessionId: 'attachment-session', ownerRoute }])
      setSessions(
        restored ? [{ id: 'attachment-session', connection_id: connectionId, profile: 'default' } as never] : []
      )

      for (const kind of ['image', 'file'] as const) {
        request.mockClear()
        const path = `/client/Application Support/attachment.${kind === 'image' ? 'png' : 'txt'}`
        await uploadComposerAttachment(
          {
            id: kind,
            kind,
            label: 'attachment',
            path,
            previewUrl: 'data:image/png;base64,aGVybWVz'
          },
          {
            remote: isSessionRemote('attachment-session'),
            sessionId: 'attachment-session',
            requestGateway: request as never
          }
        )

        const bytes = mode !== 'local'
        expect(request).toHaveBeenCalledExactlyOnceWith(
          kind === 'image' ? (bytes ? 'image.attach_bytes' : 'image.attach') : 'file.attach',
          kind === 'image'
            ? {
                session_id: 'attachment-session',
                ...(bytes ? { content_base64: 'aGVybWVz', filename: 'attachment.png' } : { path })
              }
            : { session_id: 'attachment-session', name: 'attachment', path, ...(bytes ? { data_url: dataUrl } : {}) }
        )
      }
    }
  }
)
