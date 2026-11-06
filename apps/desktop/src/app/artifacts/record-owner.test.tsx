import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { $connection } from '@/store/session'
import type { SessionInfo, SessionMessage } from '@/types/hermes'

import { collectArtifactsForSession } from './artifact-utils'

import { ArtifactsView } from './index'

const listAllProfileSessions = vi.hoisted(() => vi.fn())
const getSessionMessages = vi.hoisted(() => vi.fn())
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  listAllProfileSessions,
  getSessionMessages
}))

afterEach(() => {
  cleanup()
  $connection.set(null)
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

it('keeps artifact identities distinct across producing connections and profiles', () => {
  const message = { role: 'assistant', content: 'MEDIA:/srv/report.txt', timestamp: 1 } as SessionMessage

  const scopes = [
    { connection_id: 'producer-a', profile: 'writer' },
    { connection_id: 'producer-b', profile: 'writer' },
    { connection_id: 'producer-a', profile: 'analyst' },
    { connection_id: 'local', profile: 'writer' },
    { profile: 'writer' }
  ]

  const records = scopes.map(
    scope => collectArtifactsForSession({ ...scope, id: 'same-session' } as SessionInfo, [message])[0]
  )

  expect(new Set(records.map(record => record.id)).size).toBe(scopes.length)
  expect(records.map(record => record.owner)).toEqual(
    scopes.map(scope =>
      scope.connection_id ? { connectionId: scope.connection_id, profile: scope.profile } : undefined
    )
  )
})

it('uses the producer for preview and download while preserving explicit local and legacy routes', async () => {
  const fixtures = [
    { owner: 'producer-a', foreground: 'foreground-b', expected: 'producer-a' },
    { owner: 'producer-a', foreground: 'local', expected: 'producer-a' },
    { owner: 'local', foreground: 'foreground-b', expected: 'local' },
    { owner: undefined, foreground: 'foreground-b', expected: 'foreground-b' }
  ]

  for (const fixture of fixtures) {
    const api = vi.fn().mockResolvedValue('data:image/png;base64,QQ==')
    const readFileDataUrl = vi.fn().mockResolvedValue('data:image/png;base64,TA==')
    const saveGatewayFile = vi.fn().mockResolvedValue({ saved: true })
    const openExternal = vi.fn()
    vi.stubGlobal('hermesDesktop', { api, readFileDataUrl, saveGatewayFile, openExternal })
    $connection.set({
      isFullscreen: false,
      nativeOverlayWidth: 0,
      logs: [],
      windowButtonPosition: null,
      mode: fixture.foreground === 'local' ? 'local' : 'remote',
      authMode: 'oauth',
      remoteKind: 'url',
      connectionId: fixture.foreground,
      profile: 'foreground-profile',
      baseUrl: 'http://foreground.invalid',
      token: 'foreground-token',
      wsUrl: ''
    })
    setApiRequestConnection(fixture.foreground)
    setApiRequestProfile('foreground-profile')
    listAllProfileSessions.mockResolvedValueOnce({
      sessions: [{ id: 'same-session', profile: 'writer', connection_id: fixture.owner }]
    })
    getSessionMessages.mockResolvedValue({
      messages: [{ role: 'assistant', content: 'MEDIA:/srv/report.txt MEDIA:/srv/image.png', timestamp: 1 }]
    })

    render(
      <MemoryRouter>
        <ArtifactsView />
      </MemoryRouter>
    )
    fireEvent.click(await screen.findByRole('button', { name: 'report.txt' }))
    await screen.findByAltText('image.png')

    if (fixture.expected === 'local') {
      expect(readFileDataUrl).toHaveBeenCalledWith('/srv/image.png')
      expect(api).not.toHaveBeenCalled()
      expect(openExternal).toHaveBeenCalledWith('file:///srv/report.txt')
      expect(saveGatewayFile).not.toHaveBeenCalled()
    } else {
      await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledTimes(1))
      expect(saveGatewayFile).toHaveBeenCalledWith({
        connectionId: fixture.expected,
        profile: 'writer',
        sessionId: 'same-session',
        path: '/srv/report.txt',
        suggestedName: 'report.txt'
      })
      expect(api).toHaveBeenCalledWith(
        expect.objectContaining({
          connectionId: fixture.expected,
          profile: fixture.owner ? 'writer' : 'foreground-profile',
          path: '/api/fs/read-data-url?path=%2Fsrv%2Fimage.png'
        })
      )
      expect(readFileDataUrl).not.toHaveBeenCalled()
      expect(openExternal).not.toHaveBeenCalled()
    }

    cleanup()
  }
})
