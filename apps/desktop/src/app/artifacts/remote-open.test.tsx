import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import type * as HermesClient from '@/hermes'
import { $connection } from '@/store/session'

import { ArtifactsView } from './index'

const paths = vi.hoisted(() => [
  '~/.hermes/memories/USER.md',
  './report.md',
  '../parent.md',
  String.raw`~\home.txt`,
  String.raw`.\child.txt`,
  String.raw`..\ancestor.txt`,
  'file:///C:/output/drive.txt',
  'file://server/share/unc.txt',
  '/srv/absolute.txt'
])

const getSessionMessages = vi.hoisted(() => vi.fn())
const listAllProfileSessions = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions,
  getSessionMessages
}))
afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

it('keeps discovered file paths and originating session scope intact through remote opening', async () => {
  listAllProfileSessions.mockResolvedValue({
    sessions: [{ id: 'artifact-session', title: 'Fixture', profile: 'origin-profile' }]
  })
  getSessionMessages.mockResolvedValue({
    messages: [
      {
        role: 'assistant',
        timestamp: 1000,
        content: paths.map(path => `MEDIA:${path}`).join(' ') + ' https://example.com/report.txt'
      }
    ],
    session_id: 'artifact-session'
  })
  const saveGatewayFile = vi.fn().mockResolvedValue({ saved: true })
  const openExternal = vi.fn()
  vi.stubGlobal('hermesDesktop', { saveGatewayFile, openExternal })
  $connection.set({
    isFullscreen: false,
    nativeOverlayWidth: 0,
    logs: [],
    windowButtonPosition: null,
    mode: 'remote',
    connectionId: 'remote-fixture',
    profile: 'writer',
    baseUrl: 'http://localhost',
    token: '',
    wsUrl: ''
  })
  render(
    <MemoryRouter>
      <ArtifactsView />
    </MemoryRouter>
  )

  for (const name of [
    'USER.md',
    'report.md',
    'parent.md',
    'home.txt',
    'child.txt',
    'ancestor.txt',
    'drive.txt',
    'unc.txt',
    'absolute.txt'
  ]) {
    fireEvent.click(await screen.findByRole('button', { name }))
  }

  await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledTimes(paths.length))
  expect(saveGatewayFile.mock.calls.map(([request]) => request)).toEqual(
    paths.map(path => ({
      connectionId: 'remote-fixture',
      profile: 'origin-profile',
      sessionId: 'artifact-session',
      path,
      suggestedName: path.split(/[\\/]/).pop()
    }))
  )
  expect(screen.getByRole('link').getAttribute('href')).toBe('https://example.com/report.txt')
  expect(openExternal).not.toHaveBeenCalled()
  expect(getSessionMessages).toHaveBeenCalledWith('artifact-session', 'origin-profile', {
    includeCompacted: true,
    limit: expect.any(Number),
    offset: 0,
    order: 'oldest'
  })
})

it('reads a registry-owned session through its owning connection', async () => {
  listAllProfileSessions.mockResolvedValueOnce({
    sessions: [{ id: 'voyo-session', title: 'Remote', profile: 'default', connection_id: 'voyo' }]
  })
  const client = await vi.importActual<typeof HermesClient>('@/hermes')
  const api = vi.fn().mockResolvedValue({ messages: [], session_id: 'voyo-session' })
  vi.stubGlobal('hermesDesktop', { api })
  getSessionMessages.mockImplementation(client.getSessionMessages)
  render(
    <MemoryRouter>
      <ArtifactsView />
    </MemoryRouter>
  )

  await waitFor(() =>
    expect(getSessionMessages).toHaveBeenCalledWith(
      'voyo-session',
      { connectionId: 'voyo', profile: 'default' },
      expect.objectContaining({ includeCompacted: true, offset: 0, order: 'oldest' })
    )
  )

  expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'voyo', profile: 'default' }))
  const path = new URL(api.mock.calls[0][0].path, 'http://localhost')
  expect(path.pathname).toBe('/api/sessions/voyo-session/messages')
  expect(Object.fromEntries(path.searchParams)).toMatchObject({
    profile: 'default',
    include_compacted: 'true',
    offset: '0',
    order: 'oldest'
  })
})
