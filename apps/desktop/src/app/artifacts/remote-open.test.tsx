import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

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

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: async () => ({
    sessions: [{ id: 'artifact-session', title: 'Fixture', profile: 'origin-profile' }]
  }),
  getAllSessionMessages: async () => ({
    messages: [
      {
        role: 'assistant',
        timestamp: 1000,
        content: paths.map(path => `MEDIA:${path}`).join(' ') + ' https://example.com/report.txt'
      }
    ]
  })
}))
afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.unstubAllGlobals()
})

it('keeps discovered file paths and originating session scope intact through remote opening', async () => {
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
})
