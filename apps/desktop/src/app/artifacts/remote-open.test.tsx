import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
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

const getSessionMessages = vi.hoisted(() => vi.fn())
const getHermesConfigRecord = vi.hoisted(() => vi.fn(async () => ({})))

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: async () => ({
    sessions: [{ id: 'artifact-session', title: 'Fixture', profile: 'origin-profile' }]
  }),
  getSessionMessages,
  getHermesConfigRecord
}))
afterEach(() => {
  cleanup()
  $connection.set(null)
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

it('replays an in-flight scan with config that arrives before its transcript', async () => {
  let resolveConfig!: (value: object) => void
  let resolveMessages!: (value: object) => void
  getHermesConfigRecord.mockImplementationOnce(() => new Promise(resolve => { resolveConfig = resolve }))
  const page = { messages: [{ role: 'assistant', timestamp: 1000,
    content: 'https://example.com/loading.png MEDIA:/tmp/keep.txt' }] }
  getSessionMessages.mockImplementationOnce(() => new Promise(resolve => { resolveMessages = resolve }))
  getSessionMessages.mockResolvedValue(page)
  render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter><ArtifactsView /></MemoryRouter>
  </QueryClientProvider>)
  await waitFor(() => expect(getSessionMessages).toHaveBeenCalled())
  await act(async () => { resolveConfig({ desktop: { artifacts: { ignore: ['loading[.]png'] } } }) })
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 50)) })
  await act(async () => { resolveMessages(page) })
  await waitFor(() => expect(getSessionMessages).toHaveBeenCalledTimes(2))
  await screen.findByRole('button', { name: 'keep.txt' })
  await waitFor(() => expect(screen.queryByRole('link')).toBeNull())
})

it('keeps discovered file paths and originating session scope intact through remote opening', async () => {
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
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter>
      <ArtifactsView />
    </MemoryRouter></QueryClientProvider>
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
