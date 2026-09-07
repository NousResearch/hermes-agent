import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { ArtifactsView } from './index'

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: async () => ({ sessions: [{ id: 'artifact-session', title: 'Fixture' }] }),
  getAllSessionMessages: async () => ({ messages: [{ role: 'assistant', timestamp: 1000,
    content: '~/.hermes/memories/USER.md ./report.md ../parent.md' }] })
}))

afterEach(() => { cleanup(); $connection.set(null); vi.unstubAllGlobals() })

it('opens remote tilde and relative artifacts through their gateway without client path expansion', async () => {
  const saveGatewayFile = vi.fn().mockResolvedValue({ saved: true })
  const openExternal = vi.fn().mockRejectedValue(new Error('Invalid external URL'))
  vi.stubGlobal('hermesDesktop', { saveGatewayFile, openExternal })
  $connection.set({ isFullscreen: false, nativeOverlayWidth: 0, logs: [], windowButtonPosition: null, mode: 'remote', connectionId: 'remote-fixture', profile: 'writer', baseUrl: 'http://localhost', token: '', wsUrl: '' })
  render(<MemoryRouter><ArtifactsView /></MemoryRouter>)

  for (const name of ['USER.md', 'report.md', 'parent.md']) {
    fireEvent.click(await screen.findByRole('button', { name }))
  }

  await waitFor(() => expect(saveGatewayFile).toHaveBeenCalledTimes(3))
  expect(saveGatewayFile.mock.calls.map(([request]) => request)).toEqual([
    { connectionId: 'remote-fixture', profile: 'writer', path: '~/.hermes/memories/USER.md', suggestedName: 'USER.md' },
    { connectionId: 'remote-fixture', profile: 'writer', path: './report.md', suggestedName: 'report.md' },
    { connectionId: 'remote-fixture', profile: 'writer', path: '../parent.md', suggestedName: 'parent.md' }
  ])
  expect(openExternal).not.toHaveBeenCalled()
})
