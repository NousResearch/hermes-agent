import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { ArtifactsView } from './index'

const getSessionMessages = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: async () => ({
    sessions: [{ id: 'artifact-session', title: 'Fixture' }]
  }),
  getSessionMessages
}))
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

it('shows a placeholder instead of an empty tile when an image artifact file is gone', async () => {
  getSessionMessages.mockResolvedValue({
    messages: [{ role: 'assistant', timestamp: 1000, content: 'MEDIA:/tmp/deleted-shot.png' }],
    session_id: 'artifact-session'
  })
  const readFileDataUrl = vi.fn().mockRejectedValue(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }))
  vi.stubGlobal('hermesDesktop', { readFileDataUrl })

  render(
    <MemoryRouter>
      <ArtifactsView />
    </MemoryRouter>
  )

  expect(await screen.findByText('Artifact unavailable')).toBeTruthy()
  expect(readFileDataUrl).toHaveBeenCalledWith('/tmp/deleted-shot.png')
  expect(screen.queryByRole('img', { name: 'deleted-shot.png' })).toBeNull()
})
