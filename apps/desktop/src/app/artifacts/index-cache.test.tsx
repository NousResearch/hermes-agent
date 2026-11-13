import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { ArtifactsView } from './index'

const listAllProfileSessions = vi.fn(async () => ({
  sessions: [{ id: 'library-cache', title: 'Research', profile: 'default' }]
}))

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: () => listAllProfileSessions(),
  getSessionMessages: async () => ({
    session_id: 'library-cache',
    messages: [{ role: 'assistant', timestamp: 1000, content: 'MEDIA:/workspace/cached.md' }]
  })
}))

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

const mount = () =>
  render(
    <MemoryRouter>
      <ArtifactsView />
    </MemoryRouter>
  )

it('reopens the Library on its last index in the first paint, with no indexing state', async () => {
  $shellChoice.set(true)
  mount()
  await screen.findByRole('button', { name: 'cached.md' })
  cleanup()

  mount()

  // Synchronous: the reopened page has its rows before any fetch settles.
  expect(screen.getByRole('button', { name: 'cached.md' })).toBeTruthy()
  expect(screen.queryByRole('status', { name: /index|loading/i })).toBeNull()
})
