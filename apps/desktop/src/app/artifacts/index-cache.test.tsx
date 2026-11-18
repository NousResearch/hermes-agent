import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { ArtifactsView } from './index'

const listAllProfileSessions = vi.fn(async () => ({
  sessions: [{ id: 'library-cache', title: 'Research', profile: 'default' }]
}))

const INDEXING = { name: 'Indexing recent session artifacts' }

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
  listAllProfileSessions.mockClear()
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

it('classic re-indexes on every visit, as on main', async () => {
  $shellChoice.set(false)
  mount()
  await screen.findByRole('button', { name: 'cached.md' })
  cleanup()

  mount()

  expect(screen.getByRole('status', INDEXING)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'cached.md' })).toBeNull()
  await screen.findByRole('button', { name: 'cached.md' })
  expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
})

it('classic shows a failed index as an empty library at once, without retrying', async () => {
  $shellChoice.set(false)
  listAllProfileSessions.mockRejectedValueOnce(new Error('gateway down'))
  mount()

  expect(await screen.findByText('No artifacts found')).toBeTruthy()
  expect(listAllProfileSessions).toHaveBeenCalledTimes(1)
})
