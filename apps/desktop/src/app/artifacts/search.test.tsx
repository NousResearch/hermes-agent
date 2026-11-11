import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { ArtifactsView } from './index'

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions: async () => ({ sessions: [{ id: 'library-test', title: 'Research', profile: 'default' }] }),
  getSessionMessages: async () => ({
    session_id: 'library-test',
    messages: [{ role: 'assistant', timestamp: 1000, content: 'MEDIA:/workspace/report.md' }]
  })
}))

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

it.each([false, true])('keeps an indexed library recoverable after a search has no matches (V2=%s)', async enabled => {
  $shellChoice.set(enabled)
  render(
    <MemoryRouter>
      <ArtifactsView />
    </MemoryRouter>
  )
  await screen.findByRole('button', { name: 'report.md' })

  const search = screen.getByRole('textbox', { name: 'Search artifacts...' })
  fireEvent.change(search, { target: { value: 'unmatched-document' } })
  expect(await screen.findByText('No artifacts found')).toBeTruthy()
  expect(screen.queryByText('Generated images and file outputs will appear here as sessions produce them.')).toBeNull()

  fireEvent.change(search, { target: { value: '' } })
  expect(await screen.findByRole('button', { name: 'report.md' })).toBeTruthy()
})
