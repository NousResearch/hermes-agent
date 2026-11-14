import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { _resetItemIndexForTests } from '@/store/item-index'
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
  window.localStorage.clear()
  _resetItemIndexForTests()
})

// Classic keeps main's words; the redesign's Library has its own (app/library/library-copy.ts).
const WORDS = {
  classic: {
    search: 'Search artifacts...',
    none: 'No artifacts found',
    empty: 'Generated images and file outputs will appear here as sessions produce them.'
  },
  redesign: {
    search: 'Search library',
    none: 'No matches',
    empty: 'Images, files and links from your recent chats show up here.'
  }
}

it.each([false, true])('keeps an indexed library recoverable after a search has no matches (V2=%s)', async enabled => {
  $shellChoice.set(enabled)
  render(
    <MemoryRouter>
      <ArtifactsView />
    </MemoryRouter>
  )
  const words = WORDS[enabled ? 'redesign' : 'classic']
  await screen.findByRole('button', { name: 'report.md' })

  const search = screen.getByRole('textbox', { name: words.search })
  fireEvent.change(search, { target: { value: 'unmatched-document' } })
  expect(await screen.findByText(words.none)).toBeTruthy()
  expect(screen.queryByText(words.empty)).toBeNull()

  fireEvent.change(search, { target: { value: '' } })
  expect(await screen.findByRole('button', { name: 'report.md' })).toBeTruthy()
})
