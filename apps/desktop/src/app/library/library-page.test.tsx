import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { _resetItemIndexForTests } from '@/store/item-index'
import { $shellChoice } from '@/store/shell-variant'

import { LibraryPage } from './library-page'

const listAllProfileSessions = vi.hoisted(() => vi.fn())
const getSessionMessages = vi.hoisted(() => vi.fn())
const openSessionFromPicker = vi.hoisted(() => vi.fn())
const setSessionOwnerHint = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/hermes')),
  listAllProfileSessions,
  getSessionMessages
}))
vi.mock('../open-session', () => ({ openSessionFromPicker }))
vi.mock('@/store/session', async () => ({ ...(await vi.importActual('@/store/session')), setSessionOwnerHint }))

const IMAGES = 30

beforeEach(() => {
  $shellChoice.set(true)
  listAllProfileSessions.mockResolvedValue({
    sessions: [
      { id: 'art', title: 'Moodboard', profile: 'design', connection_id: 'studio', last_active: 10, message_count: 4 }
    ]
  })
  getSessionMessages.mockResolvedValue({
    session_id: 'art',
    messages: [
      ...Array.from({ length: IMAGES }, (_, index) => ({
        role: 'assistant',
        timestamp: 1000 + index,
        content: `MEDIA:/out/shot-${String(index).padStart(2, '0')}.png`
      })),
      { role: 'assistant', timestamp: 2000, content: 'MEDIA:/out/brief.md and https://example.com/guide' }
    ]
  })
  vi.stubGlobal('hermesDesktop', {
    saveGatewayFile: vi.fn().mockResolvedValue({ saved: true }),
    api: vi.fn().mockResolvedValue('data:image/png;base64,AA==')
  })
})

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  window.localStorage.clear()
  _resetItemIndexForTests()
})

const mount = (path = '/') =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <LibraryPage />
    </MemoryRouter>
  )

it('counts each kind in the panel and pages the images 24 at a time', async () => {
  mount('/?tab=image')

  const nav = await screen.findByRole('complementary', { name: 'Library' })
  await waitFor(() => expect(within(nav).getByRole('button', { name: /Images\s*30/ })).toBeTruthy())
  expect(within(nav).getByRole('button', { name: /All\s*32/ })).toBeTruthy()
  expect(screen.getByText('1–24 of 30 images')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Page 2' }))
  expect(screen.getByText('25–30 of 30 images')).toBeTruthy()
  expect(screen.getAllByRole('button', { name: /^shot-\d+\.png/ })).toHaveLength(6)
})

it('opens an item’s chat on the machine and profile that own it', async () => {
  mount('/?tab=file')

  fireEvent.click(await screen.findByRole('button', { name: /Moodboard/ }))

  expect(setSessionOwnerHint).toHaveBeenCalledWith('art', { connectionId: 'studio', profile: 'design' })
  expect(openSessionFromPicker).toHaveBeenCalledWith('art', expect.any(Function))
})

it('steps through an image preview with the arrows, enlarges it, and Escape closes one layer at a time', async () => {
  mount('/?tab=image')

  fireEvent.click(await screen.findByRole('button', { name: /^shot-29\.png/ }))
  expect(screen.getByText('1 of 30')).toBeTruthy()

  fireEvent.keyDown(screen.getByText('1 of 30'), { key: 'ArrowRight' })
  expect(screen.getByRole('heading', { name: 'shot-28.png' })).toBeTruthy()

  const enlarge = await screen.findByRole('button', { name: 'Open full size' })
  enlarge.focus()
  fireEvent.click(enlarge)
  const dialog = await screen.findByRole('dialog', { name: 'shot-28.png' })

  fireEvent.click(within(dialog).getByRole('button', { name: 'Download' }))
  await waitFor(() =>
    expect(window.hermesDesktop.saveGatewayFile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'studio', path: '/out/shot-28.png', profile: 'design', sessionId: 'art' })
    )
  )

  fireEvent.keyDown(dialog, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Open full size' })))

  fireEvent.keyDown(screen.getByRole('button', { name: 'Open full size' }), { key: 'Escape' })
  expect(screen.queryByText(/of 30$/)).toBeNull()
  expect(screen.getByText('1–24 of 30 images')).toBeTruthy()
})

it('says when the library could not be checked, and Try again checks again', async () => {
  listAllProfileSessions.mockRejectedValueOnce(new Error('offline'))
  mount()

  const alert = await screen.findByRole('alert')
  expect(within(alert).getByText('The library could not load')).toBeTruthy()

  await act(async () => fireEvent.click(within(alert).getByRole('button', { name: 'Try again' })))

  await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  expect(listAllProfileSessions).toHaveBeenCalledTimes(2)
  expect(await screen.findByRole('button', { name: 'brief.md' })).toBeTruthy()
})
