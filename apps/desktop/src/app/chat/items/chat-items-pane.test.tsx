import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $routineSelection } from '@/app/routines/routines-store'
import type { CronJob } from '@/hermes'
import { $artifactRegistry, upsertArtifact } from '@/store/artifacts'
import { $chatItemsOpen } from '@/store/chat-items'
import { beginCronJobsRequest, commitCronJobsRequest, invalidateCronJobsRequests } from '@/store/cron'
import { setInterfaceMode } from '@/store/interface-mode'
import { _resetItemIndexForTests } from '@/store/item-index'
import { setSelectedStoredSessionId, setSessions } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'
import type { SessionInfo } from '@/types/hermes'

import { useChatOwner } from '../use-chat-owner'

import { ChatItemsButton } from './chat-items-button'
import { ChatItemsPane } from './chat-items-pane'

const getSessionMessages = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async () => ({ ...(await vi.importActual('@/hermes')), getSessionMessages }))
vi.mock('@/app/chat/right-rail/preview-artifact', () => ({
  ArtifactPreview: ({ target }: { target: { label: string } }) => <p>Preview of {target.label}</p>
}))

const row = (id: string, title: string) =>
  ({ id, title, profile: 'default', last_active: 50, message_count: 3, started_at: 1 }) as SessionInfo

const PAGE = `<!doctype html><html><head><title>Status page</title></head><body>${'<p>Week in review.</p>'.repeat(12)}</body></html>`

beforeEach(() => {
  // jsdom has no layout: the tab bar scrolls a tapped tab into view.
  Element.prototype.scrollIntoView = vi.fn()
  $shellChoice.set(true)
  setInterfaceMode('advanced')
  setSessions([row('busy', 'Launch plan'), row('quiet', 'Small talk')])
  getSessionMessages.mockImplementation(async (id: string) => ({
    session_id: id,
    messages:
      id === 'busy'
        ? [
            { role: 'assistant', timestamp: 10, content: 'MEDIA:/out/plan.md and https://example.com/brief' },
            { role: 'tool', tool_name: 'image_generate', timestamp: 11, content: '{"image_path": "/out/cover.png"}' }
          ]
        : [{ role: 'assistant', timestamp: 10, content: 'Hello there.' }]
  }))
  upsertArtifact('busy', { kind: 'html', language: 'html', title: 'Status page' }, PAGE)
})

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  $artifactRegistry.set({})
  $chatItemsOpen.set(false)
  commitCronJobsRequest(beginCronJobsRequest('\u0000all'), [])
  invalidateCronJobsRequests()
  $routineSelection.set(null)
  setSelectedStoredSessionId(null)
  setSessions([])
  setInterfaceMode('advanced')
  window.localStorage.clear()
  _resetItemIndexForTests()
  vi.clearAllMocks()
})

it('lists the focused chat’s artifacts, files and links from the shared index, by tab', async () => {
  setSelectedStoredSessionId('busy')
  render(<ChatItemsPane />)

  const tabs = await screen.findByRole('tablist', { name: 'Chat items' })
  await waitFor(() => expect(within(tabs).getByRole('tab', { name: /All\s*4/ })).toBeTruthy())
  expect(screen.getByRole('button', { name: /Status page/ })).toBeTruthy()
  expect(screen.getByRole('button', { name: /plan\.md/ })).toBeTruthy()
  expect(screen.getByRole('link', { name: /example\.com/ })).toBeTruthy()

  fireEvent.click(within(tabs).getByRole('tab', { name: /Web/ }))
  expect(screen.queryByRole('button', { name: /plan\.md/ })).toBeNull()
  expect(screen.getByRole('link', { name: /example\.com/ })).toBeTruthy()
  // The chat was read once, through the index the Library shares.
  expect(getSessionMessages).toHaveBeenCalledTimes(1)
})

it('opens an artifact inside the pane and goes back to the list', async () => {
  setSelectedStoredSessionId('busy')
  render(<ChatItemsPane />)

  fireEvent.click(await screen.findByRole('button', { name: /Status page/ }))
  expect(screen.getByText('Preview of Status page')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'All items' }))
  expect(await screen.findByRole('tablist', { name: 'Chat items' })).toBeTruthy()
})

it('says so when the chat has nothing yet', async () => {
  setSelectedStoredSessionId('quiet')
  render(<ChatItemsPane />)

  await waitFor(() => expect(getSessionMessages).toHaveBeenCalledWith('quiet', 'default', expect.anything()))
  expect(screen.getByText('Nothing here yet')).toBeTruthy()
})

it('lists the routines of the chat’s agent first, and opens one on the Routines page', async () => {
  const routine = (id: string, name: string, profile: string) =>
    ({ enabled: true, id, name, profile, schedule: { kind: 'interval', minutes: 60 } }) as CronJob

  commitCronJobsRequest(beginCronJobsRequest('\u0000all'), [
    routine('brief', 'Morning brief', 'default'),
    routine('other', 'Someone else’s', 'ops')
  ])
  setSelectedStoredSessionId('busy')
  render(
    <MemoryRouter>
      <ChatItemsPane />
    </MemoryRouter>
  )

  const tabs = await screen.findByRole('tablist', { name: 'Chat items' })
  await waitFor(() => expect(within(tabs).getByRole('tab', { name: /All\s*5/ })).toBeTruthy())
  expect(within(tabs).getAllByRole('tab')[1].textContent).toMatch(/1$/)
  expect(screen.queryByRole('button', { name: /Someone else/ })).toBeNull()

  fireEvent.click(screen.getByRole('button', { name: /Morning brief/ }))
  expect($routineSelection.get()).toEqual({ id: 'brief', profile: 'default' })
})

function SimpleButton({ sessionId }: { sessionId: string }) {
  return <ChatItemsButton chat={useChatOwner(sessionId)} sessionId={sessionId} />
}

it('offers the items in simple mode only, with the count, and opens them on click', async () => {
  setInterfaceMode('simple')
  render(<SimpleButton sessionId="busy" />)

  const button = await screen.findByRole('button', { name: '4 items' })
  expect($chatItemsOpen.get()).toBe(false)

  await act(async () => fireEvent.click(button))
  expect($chatItemsOpen.get()).toBe(true)

  cleanup()
  setInterfaceMode('advanced')
  render(<SimpleButton sessionId="busy" />)
  expect(screen.queryByRole('button', { name: /items/ })).toBeNull()
})
