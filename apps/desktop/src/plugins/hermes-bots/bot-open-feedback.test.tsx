/** Regression for #120277: acknowledge a cold bot click before the gateway answers. */
import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { BotRow } from './bot-row'
import { $openBotChat } from './bot-state'
import { $groupChats, $groupChatWorkspace } from './group-chat'
import { openGroupChat } from './group-chat-view'
import { translateBotsIn } from './i18n-test-helper'
import { bumpBotOpenGeneration } from './shared'
import type { RosterRow } from './types'

const { requestProfile, openSession, focusOpenWorkspaceSession, notify } = vi.hoisted(() => ({
  requestProfile: vi.fn(),
  openSession: vi.fn(),
  focusOpenWorkspaceSession: vi.fn(),
  notify: vi.fn()
}))

// Only the host I/O boundary is scripted; row, open action, registry lookup,
// generation guards, state subscriptions, and loading primitive are real.
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, requestProfile, openSession, focusOpenWorkspaceSession, notify },
    usePluginI18n: () => translateBotsIn('en')
  }
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

const noop = () => undefined

const bot: RosterRow = {
  name: 'alpha',
  connectionId: 'local',
  sourceScoped: true,
  canonical_session: { id: 'alpha-chat', title: 'Bot Chat' }
}

const twin: RosterRow = { ...bot, connectionId: 'remote', remoteSource: true }
const registry = { sessions: [{ id: 'alpha-chat', title: 'Bot Chat', message_count: 2 }] }

function renderRows() {
  render(
    <>
      {[bot, twin].map(row => (
        <BotRow bot={row} key={row.connectionId} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
      ))}
    </>
  )

  return screen.getAllByRole('button')
}

beforeEach(() => {
  vi.clearAllMocks()
  bumpBotOpenGeneration()
  $groupChatWorkspace.set(null)
  $groupChats.set({})
  focusOpenWorkspaceSession.mockReturnValue(null)
  $openBotChat.set({ key: 'local::previous', openedRegistryId: 'previous-chat' })
})
afterEach(cleanup)

it('shows exact-target feedback throughout lookup and hydration, coalescing repeated clicks', async () => {
  const lookup = deferred<typeof registry>()
  const hydration = deferred<void>()
  requestProfile.mockReturnValue(lookup.promise)
  openSession.mockReturnValue(hydration.promise)
  const [row, other] = renderRows()

  fireEvent.click(row)

  // Synchronous acknowledgement, before even one network reply.
  expect(row.getAttribute('aria-busy')).toBe('true')
  expect(within(row).getByText('Opening chat…')).toBeTruthy()
  expect(other.getAttribute('aria-busy')).not.toBe('true')
  expect($openBotChat.get()?.openedRegistryId).toBe('previous-chat')
  expect(openSession).not.toHaveBeenCalled()

  fireEvent.click(row)
  await act(async () => {
    await Promise.resolve()
  })
  expect(requestProfile).toHaveBeenCalledTimes(1)
  expect(requestProfile.mock.calls[0][0]).toMatchObject({ connectionId: 'local', profile: 'alpha' })
  expect(requestProfile.mock.calls[0][1]).toBe('session.list')

  await act(async () => {
    lookup.resolve(registry)
  })
  expect(openSession).toHaveBeenCalledTimes(1)
  expect(openSession).toHaveBeenCalledWith(
    'alpha-chat',
    expect.objectContaining({
      profile: 'alpha',
      awaitHydration: true,
      route: expect.objectContaining({ connectionId: 'local' })
    })
  )
  expect(row.getAttribute('aria-busy')).toBe('true')

  // Hydration mounted a canonical tile; the user then focused a side chat.
  // A new click must front that tile even while it joins the same I/O flight.
  focusOpenWorkspaceSession.mockClear()
  focusOpenWorkspaceSession.mockReturnValue('alpha-chat')
  fireEvent.click(row)
  expect(focusOpenWorkspaceSession).toHaveBeenCalledWith('bot:local::alpha', expect.any(Function), ['alpha-chat'])
  expect(requestProfile).toHaveBeenCalledTimes(1)
  expect(openSession).toHaveBeenCalledTimes(1)

  await act(async () => {
    hydration.resolve()
  })
  expect(row.getAttribute('aria-busy')).not.toBe('true')
  expect(within(row).queryByText('Opening chat…')).toBeNull()
  expect($openBotChat.get()?.openedRegistryId).toBe('alpha-chat')
})

it('only the latest intent owns feedback; failure, retry and group navigation release it', async () => {
  const oldLookup = deferred<typeof registry>()
  const latestLookup = deferred<typeof registry>()
  requestProfile.mockReturnValueOnce(oldLookup.promise).mockReturnValueOnce(latestLookup.promise)
  openSession.mockResolvedValue(undefined)
  const [row, other] = renderRows()

  fireEvent.click(row)
  await act(async () => {
    await Promise.resolve()
  })
  fireEvent.click(other)
  await act(async () => {
    await Promise.resolve()
  })
  expect(row.getAttribute('aria-busy')).not.toBe('true')
  expect(other.getAttribute('aria-busy')).toBe('true')

  // An older lookup may finish, but cannot front a chat or clear the new row.
  await act(async () => {
    oldLookup.resolve(registry)
  })
  expect(openSession).not.toHaveBeenCalled()
  expect(other.getAttribute('aria-busy')).toBe('true')

  await act(async () => {
    latestLookup.reject(new Error('gateway unavailable'))
  })
  expect(other.getAttribute('aria-busy')).not.toBe('true')
  expect(notify).toHaveBeenCalledTimes(1)
  expect(openSession).not.toHaveBeenCalled()

  requestProfile.mockResolvedValue(registry)
  await act(async () => {
    fireEvent.click(other)
  })
  expect(other.getAttribute('aria-busy')).not.toBe('true')
  expect($openBotChat.get()?.key).toBe('remote::alpha')
  expect(openSession.mock.calls[0][1].route.connectionId).toBe('remote')

  // Warm canonical tabs keep their immediate fronting path.
  cleanup()
  focusOpenWorkspaceSession.mockReturnValue('alpha-chat')
  render(
    <BotRow
      bot={{ ...bot, canonical_session: registry.sessions[0] }}
      onDelete={noop}
      onEdit={noop}
      onGroup={noop}
      onNewSection={noop}
    />
  )
  const warmRow = screen.getByRole('button')
  const calls = requestProfile.mock.calls.length
  await act(async () => {
    fireEvent.click(warmRow)
  })
  expect(warmRow.getAttribute('aria-busy')).not.toBe('true')
  expect(requestProfile).toHaveBeenCalledTimes(calls)
  expect($openBotChat.get()?.key).toBe('local::alpha')

  // Choosing a group cancels feedback synchronously, not when the RPC ends.
  focusOpenWorkspaceSession.mockReturnValue(null)
  const cancelled = deferred<typeof registry>()
  requestProfile.mockReturnValue(cancelled.promise)
  fireEvent.click(warmRow)
  await act(async () => {
    await Promise.resolve()
  })
  expect(warmRow.getAttribute('aria-busy')).toBe('true')
  const opens = openSession.mock.calls.length
  act(() => {
    $groupChats.set({ Team: { log: [], sessions: {}, watermarks: {} } })
    openGroupChat('Team')
  })
  expect(warmRow.getAttribute('aria-busy')).not.toBe('true')
  await act(async () => {
    cancelled.resolve(registry)
  })
  expect(openSession).toHaveBeenCalledTimes(opens)
  expect($groupChatWorkspace.get()).toBe('Team')
})
