/**
 * Cold bot switch acknowledges the clicked row before the backend answers
 * (hermes-agent#120277).
 *
 * The mark is published only after the fronted-tab check misses, and before
 * prepareBotSource. It is not chat ownership: highlight, routing, drafts, and
 * running turns stay on their existing paths.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { openBotCanonicalChat, prepareBotSource } = vi.hoisted(() => ({
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn()
}))

vi.mock('./canonical-chat', () => ({
  CANONICAL_CHAT_TITLE: 'Bot Chat',
  ensureBotMetadata: vi.fn(async () => ({})),
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat,
  prepareBotSource,
  PROFILE_SESSION_LIST_LIMIT: 200
}))

const { host } = await import('@hermes/plugin-sdk')
const { $openBotChat, $pendingBotOpen } = await import('./bot-state')
const { $groupChats, $groupChatWorkspace } = await import('./group-chat')
const { openGroupChat } = await import('./group-chat-view')
const { bumpBotOpenGeneration } = await import('./shared')
const { openRosterBot } = await import('./roster-actions')

const botB = {
  connectionId: 'local',
  name: 'bravo',
  canonical_session: { id: 'b-chat', resolved_id: 'b-tip' }
} as RosterRow

const botA = {
  connectionId: 'local',
  name: 'alpha',
  canonical_session: { id: 'a-chat', resolved_id: 'a-tip' }
} as RosterRow

function deferred<T = void>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

beforeEach(() => {
  vi.clearAllMocks()
  $openBotChat.set(null)
  $pendingBotOpen.set(null)
  $groupChats.set({})
  $groupChatWorkspace.set(null)
  // @ts-expect-error — restore the harness default (no focus verb).
  delete host.focusOpenWorkspaceSession
})

describe('cold bot switch publishes its target after the fronted-tab miss', () => {
  it('marks the target before prepareBotSource, and not when a tab is already fronted', async () => {
    const seenAtPrepare: Array<string | null> = []

    prepareBotSource.mockImplementation(() => {
      seenAtPrepare.push($pendingBotOpen.get()?.key ?? null)

      return Promise.resolve()
    })
    openBotCanonicalChat.mockResolvedValue({ openedId: 'b-tip', registryId: 'b-chat' })

    host.focusOpenWorkspaceSession = vi.fn(() => 'a-tip') as never
    await openRosterBot(botA)

    expect(prepareBotSource).not.toHaveBeenCalled()
    expect($pendingBotOpen.get()).toBeNull()

    // @ts-expect-error — cold path: the shell cannot front a tab.
    delete host.focusOpenWorkspaceSession
    const flight = openRosterBot(botB)

    expect($pendingBotOpen.get()?.key).toBe('local::bravo')
    await flight
    expect(seenAtPrepare).toEqual(['local::bravo'])
    expect($pendingBotOpen.get()).toBeNull()
    expect($openBotChat.get()?.openedSessionId).toBe('b-tip')
  })

  it('clears the mark when source prep fails, and a retry marks and clears again', async () => {
    const gate = deferred()

    prepareBotSource.mockReturnValueOnce(gate.promise)
    const flight = openRosterBot(botB)

    expect($pendingBotOpen.get()?.key).toBe('local::bravo')
    gate.reject(new Error('backend away'))
    await expect(flight).resolves.toBe(false)
    expect($pendingBotOpen.get()).toBeNull()

    const retryGate = deferred()

    prepareBotSource.mockReturnValueOnce(retryGate.promise)
    openBotCanonicalChat.mockResolvedValueOnce({ openedId: 'b-tip', registryId: 'b-chat' })
    const retry = openRosterBot(botB)

    expect($pendingBotOpen.get()?.key).toBe('local::bravo')
    retryGate.resolve()
    await expect(retry).resolves.toBe(true)
    expect($pendingBotOpen.get()).toBeNull()
  })

  it('opening a group drops the mark at once, and the late flight cannot reclaim it', async () => {
    const gate = deferred()

    prepareBotSource.mockReturnValueOnce(gate.promise)
    const flight = openRosterBot(botB)

    expect($pendingBotOpen.get()?.key).toBe('local::bravo')
    $groupChats.set({ Team: { log: [], sessions: {}, watermarks: {} } })
    openGroupChat('Team')
    expect($pendingBotOpen.get()).toBeNull()

    gate.resolve()
    await expect(flight).resolves.toBe(false)
    expect($pendingBotOpen.get()).toBeNull()
    expect(openBotCanonicalChat).not.toHaveBeenCalled()
    expect($groupChatWorkspace.get()).toBe('Team')
  })

  it('a superseded flight never clears its successor, and an external supersede drops the mark', async () => {
    const gateA = deferred()
    const gateB = deferred()

    prepareBotSource.mockReturnValueOnce(gateA.promise).mockReturnValueOnce(gateB.promise)
    openBotCanonicalChat.mockResolvedValue({ openedId: 'b-tip', registryId: 'b-chat' })

    const flightA = openRosterBot(botA)

    expect($pendingBotOpen.get()?.key).toBe('local::alpha')

    const flightB = openRosterBot(botB)

    expect($pendingBotOpen.get()?.key).toBe('local::bravo')
    gateA.resolve()
    await flightA
    expect($pendingBotOpen.get()?.key).toBe('local::bravo')

    bumpBotOpenGeneration()
    expect($pendingBotOpen.get()).toBeNull()
    gateB.resolve()
    await flightB
    expect($pendingBotOpen.get()).toBeNull()
    expect($openBotChat.get()?.key).not.toBe('local::bravo')
  })
})
