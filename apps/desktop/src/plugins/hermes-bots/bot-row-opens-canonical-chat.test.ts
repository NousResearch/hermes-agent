/**
 * A bot row click lands on the bot's canonical Bot Chat — the session the row
 * previews (`canonical_session`, resolved by name on every roster poll).
 *
 * The regression this pins: a plain click used to front whatever
 * bots-workspace tab the user last had open for that bot. A `+` side thread
 * persisted in Local Storage across restarts and won every click forever while
 * the row kept previewing the Bot Chat — sidebar and center described two
 * different conversations ("[Bots] - Sessions is not in sync again").
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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
const { $openBotChat, $selectedBot, $selectedRosterKey } = await import('./bot-state')
const { openRosterBot, trackInboundActivity } = await import('./roster-actions')
const { $selectedStoredSessionId } = await import('@/store/session')

const bot = { connectionId: 'local', name: 'alpha' } as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  prepareBotSource.mockResolvedValue(undefined)
  openBotCanonicalChat.mockResolvedValue({ openedId: 'bot-chat-tip', registryId: 'bot-chat' })
  $openBotChat.set(null)
  $selectedBot.set('')
  $selectedRosterKey.set('')
})

describe('a row click lands on the canonical chat, never a remembered side tab', () => {
  const canonicalBot = {
    ...bot,
    canonical_session: { id: 'bot-chat', resolved_id: 'bot-chat-tip' }
  } as RosterRow

  afterEach(() => {
    // @ts-expect-error — restore the harness default (no focus verb).
    delete host.focusOpenWorkspaceSession
  })

  it('fronts an open Bot Chat tab without a registry round-trip, side tabs excluded', async () => {
    const focus = vi.fn((_key: string, _probe: unknown, only?: readonly string[]) =>
      only?.includes('bot-chat-tip') ? 'bot-chat-tip' : null
    )

    host.focusOpenWorkspaceSession = focus as never

    await expect(openRosterBot(canonicalBot)).resolves.toBe(true)

    expect(focus).toHaveBeenCalledWith('bot:alpha', expect.any(Function), ['bot-chat', 'bot-chat-tip'])
    expect(openBotCanonicalChat).not.toHaveBeenCalled()
    expect($openBotChat.get()).toEqual({
      key: 'local::alpha',
      openedRegistryId: 'bot-chat',
      openedSessionId: 'bot-chat-tip'
    })
  })

  it('resolves the registry when only a side thread is open', async () => {
    // The shell would happily front 'side-thread' — the allowlist excludes it.
    host.focusOpenWorkspaceSession = vi.fn((_key: string, _probe: unknown, only?: readonly string[]) =>
      only?.includes('side-thread') ? 'side-thread' : null
    ) as never

    await expect(openRosterBot(canonicalBot)).resolves.toBe(true)

    expect(openBotCanonicalChat).toHaveBeenCalledWith(canonicalBot, expect.any(Function), expect.any(Function))
    expect($openBotChat.get()?.openedSessionId).toBe('bot-chat-tip')
  })

  it('commits the roster owner only after its canonical chat owns the center', async () => {
    let releaseSource!: () => void

    const sourceReady = new Promise<void>(resolve => {
      releaseSource = resolve
    })

    prepareBotSource.mockReturnValueOnce(sourceReady)
    $selectedBot.set('omega')
    $selectedRosterKey.set('local::omega')

    const opening = openRosterBot(canonicalBot)

    await vi.waitFor(() => expect(prepareBotSource).toHaveBeenCalledOnce())
    expect($selectedBot.get()).toBe('omega')
    expect($selectedRosterKey.get()).toBe('local::omega')

    releaseSource()
    await expect(opening).resolves.toBe(true)
    expect($selectedBot.get()).toBe('alpha')
    expect($selectedRosterKey.get()).toBe('local::alpha')
  })

  it('a failed open records no claim', async () => {
    openBotCanonicalChat.mockRejectedValueOnce(new Error('gateway away'))
    $selectedBot.set('omega')
    $selectedRosterKey.set('local::omega')

    await expect(openRosterBot(bot)).resolves.toBe(false)

    expect($openBotChat.get()).toBeNull()
    expect($selectedBot.get()).toBe('omega')
    expect($selectedRosterKey.get()).toBe('local::omega')
  })
})

describe('the open Bot Chat follows its session on the gateway', () => {
  // The roster poll is the only signal for turns that never reach this
  // window's stream (cron bot-chat deliveries, message_agent, group rounds).
  // When the FOCUSED chat's canonical session moves, it re-resolves so the pane
  // repaints from the gateway instead of waiting for a restart (#99393).
  const activeBot = (lastActive: number) =>
    ({
      connectionId: 'local',
      name: 'alpha',
      canonical_session: { id: 'bot-chat', resolved_id: 'bot-chat-tip', last_active: lastActive }
    }) as RosterRow

  it('re-opens the focused Bot Chat when its canonical session advances', () => {
    $selectedBot.set('alpha')
    $selectedStoredSessionId.set('bot-chat-tip')
    trackInboundActivity([activeBot(100)]) // seeds the watermark

    trackInboundActivity([activeBot(200)])

    expect(openBotCanonicalChat).toHaveBeenCalledTimes(1)
    $selectedStoredSessionId.set(null)
  })

  it('leaves the center alone when the Bot Chat is not what is focused', () => {
    $selectedBot.set('alpha')
    $selectedStoredSessionId.set('some-group-room')
    trackInboundActivity([activeBot(300)])

    trackInboundActivity([activeBot(400)])

    expect(openBotCanonicalChat).not.toHaveBeenCalled()
    $selectedStoredSessionId.set(null)
  })
})
