/**
 * A bot's OTHER conversations, listed under its own row (#112184).
 *
 * Discovery and navigation only: the list reads the sessions the bot's exact
 * profile/source already holds (host.listPersistedSessions) and opens them
 * (host.openSession) — no new storage, no id pointer, no routing. The four
 * contracts pinned here are the whole feature: isolation between bots, the
 * canonical Bot Chat's untouched identity, the collapsed default, and rows
 * that arrive without a title.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $expandedBotSessions } from './bot-session-list'
import { BotRow } from './bot-row'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const { listPersistedSessions, openRosterBot, openSession } = vi.hoisted(() => ({
  listPersistedSessions: vi.fn(),
  openRosterBot: vi.fn(),
  openSession: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, listPersistedSessions, openSession },
    usePluginI18n: () => translateBots
  }
})

vi.mock('./canonical-chat', () => ({
  CANONICAL_CHAT_TITLE: 'Bot Chat',
  ensureBotMetadata: vi.fn(async () => ({})),
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn(),
  PROFILE_SESSION_LIST_LIMIT: 200
}))

vi.mock('./roster-actions', () => ({ openRosterBot }))

/** Sessions per BACKEND profile — the key host.listPersistedSessions is
 *  actually asked for, so a bot whose logical name differs from its backend
 *  target cannot pass this suite by accident. */
const SESSIONS_BY_PROFILE: Record<string, Array<Record<string, unknown>>> = {
  alpha: [
    { id: 'a-office', last_active: 2_000, title: 'Office Operations' },
    { id: 'a-sales', last_active: 1_000, title: 'Sales & Outreach' },
    // A canonical row that a windowed/not-yet-hidden listing could still
    // report: it must never appear as a conversation under the row.
    { id: 'a-bot-chat', last_active: 3_000, title: 'Bot Chat' }
  ],
  beta: [{ id: 'b-research', last_active: 500, title: 'Beta Research' }],
  gamma: [],
  delta: [
    { id: 'd-untitled', last_active: 10, title: null },
    { id: 'd-blank', last_active: 5, title: '   ' }
  ]
}

const alphaBot = () => ({ connectionId: 'local', name: 'alpha' }) as RosterRow

/** Same gateway, different profile — the isolating pair for the alias case. */
const betaBot = () =>
  ({
    connectionId: 'remote-a',
    name: 'beta',
    remoteSource: true,
    route: { connectionId: 'remote-a', mode: 'remote', profile: 'beta', targetProfile: 'beta' },
    sourceScoped: true
  }) as RosterRow

function renderRow(bot: RosterRow) {
  const { container } = render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

  // The row's subtree also holds its conversations caret, so locate the row
  // itself by the attribute every click path already keys off.
  return container.querySelector<HTMLElement>('[data-roster-key]')!
}

const noop = () => undefined

function disclosure(bot: RosterRow) {
  return screen.getByRole('button', { name: new RegExp(`conversations with ${bot.name}`, 'i') })
}

function listedIds(container: HTMLElement) {
  return [...container.querySelectorAll('[data-bot-session-id]')].map(node => node.getAttribute('data-bot-session-id'))
}

beforeEach(() => {
  vi.clearAllMocks()
  $expandedBotSessions.set(new Set())
  openRosterBot.mockResolvedValue(true)
  openSession.mockResolvedValue(undefined)
  listPersistedSessions.mockImplementation(async (_route: unknown, options: { profile: string }) => ({
    limit: 200,
    offset: 0,
    sessions: SESSIONS_BY_PROFILE[options.profile] || [],
    total: (SESSIONS_BY_PROFILE[options.profile] || []).length
  }))
})

describe('a bot lists only its own profile’s conversations', () => {
  it('lists the expanded bot’s sessions and never another bot’s', async () => {
    const { container } = render(
      <>
        <BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
        <BotRow bot={betaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
      </>
    )

    fireEvent.click(disclosure(alphaBot()))

    await screen.findByText('Office Operations')
    expect(screen.getByText('Sales & Outreach')).toBeDefined()
    expect(screen.queryByText('Beta Research')).toBeNull()
    // Only the expanded owner is ever read.
    expect(listPersistedSessions.mock.calls.map(([, options]) => options.profile)).toEqual(['alpha'])
    expect(listedIds(container)).toEqual(['a-office', 'a-sales'])
  })

  it('reads them from the bot’s own source and target profile', async () => {
    render(<BotRow bot={betaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(betaBot()))

    await screen.findByText('Beta Research')
    expect(listPersistedSessions.mock.calls).toEqual([[betaBot().route, { limit: 200, profile: 'beta' }]])
  })

  it('opens a listed conversation through the existing open path', async () => {
    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))
    fireEvent.click(await screen.findByText('Sales & Outreach'))

    expect(openSession).toHaveBeenCalledWith('a-sales', {
      intent: 'in-place',
      keepAllProfilesScope: true,
      profile: 'alpha',
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:alpha'
    })
    // Navigation only: the list never touches the canonical open path.
    expect(openRosterBot).not.toHaveBeenCalled()
  })
})

describe('the canonical Bot Chat keeps its identity', () => {
  it('stays the row’s own click target, above the list and never a list entry', async () => {
    const bot = alphaBot()
    const { container } = render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)
    const row = container.querySelector<HTMLElement>('[data-roster-key]')!

    expect(screen.queryByText('Bot Chat')).toBeNull()

    fireEvent.click(disclosure(bot))

    await screen.findByText('Office Operations')
    // Still absent with the list open, even though the listing reported it.
    expect(screen.queryByText('Bot Chat')).toBeNull()
    expect(listedIds(container)).not.toContain('a-bot-chat')
    // The row still opens the forever-chat, and it is still first.
    fireEvent.click(row)
    expect(openRosterBot).toHaveBeenCalledWith(bot)
    expect(row.compareDocumentPosition(container.querySelector('[data-bot-sessions]')!) & 4).toBe(4)
  })
})

describe('the list is collapsed until it is asked for', () => {
  it('reads nothing on paint and leaves the row’s click untouched', () => {
    const bot = alphaBot()
    const { container } = render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    expect(listPersistedSessions).not.toHaveBeenCalled()
    expect(listedIds(container)).toEqual([])
    expect(container.querySelector('[data-bot-sessions]')).toBeNull()
    expect(disclosure(bot).getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(container.querySelector<HTMLElement>('[data-roster-key]')!)

    expect(openRosterBot).toHaveBeenCalledWith(bot)
  })
})

describe('sessions that arrive without a title', () => {
  it('renders a placeholder instead of crashing on a null or blank title', async () => {
    render(
      <BotRow
        bot={{ connectionId: 'local', name: 'delta' } as RosterRow}
        onDelete={noop}
        onEdit={noop}
        onGroup={noop}
        onNewSection={noop}
      />
    )

    fireEvent.click(disclosure({ name: 'delta' } as RosterRow))

    expect(await screen.findAllByText('(untitled)')).toHaveLength(2)
  })

  it('shows an empty note when the profile has no other conversations', async () => {
    render(
      <BotRow
        bot={{ connectionId: 'local', name: 'gamma' } as RosterRow}
        onDelete={noop}
        onEdit={noop}
        onGroup={noop}
        onNewSection={noop}
      />
    )

    fireEvent.click(disclosure({ name: 'gamma' } as RosterRow))

    expect(await screen.findByText(/no other conversations/i)).toBeDefined()
    expect(screen.queryByText('(untitled)')).toBeNull()
  })

  it('reports a failed read instead of throwing', async () => {
    listPersistedSessions.mockRejectedValueOnce(new Error('source unreachable'))

    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))

    await waitFor(() => expect(screen.getByText(/could not load conversations/i)).toBeDefined())
  })
})
