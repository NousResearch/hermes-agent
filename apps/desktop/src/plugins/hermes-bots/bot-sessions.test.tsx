/**
 * The per-bot session browser: opened from the bot row's context menu, lists
 * that bot's stored chats under the row, and opens one into the bot's
 * workspace — without ever becoming an identity for the canonical chat.
 *
 * Contract:
 * - The row's context menu offers "Show sessions"; selecting it lists the
 *   bot's profile sessions via `session.list` with include_hidden on the
 *   bot's own profile.
 * - Rows sort by last_active (newest first); the canonical "Bot Chat" row is
 *   labelled with the bot's name, not the plumbing title.
 * - Clicking a listed row opens THAT stored session in the `bots` workspace
 *   under the bot's owner key. The canonical registry path (openRosterBot)
 *   is never invoked by the browser.
 * - Only one bot's browser is open at a time.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BotRow } from './bot-row'
import { $botSessionsOpen, toggleBotSessions } from './bot-sessions'
import type * as CanonicalChat from './canonical-chat'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const { ensureBotMetadata, openRosterBot, openSession, request, requestProfile } = vi.hoisted(() => ({
  ensureBotMetadata: vi.fn(),
  openRosterBot: vi.fn(),
  openSession: vi.fn(),
  request: vi.fn(),
  requestProfile: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, openSession, request, requestProfile, warmProfile: vi.fn(), notifyError: vi.fn() },
    usePluginI18n: () => translateBots
  }
})

vi.mock('./canonical-chat', async importOriginal => {
  const actual = await importOriginal<typeof CanonicalChat>()

  return {
    ...actual,
    ensureBotMetadata,
    notifyBotOpenFailure: vi.fn(),
    openBotCanonicalChat: vi.fn(),
    prepareBotSource: vi.fn()
  }
})

vi.mock('./roster-actions', () => ({ openRosterBot }))

const noop = () => undefined

const SESSIONS = [
  { id: 'side-old', title: 'Old side thread', started_at: 1000, last_active: 1000, message_count: 3, hidden: false },
  { id: 'forever', title: 'Bot Chat', started_at: 500, last_active: 9000, message_count: 40, hidden: true },
  { id: 'side-new', title: 'Newer side thread', started_at: 2000, last_active: 5000, message_count: 2, hidden: false }
]

function Providers({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      {children}
    </QueryClientProvider>
  )
}

function renderRow(bot: RosterRow) {
  render(
    <Providers>
      <BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
    </Providers>
  )
}

const bot = { name: 'alpha', connectionId: 'local' } as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  $botSessionsOpen.set(null)
  ensureBotMetadata.mockResolvedValue({})
  openSession.mockResolvedValue(undefined)
  request.mockImplementation(async (method: string) => (method === 'session.list' ? { sessions: SESSIONS } : {}))
})

describe('bot session browser', () => {
  it('lists the bot profile sessions newest-first with the canonical row named after the bot', async () => {
    renderRow(bot)
    toggleBotSessions(bot)

    const region = await screen.findByRole('region')
    await waitFor(() => expect(region.querySelectorAll('[data-session-id]').length).toBe(3))

    expect(request).toHaveBeenCalledWith(
      'session.list',
      expect.objectContaining({ profile: 'alpha', include_hidden: true })
    )

    const ids = [...region.querySelectorAll('[data-session-id]')].map(el => el.getAttribute('data-session-id'))
    expect(ids).toEqual(['forever', 'side-new', 'side-old'])
    // Plumbing title never shows; the forever-chat reads as the bot.
    expect(region.textContent).not.toContain('Bot Chat')
    expect(region.textContent).toContain('Alpha')
  })

  it('opens a listed session in the bot workspace, not through the canonical registry', async () => {
    renderRow(bot)
    toggleBotSessions(bot)
    const region = await screen.findByRole('region')

    const row = await waitFor(() => {
      const el = region.querySelector('[data-session-id="side-new"]')
      expect(el).toBeTruthy()

      return el as HTMLElement
    })

    fireEvent.click(row)

    await waitFor(() => expect(openSession).toHaveBeenCalledTimes(1))
    expect(openSession).toHaveBeenCalledWith(
      'side-new',
      expect.objectContaining({
        profile: 'alpha',
        workspaceMode: 'bots',
        workspaceOwnerKey: expect.stringMatching(/^bot:/)
      })
    )
    expect(openRosterBot).not.toHaveBeenCalled()
  })

  it('renders nothing for a bot whose browser is not the open one', async () => {
    renderRow(bot)
    toggleBotSessions({ name: 'beta', connectionId: 'local' } as RosterRow)

    await new Promise(resolve => setTimeout(resolve, 20))
    expect(screen.queryByRole('region')).toBeNull()
    expect(request).not.toHaveBeenCalledWith('session.list', expect.anything())
  })
})
