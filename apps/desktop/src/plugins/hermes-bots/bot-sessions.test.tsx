/**
 * BotSessionsList shows each bot's sessions underneath its row, pulling data
 * via `session.list` and opening them through `host.openSession`. A new-session
 * affordance calls `newBotChat`.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BotSessionsList } from './bot-sessions'
import type { RosterRow } from './types'

const { hostMock, requestForBotMock, newBotChatMock } = vi.hoisted(() => ({
  hostMock: {
    openSession: vi.fn(),
    state: {
      i18n: {
        get: () => ({ t: (k: string) => k })
      }
    }
  },
  requestForBotMock: vi.fn(),
  newBotChatMock: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { useQuery } = await import('@tanstack/react-query')

  return {
    cn: (...args: unknown[]) => args.join(' '),
    coarseElapsed: (ms: number) => ({ unit: ms < 3600000 ? 'minute' : 'hour', value: Math.floor(ms / 60000) }),
    Codicon: (props: React.ComponentProps<'span'>) => <span {...props} />,
    DisclosureCaret: ({ open }: { open: boolean }) => <span>{open ? '▼' : '▶'}</span>,
    host: hostMock,
    RowButton: (props: React.ComponentProps<'button'>) => <button {...props} />,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>,
    useI18n: () => ({
      t: {
        sidebar: {
          row: {
            ageNow: 'now',
            ageMin: 'm',
            ageHour: 'h',
            ageDay: 'd'
          }
        }
      }
    }),
    useQuery,
    useValue: (a: { get?: () => unknown }) => a.get?.() ?? {}
  }
})

vi.mock('./routing', () => ({
  backendTargetProfile: (_route: unknown, profile: string) => profile,
  botWorkspaceOwnerKey: (bot: RosterRow) => `bot:${bot.name}`,
  requestForBot: requestForBotMock
}))

vi.mock('./data', () => ({
  botRosterKey: (bot: RosterRow) => bot.name,
  newBotChat: newBotChatMock
}))

vi.mock('./i18n', () => ({
  useBots: () => ({
    bot: {
      sessionsHeading: 'Sessions',
      sessionsLoading: 'Loading…',
      sessionsEmpty: 'No sessions yet',
      sessionUntitled: 'Untitled',
      newSession: 'New session'
    }
  })
}))

beforeEach(() => {
  vi.clearAllMocks()
})

describe('BotSessionsList', () => {
  const bot: RosterRow = {
    connectionId: 'local',
    name: 'alpha',
    sourceScoped: false
  }

  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )

  it('expands and fetches sessions on click', async () => {
    requestForBotMock.mockResolvedValueOnce({
      sessions: [
        { id: 's1', title: 'First session', started_at: Date.now() / 1000 - 3600, message_count: 5 },
        { id: 's2', title: 'Second session', started_at: Date.now() / 1000 - 7200, message_count: 2 }
      ]
    })

    const { getByText, findByText } = render(<BotSessionsList bot={bot} />, { wrapper })

    expect(getByText('Sessions')).toBeTruthy()
    fireEvent.click(getByText('Sessions'))

    expect(await findByText('First session')).toBeTruthy()
    expect(getByText('Second session')).toBeTruthy()
    expect(requestForBotMock).toHaveBeenCalledWith(
      bot,
      'session.list',
      expect.objectContaining({ profile: 'alpha', limit: 15 })
    )
  })

  it('opens a session via host.openSession on row click', async () => {
    requestForBotMock.mockResolvedValueOnce({
      sessions: [{ id: 's1', title: 'Click me', started_at: Date.now() / 1000, message_count: 1 }]
    })

    const { getByText, findByText } = render(<BotSessionsList bot={bot} />, { wrapper })

    fireEvent.click(getByText('Sessions'))
    fireEvent.click(await findByText('Click me'))

    await waitFor(() => {
      expect(hostMock.openSession).toHaveBeenCalledWith(
        's1',
        expect.objectContaining({
          profile: 'alpha',
          intent: 'in-place',
          workspaceMode: 'bots',
          workspaceOwnerKey: 'bot:alpha'
        })
      )
    })
  })

  it('calls newBotChat when new-session affordance is clicked', async () => {
    requestForBotMock.mockResolvedValueOnce({ sessions: [] })

    const { getByText, findByText } = render(<BotSessionsList bot={bot} />, { wrapper })

    fireEvent.click(getByText('Sessions'))
    fireEvent.click(await findByText('New session'))

    expect(newBotChatMock).toHaveBeenCalledWith(bot)
  })

  it('shows empty state when no sessions exist', async () => {
    requestForBotMock.mockResolvedValueOnce({ sessions: [] })

    const { getByText, findByText } = render(<BotSessionsList bot={bot} />, { wrapper })

    fireEvent.click(getByText('Sessions'))
    expect(await findByText('No sessions yet')).toBeTruthy()
  })
})
