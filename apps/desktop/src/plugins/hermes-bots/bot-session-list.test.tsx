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
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BotRow } from './bot-row'
import { $expandedBotSessions } from './bot-session-list'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const { eventListeners, listPersistedSessions, newChat, onEvent, openRosterBot, openSession } = vi.hoisted(() => ({
  eventListeners: {} as Record<string, Set<() => void>>,
  listPersistedSessions: vi.fn(),
  newChat: vi.fn(),
  onEvent: vi.fn((event: string, callback: () => void) => {
    const listeners = eventListeners[event] ?? (eventListeners[event] = new Set())
    listeners.add(callback)

    return () => {
      listeners.delete(callback)
    }
  }),
  openRosterBot: vi.fn(),
  openSession: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, listPersistedSessions, newChat, onEvent, openSession },
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
  ],
  empty: [{ id: 'e-empty', last_active: 500, message_count: 0, title: 'Empty side chat' }],
  populated: [{ id: 'p-populated', last_active: 500, message_count: 3, title: 'Populated side chat' }]
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

/** Alias identity differs from the backend profile it targets. */
const aliasBot = () =>
  ({
    connectionId: 'remote-a',
    name: 'moxie',
    remoteSource: true,
    route: { connectionId: 'remote-a', mode: 'remote', profile: 'moxie', targetProfile: 'beta' },
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

function emitEvent(event: string) {
  eventListeners[event]?.forEach(listener => listener())
}

beforeEach(() => {
  vi.clearAllMocks()
  Object.keys(eventListeners).forEach(event => delete eventListeners[event])
  $expandedBotSessions.set(new Set())
  SESSIONS_BY_PROFILE.alpha = [
    { id: 'a-office', last_active: 2_000, title: 'Office Operations' },
    { id: 'a-sales', last_active: 1_000, title: 'Sales & Outreach' },
    { id: 'a-bot-chat', last_active: 3_000, title: 'Bot Chat' }
  ]
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
  it('does not open the profile Delete menu when right-clicking a nested conversation', async () => {
    const onDelete = vi.fn()
    const parentContextMenu = vi.fn()

    const { container } = render(
      <div onContextMenu={parentContextMenu}>
        <BotRow bot={betaBot()} onDelete={onDelete} onEdit={noop} onGroup={noop} onNewSection={noop} />
      </div>
    )

    fireEvent.click(disclosure(betaBot()))
    const child = await screen.findByText('Beta Research')
    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true, button: 2 })

    child.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
    expect(parentContextMenu).not.toHaveBeenCalled()
    expect(screen.queryByRole('menuitem', { name: 'Delete' })).toBeNull()
    expect(onDelete).not.toHaveBeenCalled()
    expect(container.querySelector('[data-bot-session-id="b-research"]')).not.toBeNull()

    fireEvent.click(child)
    expect(openSession).toHaveBeenCalledWith('b-research', expect.objectContaining({ intent: 'tab' }))

    fireEvent.contextMenu(container.querySelector('[data-roster-key]')!)
    expect(parentContextMenu).toHaveBeenCalledTimes(1)
  })

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

  it('uses the alias route for list/create and preserves the exact ID and owner on open', async () => {
    render(<BotRow bot={aliasBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(aliasBot()))
    await screen.findByText('Beta Research')
    expect(listPersistedSessions).toHaveBeenCalledWith(aliasBot().route, { limit: 200, profile: 'beta' })

    fireEvent.click(await screen.findByRole('button', { name: 'New chat with this bot' }))
    expect(newChat).toHaveBeenCalledWith(aliasBot().route, {
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:remote-a::moxie'
    })

    fireEvent.click(screen.getByText('Beta Research'))
    expect(openSession).toHaveBeenCalledWith('b-research', {
      route: aliasBot().route,
      intent: 'tab',
      awaitHydration: true,
      expectHistory: true,
      forceResume: true,
      hydrationTimeoutMs: 60_000,
      keepAllProfilesScope: true,
      profile: 'moxie',
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:remote-a::moxie',
      retryHydrationTimeoutOnce: true
    })
  })

  it('opens a listed conversation through the existing open path', async () => {
    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))
    fireEvent.click(await screen.findByText('Sales & Outreach'))

    expect(openSession).toHaveBeenCalledWith('a-sales', {
      intent: 'tab',
      awaitHydration: true,
      expectHistory: true,
      forceResume: true,
      hydrationTimeoutMs: 60_000,
      keepAllProfilesScope: true,
      profile: 'alpha',
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:alpha',
      retryHydrationTimeoutOnce: true
    })
    // Navigation only: the list never touches the canonical open path.
    expect(openRosterBot).not.toHaveBeenCalled()
  })

  it('opens successive child clicks as tabs in the same bot scope', async () => {
    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))
    fireEvent.click(await screen.findByText('Office Operations'))
    fireEvent.click(screen.getByText('Sales & Outreach'))

    expect(openSession.mock.calls.map(([id, options]) => [id, options.intent, options.workspaceOwnerKey])).toEqual([
      ['a-office', 'tab', 'bot:alpha'],
      ['a-sales', 'tab', 'bot:alpha']
    ])
  })

  it('refreshes an expanded list when the backend reports a session lifecycle change', async () => {
    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))
    await screen.findByText('Office Operations')

    SESSIONS_BY_PROFILE.alpha.push({ id: 'a-new', last_active: 4_000, title: 'Freshly titled' })
    emitEvent('sessions.changed')

    expect(await screen.findByText('Freshly titled')).toBeDefined()
    SESSIONS_BY_PROFILE.alpha.pop()
  })

  it('keeps two expanded lists visible across unrelated session refreshes', async () => {
    render(
      <>
        <BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
        <BotRow bot={betaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
      </>
    )
    fireEvent.click(disclosure(alphaBot()))
    fireEvent.click(disclosure(betaBot()))
    await screen.findByText('Office Operations')
    await screen.findByText('Beta Research')
    const alphaList = screen.queryByText('Office Operations')?.closest('[data-bot-sessions]')
    const betaList = screen.queryByText('Beta Research')?.closest('[data-bot-sessions]')
    expect(alphaList).not.toBeNull()
    expect(betaList).not.toBeNull()

    const pending = new Map<string, (result: unknown) => void>()
    listPersistedSessions.mockImplementation((_route: unknown, options: { profile: string }) =>
      new Promise(resolve => pending.set(options.profile, resolve))
    )
    act(() => emitEvent('sessions.changed'))
    expect(pending.size).toBe(2)
    expect(screen.queryByText('Office Operations')?.closest('[data-bot-sessions]')).toBe(alphaList)
    expect(screen.queryByText('Beta Research')?.closest('[data-bot-sessions]')).toBe(betaList)
    expect(screen.getByText('Office Operations')).toBeDefined()
    expect(screen.getByText('Beta Research')).toBeDefined()

    await act(async () => {
      pending.get('alpha')?.({ sessions: SESSIONS_BY_PROFILE.alpha, total: 3 })
      pending.get('beta')?.({ sessions: SESSIONS_BY_PROFILE.beta, total: 1 })
    })
    expect(screen.queryByText('Office Operations')?.closest('[data-bot-sessions]')).toBe(alphaList)
    expect(screen.queryByText('Beta Research')?.closest('[data-bot-sessions]')).toBe(betaList)

    SESSIONS_BY_PROFILE.beta.push({ id: 'b-new', last_active: 4_000, title: 'New beta conversation' })
    act(() => emitEvent('sessions.changed'))
    expect(screen.getByText('Beta Research')).toBeDefined()
    await act(async () => {
      pending.get('alpha')?.({ sessions: SESSIONS_BY_PROFILE.alpha, total: 3 })
      pending.get('beta')?.({ sessions: SESSIONS_BY_PROFILE.beta, total: 2 })
    })
    expect(screen.getByText('New beta conversation')).toBeDefined()
    expect(screen.getByText('Office Operations')).toBeDefined()
  })

  it('removes an archived or hidden session when refreshed rows no longer contain it', async () => {
    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))
    await screen.findByText('Office Operations')

    SESSIONS_BY_PROFILE.alpha = SESSIONS_BY_PROFILE.alpha.filter(row => row.id !== 'a-office')
    emitEvent('sessions.changed')

    await waitFor(() => expect(screen.queryByText('Office Operations')).toBeNull())
    expect(screen.getByText('Sales & Outreach')).toBeDefined()
  })

  it('keeps prior rows visible during a failed refresh and retries successfully', async () => {
    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))
    await screen.findByText('Office Operations')

    listPersistedSessions.mockRejectedValueOnce(new Error('refresh failed'))
    emitEvent('sessions.changed')

    await screen.findByText(/could not refresh conversations/i)
    expect(screen.getByText('Office Operations')).toBeDefined()

    SESSIONS_BY_PROFILE.alpha = [{ id: 'a-retry', last_active: 5_000, title: 'After retry' }]
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    await screen.findByText('After retry')
    expect(screen.queryByText('Office Operations')).toBeNull()
  })
  it('creates a new independent conversation through the bot-owned creation path', async () => {
    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))
    fireEvent.click(await screen.findByRole('button', { name: 'New chat with this bot' }))

    expect(newChat).toHaveBeenCalledWith(
      {
        connectionId: 'local',
        mode: 'local',
        profile: 'alpha',
        targetProfile: 'alpha'
      },
      { workspaceMode: 'bots', workspaceOwnerKey: 'bot:alpha' }
    )
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

    await screen.findByText(/no other conversations/i)
    expect(screen.queryByText('(untitled)')).toBeNull()
  })

  it('shows a recoverable profile error instead of an empty state when the backend reports one', async () => {
    listPersistedSessions.mockResolvedValueOnce({
      errors: [{ error: 'state.db locked', profile: 'alpha' }],
      limit: 200,
      offset: 0,
      sessions: [],
      total: 0
    })

    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)
    fireEvent.click(disclosure(alphaBot()))

    await screen.findByText(/state\.db locked/i)
    expect(screen.queryByText(/no other conversations/i)).toBeNull()
  })

  it('reports a failed read instead of throwing', async () => {
    listPersistedSessions.mockRejectedValueOnce(new Error('source unreachable'))

    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    fireEvent.click(disclosure(alphaBot()))

    await waitFor(() => expect(screen.getByText(/could not load conversations/i)).toBeDefined())
  })

  it.each([
    [201, true],
    [200, false]
  ])(
    'shows the bounded pagination notice only when total exceeds the fetched 200 rows (total=%s)',
    async (total, hasNotice) => {
      listPersistedSessions.mockResolvedValueOnce({
        limit: 200,
        offset: 0,
        sessions: Array.from({ length: 200 }, (_, index) => ({
          id: index === 0 ? 'canonical-bot-chat' : `page-${index}`,
          last_active: 200 - index,
          title: index === 0 ? 'Bot Chat' : `Page conversation ${index}`
        })),
        total
      })

      const { container } = render(
        <BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
      )

      fireEvent.click(disclosure(alphaBot()))

      await waitFor(() => expect(container.querySelectorAll('[data-bot-session-id]')).toHaveLength(5))
      fireEvent.click(screen.getByRole('button', { name: 'Earlier conversations' }))
      await waitFor(() => expect(container.querySelectorAll('[data-bot-session-id]')).toHaveLength(199))
      expect(listedIds(container)).not.toContain('canonical-bot-chat')

      if (hasNotice) {
        expect(screen.getByText('Showing up to 200 recent conversations')).toBeDefined()
      } else {
        expect(screen.queryByText('Showing up to 200 recent conversations')).toBeNull()
      }
    }
  )

  it('groups the five newest conversations and reveals every older row accessibly', async () => {
    SESSIONS_BY_PROFILE.alpha = Array.from({ length: 7 }, (_, index) => ({
      id: `a-${index}`,
      last_active: 7 - index,
      title: `Conversation ${index}`
    }))

    const { container } = render(
      <BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
    )

    fireEvent.click(disclosure(alphaBot()))
    await screen.findByText('Conversation 0')

    expect(listedIds(container)).toEqual(['a-0', 'a-1', 'a-2', 'a-3', 'a-4'])
    fireEvent.click(screen.getByRole('button', { name: 'Earlier conversations' }))
    expect(listedIds(container)).toEqual(['a-0', 'a-1', 'a-2', 'a-3', 'a-4', 'a-5', 'a-6'])
  })
})

describe('child conversation ages', () => {
  it('renders coarse ages from seconds as minutes, hours, days, and omits malformed zero', async () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_000_000_000_000)
    SESSIONS_BY_PROFILE.alpha = [
      { id: 'age-min', last_active: 1_000_000_000 - 6 * 60, title: '6 minute age' },
      { id: 'age-hour', last_active: 1_000_000_000 - 6 * 3600, title: '6 hour age' },
      { id: 'age-day', last_active: 1_000_000_000 - 6 * 86400, title: '6 day age' },
      { id: 'age-zero', last_active: 0, title: 'Malformed age' }
    ]

    render(<BotRow bot={alphaBot()} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)
    fireEvent.click(disclosure(alphaBot()))
    await screen.findByText('6 minute age')

    expect(screen.getByText('6m')).toBeDefined()
    expect(screen.getByText('6h')).toBeDefined()
    expect(screen.getByText('6d')).toBeDefined()
    expect(screen.queryByText('Malformed age')?.parentElement?.textContent).not.toContain('now')
  })
})

describe('child conversation hydration', () => {
  it('does not wait for history when a listed session has zero messages', async () => {
    render(
      <BotRow
        bot={{ connectionId: 'local', name: 'empty' } as RosterRow}
        onDelete={noop}
        onEdit={noop}
        onGroup={noop}
        onNewSection={noop}
      />
    )
    fireEvent.click(disclosure({ name: 'empty' } as RosterRow))
    fireEvent.click(await screen.findByText('Empty side chat'))

    expect(openSession).toHaveBeenCalledWith('e-empty', expect.objectContaining({ expectHistory: false }))
  })

  it('waits for history when a listed session has messages', async () => {
    render(
      <BotRow
        bot={{ connectionId: 'local', name: 'populated' } as RosterRow}
        onDelete={noop}
        onEdit={noop}
        onGroup={noop}
        onNewSection={noop}
      />
    )
    fireEvent.click(disclosure({ name: 'populated' } as RosterRow))
    fireEvent.click(await screen.findByText('Populated side chat'))

    expect(openSession).toHaveBeenCalledWith('p-populated', expect.objectContaining({ expectHistory: true }))
  })
})
