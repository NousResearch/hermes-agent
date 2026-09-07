import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { BotMetaSnapshot } from './data'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const fixtures = vi.hoisted(() => {
  const store = <T,>(value: T) => ({
    get: () => value,
    set: (next: T) => {
      value = next
    }
  })

  const alpha: RosterRow = {
    canonical_session: { id: 'alpha-chat' },
    name: 'alpha'
  }

  const beta: RosterRow = {
    canonical_session: { id: 'beta-chat' },
    name: 'beta'
  }

  const remoteAlpha: RosterRow = {
    canonical_session: { id: 'remote-alpha-chat' },
    connectionId: 'remote-a',
    connectionLabel: 'Homelab',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  }

  const remoteBeta: RosterRow = {
    canonical_session: { id: 'remote-beta-chat' },
    connectionId: 'remote-b',
    connectionLabel: 'Studio',
    name: 'default',
    remoteSource: true,
    sourceScoped: true
  }

  const meta = store<BotMetaSnapshot>({
    alpha: { pinned: true },
    beta: { pinned: true },
    'remote-a::default': { pinned: true },
    'remote-b::default': { pinned: true }
  })

  const attention = store<Record<string, { at: number; message: string; reason: string }>>({})

  const rosterState = store<{
    data: { profiles: RosterRow[] } | null
    error: Error | null
  }>({
    data: { profiles: [alpha, beta] },
    error: null
  })

  return {
    alpha,
    beta,
    attention,
    hostRequest: vi.fn(),
    mergeServerMeta: vi.fn(),
    meta,
    openRosterBot: vi.fn(),
    pullServerAvatars: vi.fn(),
    remoteAlpha,
    remoteBeta,
    rosterState,
    setWorkspaceOwnerLabel: vi.fn(),
    trackInboundActivity: vi.fn(),
    backfillMessagingProtocol: vi.fn(),
    store
  }
})

vi.mock('@hermes/plugin-sdk', () => ({
  Codicon: ({ name }: { name: string }) => <span data-testid={`icon-${name}`} />,
  SessionStatusDot: ({ storedSessionId }: { storedSessionId?: string }) => (
    <span data-testid={`status-${storedSessionId || 'none'}`} />
  ),
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(' '),
  host: {
    state: {
      busy: { get: () => false },
      connectionId: { get: () => 'local' },
      focusedStoredSessionId: { get: () => null },
      gateway: { get: () => 'open' },
      profile: { get: () => 'default' },
      request: fixtures.hostRequest,
      setWorkspaceOwnerLabel: fixtures.setWorkspaceOwnerLabel
    }
  },
  usePluginI18n: () => translateBots,
  useValue: <T,>(store: { get: () => T }) => store.get()
}))

vi.mock('./avatar', () => ({
  BotFace: ({ name }: { name: string }) => <span data-testid={`face-${name}`} />,
  avatarColor: (color: string) => color,
  botAppearance: (_name: string, meta: { color?: string } | null) => ({
    color: meta?.color || 'var(--ui-accent)',
    image: null,
    shape: 'blobatar'
  })
}))
vi.mock('./avatar-image', () => ({ isBackfilledFacePng: () => false }))
vi.mock('./bot-state', () => ({
  $botChatFocused: fixtures.store(false),
  $focusedBotOwner: fixtures.store(null),
  $selectedRosterKey: fixtures.store(''),
  focusedRosterOwner: (owner: unknown) => owner
}))

vi.mock('./data', () => ({
  $botAttention: fixtures.attention,
  $botMeta: fixtures.meta,
  $lastRoster: fixtures.store<RosterRow[]>([]),
  botActivitySession: (bot: RosterRow) => bot.canonical_session || bot.last_session || null,
  botRosterKey: (bot: RosterRow) => `${bot.connectionId || 'legacy'}::${bot.name}`,
  botSelectionKey: (bot: RosterRow) =>
    bot.remoteSource || bot.sourceScoped ? `${bot.connectionId || 'legacy'}::${bot.name}` : bot.name,
  botSourceStatus: () => ({ available: true, label: 'Ready' }),
  useRoster: () => fixtures.rosterState.get()
}))

vi.mock('./group-chat', () => ({ $groupChatWorkspace: fixtures.store(null) }))

vi.mock('./i18n', () => ({
  useBots: () => ({
    bot: { openBotChat: 'Open Bot Chat' },
    roster: {
      botsOnly: 'Bots only',
      emptyDesc: 'Pin a bot from Bots to keep it here.',
      emptyTitle: 'No bots yet',
      pinned: 'Pinned',
      waitingForGateway: 'Waiting for gateway'
    }
  })
}))

vi.mock('./labels', () => ({
  displayName: (bot: RosterRow, meta: { title?: string } | null) =>
    meta?.title || bot.name[0].toUpperCase() + bot.name.slice(1)
}))

vi.mock('./routing', () => ({
  botRosterMeta: (bot: RosterRow, meta: BotMetaSnapshot) => {
    const key = bot.remoteSource || bot.sourceScoped ? `${bot.connectionId || 'legacy'}::${bot.name}` : bot.name

    return meta[key] || bot.ui_meta?.['hermes-bots'] || null
  }
}))

vi.mock('./roster-actions', () => ({
  openRosterBot: fixtures.openRosterBot,
  trackInboundActivity: fixtures.trackInboundActivity
}))
vi.mock('./profile-ops', () => ({
  mergeServerMeta: fixtures.mergeServerMeta,
  pullServerAvatars: fixtures.pullServerAvatars
}))
vi.mock('./row-helpers', () => ({
  botCanonicalSessionId: (bot: RosterRow) => bot.canonical_session?.id || null,
  botRowOwnsWorkspace: () => false,
  workerActiveAt: () => false
}))
vi.mock('./soul', () => ({ backfillMessagingProtocol: fixtures.backfillMessagingProtocol }))

beforeEach(() => {
  vi.clearAllMocks()
  fixtures.attention.set({})
  fixtures.rosterState.set({
    data: { profiles: [fixtures.alpha, fixtures.beta] },
    error: null
  })
})

describe('BotPinboard', () => {
  it('renders a three-column bot grid and opens the clicked canonical bot', async () => {
    const { BotPinboard } = await import('./pinboard')
    const { container } = render(<BotPinboard />)

    const grid = container.querySelector('[data-slot="bot-pinboard-grid"]')

    expect(grid?.getAttribute('role')).toBe('grid')
    expect(grid?.classList.contains('grid-cols-3')).toBe(true)
    expect(screen.getByRole('button', { name: 'Open Bot Chat: Alpha' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open Bot Chat: Beta' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Open Bot Chat: Alpha' }))

    expect(fixtures.openRosterBot).toHaveBeenCalledWith(fixtures.alpha)
  })

  it('does not perform Bot Mode writes when the read-only pinboard mounts', async () => {
    const { BotPinboard } = await import('./pinboard')

    render(<BotPinboard />)

    expect(fixtures.hostRequest).not.toHaveBeenCalled()
    expect(fixtures.setWorkspaceOwnerLabel).not.toHaveBeenCalled()
    expect(fixtures.mergeServerMeta).not.toHaveBeenCalled()
    expect(fixtures.pullServerAvatars).not.toHaveBeenCalled()
    expect(fixtures.trackInboundActivity).not.toHaveBeenCalled()
    expect(fixtures.backfillMessagingProtocol).not.toHaveBeenCalled()
  })

  it('keeps same-name source-qualified bots discoverable and routes the exact row', async () => {
    fixtures.rosterState.set({
      data: { profiles: [fixtures.remoteAlpha, fixtures.remoteBeta] },
      error: null
    })

    const { BotPinboard } = await import('./pinboard')
    const { container } = render(<BotPinboard />)

    expect(screen.getByRole('button', { name: 'Open Bot Chat: Default (Homelab)' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open Bot Chat: Default (Studio)' })).toBeTruthy()
    expect(container.textContent).toContain('Homelab')
    expect(container.textContent).toContain('Studio')

    fireEvent.click(screen.getByRole('button', { name: 'Open Bot Chat: Default (Homelab)' }))

    expect(fixtures.openRosterBot).toHaveBeenCalledWith(fixtures.remoteAlpha)
  })

  it('uses its own stale roster when a later roster refresh fails', async () => {
    const { BotPinboard } = await import('./pinboard')
    const view = render(<BotPinboard />)

    expect(screen.getByRole('button', { name: 'Open Bot Chat: Alpha' })).toBeTruthy()

    fixtures.rosterState.set({
      data: { profiles: [fixtures.alpha, fixtures.beta] },
      error: new Error('gateway offline')
    })
    view.rerender(<BotPinboard />)

    expect(screen.getByRole('button', { name: 'Open Bot Chat: Alpha' })).toBeTruthy()
  })

  it('renders an accessible compact zero-state instead of a blank reserved rail', async () => {
    fixtures.rosterState.set({ data: { profiles: [] }, error: null })

    const { BotPinboard } = await import('./pinboard')
    const { container } = render(<BotPinboard />)
    const rail = container.querySelector('[data-slot="bot-pinboard"]')

    expect(rail?.getAttribute('aria-label')).toBe('Bots only')
    expect(screen.getByText('No bots yet')).toBeTruthy()
    expect(screen.queryByRole('grid')).toBeNull()
  })

  it('renders attention and canonical-session status indicators', async () => {
    fixtures.attention.set({
      alpha: { at: 1, message: 'Quota exceeded', reason: 'provider_quota_limit' }
    })

    const { BotPinboard } = await import('./pinboard')

    render(<BotPinboard />)

    expect(screen.getByRole('status', { name: 'Needs attention' })).toBeTruthy()
    expect(screen.getByTestId('status-alpha-chat')).toBeTruthy()
  })
})
