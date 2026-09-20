// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>(null),
  ensureGatewayForAgent: vi.fn(async () => undefined),
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))

const { $gateway } = await import('@/store/gateway')
const { $sessionTiles, setSessionTileWorkspaceScope } = await import('@/store/session-states')
const { agentAvatarCache } = await import('./user-message')
const { BotChatAvatar, botChatHandle, useBotChatHandle } = await import('./bot-chat-avatar')

/** Renders the hook's value as text so the gate can be asserted directly. */
const HandleProbe = ({ runtimeId }: { runtimeId: null | string }) => (
  <span data-testid="handle">{useBotChatHandle(runtimeId) ?? 'none'}</span>
)

beforeEach(() => {
  $sessionTiles.set([{ runtimeId: 'rt-1', storedSessionId: 'stored-1' }])
  agentAvatarCache.clear()
})

afterEach(cleanup)

describe('botChatHandle', () => {
  it('reads the profile handle out of a bot owner key', () => {
    expect(botChatHandle('bot:personal')).toBe('personal')
  })

  it('drops the local connection prefix a local owner key carries', () => {
    expect(botChatHandle('bot:local::researcher')).toBe('researcher')
  })

  it('is null for a working session scope', () => {
    expect(botChatHandle(undefined)).toBeNull()
    expect(botChatHandle('local::writer')).toBeNull()
  })
})

describe('useBotChatHandle', () => {
  it('answers only while the live session is a bot chat', () => {
    setSessionTileWorkspaceScope('stored-1', { workspaceMode: 'bots', workspaceOwnerKey: 'bot:personal' })
    const { unmount } = render(<HandleProbe runtimeId="rt-1" />)
    expect(screen.getByTestId('handle').textContent).toBe('personal')
    unmount()

    setSessionTileWorkspaceScope('stored-1', { workspaceMode: 'sessions', workspaceOwnerKey: undefined })
    render(<HandleProbe runtimeId="rt-1" />)
    expect(screen.getByTestId('handle').textContent).toBe('none')
  })

  it('stays null without a live session', () => {
    render(<HandleProbe runtimeId={null} />)
    expect(screen.getByTestId('handle').textContent).toBe('none')
  })
})

describe('BotChatAvatar', () => {
  it('shows the profile avatar the gateway serves', async () => {
    $gateway.set({
      request: vi.fn(async (method: string) =>
        method === 'profiles.list'
          ? { profiles: [{ has_avatar: true, name: 'artful' }] }
          : { data: 'data:image/png;base64,iVBORw0KGgo=', found: true }
      )
    } as never)

    render(<BotChatAvatar handle="artful" />)

    await waitFor(() => {
      const img = document.querySelector('img[data-slot="bot-chat-avatar"]')

      expect(img).toBeTruthy()
      expect(img?.getAttribute('src')).toContain('data:image/png')
    })
  })

  it('falls back to the agent glyph when the profile has no art', async () => {
    $gateway.set({
      request: vi.fn(async () => ({ profiles: [{ has_avatar: false, name: 'artless' }] }))
    } as never)

    render(<BotChatAvatar handle="artless" />)

    await waitFor(() => {
      expect(document.querySelector('span[data-slot="bot-chat-avatar"]')).toBeTruthy()
      expect(document.querySelector('img[data-slot="bot-chat-avatar"]')).toBeNull()
    })
  })
})
