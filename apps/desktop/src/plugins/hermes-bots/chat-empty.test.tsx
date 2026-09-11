import type * as HermesSdk from '@hermes/plugin-sdk'
import { render, screen } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

import { $openBotChat } from './bot-state'
import { BotChatEmpty } from './chat-empty'
import { $botMeta, $lastRoster } from './data'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { atom } = await import('nanostores')

  return {
    ...sdk,
    host: { ...sdk.host, state: { ...sdk.host.state, focusedStoredSessionId: atom('draft-session') } },
    useValue: <T,>(store: { get: () => T }) => store.get(),
    Wordmark: ({ text }: { text: string }) => <div>{text}</div>
  }
})

vi.mock('./avatar', () => ({
  avatarColor: () => '#000',
  botAppearance: () => ({ color: '#000', image: null, shape: 'blobatar' }),
  BotFace: () => <div data-testid="bot-face" />
}))
vi.mock('./avatar-image', () => ({ isBackfilledFacePng: () => false }))
vi.mock('./i18n', () => ({ useBots: () => ({ bot: { chatEmpty: 'Say something to get started.' } }) }))
vi.mock('./labels', () => ({ displayName: (bot: RosterRow) => bot.title || bot.name }))
vi.mock('./routing', () => ({ botRosterMeta: () => null }))

beforeEach(() => {
  $lastRoster.set([])
  $botMeta.set({})
  $openBotChat.set(null)
})

it('identifies an empty legacy bot draft from its open claim', () => {
  $lastRoster.set([
    {
      connectionId: 'remote',
      name: 'seo-ops',
      remoteSource: true,
      sourceScoped: true,
      title: 'Seo Ops'
    } as RosterRow
  ])
  $openBotChat.set({ key: 'remote::seo-ops', openedRegistryId: '' })

  render(<BotChatEmpty sessionId="draft-session" />)

  expect(screen.getByText('Seo Ops')).toBeTruthy()
  expect(screen.getByText('Say something to get started.')).toBeTruthy()
})
