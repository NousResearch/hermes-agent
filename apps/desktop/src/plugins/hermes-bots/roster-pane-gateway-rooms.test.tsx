/**
 * The Bots pane with only its gateway transport mocked. The bot list
 * (`profiles.list` / `host.agents`) failing must not hide gateway Group Chats
 * that `groups.list` returned: rooms are part of the list, so the pane shows
 * them instead of the bot-list error.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'

const { request, requestProfile } = vi.hoisted(() => ({ request: vi.fn(), requestProfile: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: {
      ...sdk.host,
      agents: vi.fn(async () => { throw new Error('profiles.list exploded') }),
      request,
      requestProfile,
      state: { ...sdk.host.state, connectionId: atom<null | string>('local'), gateway: atom('open'), profile: atom('default') }
    },
    usePluginI18n: () => translateBotsIn('en')
  }
})

import { BotsPane } from './roster-pane'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it('shows gateway Group Chats when the bot list fails', async () => {
  request.mockRejectedValue(new Error('profiles.list exploded'))
  requestProfile.mockImplementation(async (_route: unknown, method: string) => {
    if (method === 'groups.list') {
      return { rooms: [{ room_id: 'room-1', name: 'Launch crew', members: [], updated_at: Date.now() / 1000 }], next_offset: null }
    }

    throw new Error(`unexpected ${method}`)
  })

  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <BotsPane />
    </QueryClientProvider>
  )

  expect(await screen.findByRole('button', { name: 'Launch crew' })).toBeTruthy()
  expect(screen.queryByText(/exploded/)).toBeNull()
})
