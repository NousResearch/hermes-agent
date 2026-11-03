// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import type { MessagingPlatformInfo } from '@/types/hermes'

import { RelayLinkSetup } from './relay-link'

const requestRelayLink = vi.fn()
const getRelayLinkState = vi.fn()
const notify = vi.fn()
const noAccounts = { platforms: [{ platform: 'telegram', accounts: [], bot_url: null }] }
const getRelayLinks = vi.fn(async (): Promise<unknown> => noAccounts)

vi.mock('@/hermes', () => ({
  getRelayLinks: () => getRelayLinks(),
  getRelayLinkState: (code: string) => getRelayLinkState(code),
  requestRelayLink: (platform: string) => requestRelayLink(platform),
  unlinkRelayAccount: vi.fn()
}))

vi.mock('@/store/notifications', () => ({ notify: (n: unknown) => notify(n), notifyError: vi.fn() }))

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  requestRelayLink.mockReset()
  getRelayLinkState.mockReset()
  notify.mockReset()
  getRelayLinks.mockReset()
  getRelayLinks.mockImplementation(async () => noAccounts)
})

const telegram = { id: 'telegram', name: 'Telegram' } as MessagingPlatformInfo

// What the Electron IPC bridge throws for a relay router refusal (real Desktop + connector readback).
const ipcRefusal = (status: number, detail: unknown) =>
  new Error(`Error invoking remote method 'hermes:api': Error: ${status}: ${JSON.stringify({ detail })}`)

test.each([
  [
    ipcRefusal(409, {
      code: 'platform_not_fronted',
      message: 'Team Gateway does not carry telegram on this instance.'
    }),
    'Team Gateway does not carry telegram on this instance.'
  ],
  [
    ipcRefusal(503, {
      code: 'gateway_not_running',
      message: 'Start the gateway first: it holds the Team Gateway connection.'
    }),
    'Start the gateway first: it holds the Team Gateway connection.'
  ],
  [
    new Error("Error invoking remote method 'hermes:api': Error: Gateway ticket control socket missing"),
    'Gateway ticket control socket missing'
  ]
])('a refused link shows the backend sentence, not the IPC envelope', async (error, sentence) => {
  requestRelayLink.mockRejectedValueOnce(error)
  render(<RelayLinkSetup platform={telegram} />)

  fireEvent.click(await screen.findByRole('button', { name: /Link Telegram/ }))

  expect(await screen.findByText(sentence)).toBeTruthy()
  expect(screen.queryByText(/Error invoking remote method|"detail"/)).toBeNull()
})

test('a Portal-managed agent is sent to the Portal: no code is minted and nothing can be unlinked here', async () => {
  // A gateway-minted code on a managed agent carries no owner, so its redeem binds nothing.
  getRelayLinks.mockImplementation(async () => ({
    managed_by: 'portal',
    portal_url: 'https://portal.nousresearch.com/agents',
    platforms: [{ platform: 'telegram', accounts: [{ platformUserId: '7', displayName: 'Ada' }], bot_url: null }]
  }))
  render(<RelayLinkSetup platform={telegram} />)

  expect(await screen.findByRole('button', { name: /Link accounts in Nous Portal/ })).toBeTruthy()
  expect(screen.getByText('Ada')).toBeTruthy()
  expect(screen.queryByRole('button', { name: /^Link Telegram|Unlink/ })).toBeNull()
})

test('a code that goes inactive with no new account is never reported as linked', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  requestRelayLink.mockResolvedValueOnce({
    code: 'AB12CD',
    command: '/link AB12CD',
    expires_at: new Date(Date.now() + 600_000).toISOString(),
    platform: 'telegram'
  })
  getRelayLinkState.mockResolvedValue({ state: 'inactive' })
  render(<RelayLinkSetup platform={telegram} />)

  fireEvent.click(await screen.findByRole('button', { name: /Link Telegram/ }))
  await screen.findByText('/link AB12CD')
  await act(() => vi.advanceTimersByTimeAsync(3500))

  expect(await screen.findByText(/no new account appeared/)).toBeTruthy()
  expect(notify).not.toHaveBeenCalledWith(expect.objectContaining({ kind: 'success' }))
})
