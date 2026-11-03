// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import type { MessagingPlatformInfo } from '@/types/hermes'

import { RelayLinkSetup } from './relay-link'

const requestRelayLink = vi.fn()

vi.mock('@/hermes', () => ({
  getRelayLinks: vi.fn(async () => ({ platforms: [{ platform: 'telegram', accounts: [], bot_url: null }] })),
  getRelayLinkState: vi.fn(),
  requestRelayLink: (platform: string) => requestRelayLink(platform),
  unlinkRelayAccount: vi.fn()
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

afterEach(() => {
  cleanup()
  requestRelayLink.mockReset()
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
