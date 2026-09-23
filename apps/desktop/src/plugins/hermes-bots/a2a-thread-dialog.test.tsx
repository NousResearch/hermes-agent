/**
 * The bot-to-bot thread dialog: both sides read through the SDK, merged into
 * one exchange, with the state line that answers "are they done?".
 *
 * The read itself is `host.sessionMessages` — mocked here, because the point
 * under test is the composition: two transcripts in, one ordered thread and
 * one honest status out.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

const { sessionMessages } = vi.hoisted(() => ({
  sessionMessages: vi.fn<(route: unknown, options: { sessionId: string }) => Promise<{ messages: unknown[]; sessionId: string }>>(
    async () => ({ messages: [], sessionId: 'x' })
  )
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const Box = ({ children }: { children?: ReactNode }) => <div>{children}</div>

  return {
    ...sdk,
    Badge: Box,
    Button: ({ children, onClick }: { children?: ReactNode; onClick?: () => void }) => <button onClick={onClick}>{children}</button>,
    Codicon: () => null,
    Dialog: Box,
    DialogContent: Box,
    DialogDescription: Box,
    DialogFooter: Box,
    DialogHeader: Box,
    DialogTitle: Box,
    ScrollArea: Box,
    host: { ...sdk.host, sessionMessages },
    useI18n: () => ({
      t: {
        common: { loading: 'Loading…' },
        sidebar: { row: { ageDay: 'd', ageHour: 'h', ageMin: 'm', ageNow: 'now' } }
      }
    }),
    usePluginI18n: () => translateBots
  }
})

const { A2aThreadDialog } = await import('./a2a-thread-dialog')

const row = (name: string, sessionId: string) =>
  ({ canonical_session: { id: sessionId, resolved_id: sessionId }, name }) as unknown as Parameters<typeof A2aThreadDialog>[0]['bot']

const at = (seconds: number) => 1_758_300_000 + seconds

afterEach(() => {
  cleanup()
  sessionMessages.mockReset()
})

describe('A2aThreadDialog', () => {
  it('merges both bots’ transcripts into one exchange', async () => {
    sessionMessages.mockImplementation(async (_route: unknown, options: { sessionId: string }) =>
      options.sessionId === 's-platform'
        ? {
            messages: [
              { content: 'Message from 🤖 hotel dev (@hotel-dev): SDK gap report attached', role: 'user', timestamp: at(10) },
              { content: 'Confirmed — shipping the fix.', role: 'assistant', timestamp: at(20) }
            ],
            sessionId: 's-platform'
          }
        : {
            messages: [{ content: 'Message from 🤖 platform-engineer (@platform-engineer): Ready for verification.', role: 'user', timestamp: at(40) }],
            sessionId: 's-hotel'
          }
    )

    render(
      <A2aThreadDialog
        bot={row('platform-engineer', 's-platform')}
        onClose={() => undefined}
        open
        peerHandle="hotel-dev"
        roster={[row('hotel-dev', 's-hotel')]}
      />
    )

    await waitFor(() => expect(screen.getByText('Confirmed — shipping the fix.')).toBeTruthy())
    expect(screen.getByText('SDK gap report attached')).toBeTruthy()
    expect(screen.getByText('Ready for verification.')).toBeTruthy()
  })

  it('answers “are they done?” — names the bot that still owes a reply', async () => {
    sessionMessages.mockImplementation(async (_route: unknown, options: { sessionId: string }) =>
      options.sessionId === 's-platform'
        ? { messages: [{ content: 'Message from 🤖 hotel dev (@hotel-dev): waiting on your audit', role: 'user', timestamp: at(10) }], sessionId: 's-platform' }
        : { messages: [], sessionId: 's-hotel' }
    )

    render(
      <A2aThreadDialog
        bot={row('platform-engineer', 's-platform')}
        onClose={() => undefined}
        open
        peerHandle="hotel-dev"
        roster={[row('hotel-dev', 's-hotel')]}
      />
    )

    await waitFor(() => expect(screen.getByText('Waiting on @platform-engineer to reply')).toBeTruthy())
  })

  it('reads a settled exchange as settled, not as silence', async () => {
    sessionMessages.mockImplementation(async (_route: unknown, options: { sessionId: string }) =>
      options.sessionId === 's-platform'
        ? {
            messages: [
              { content: 'Message from 🤖 hotel dev (@hotel-dev): gap report', role: 'user', timestamp: at(10) },
              { content: 'Fixed and shipped.', role: 'assistant', timestamp: at(30) }
            ],
            sessionId: 's-platform'
          }
        : { messages: [], sessionId: 's-hotel' }
    )

    render(
      <A2aThreadDialog
        bot={row('platform-engineer', 's-platform')}
        onClose={() => undefined}
        open
        peerHandle="hotel-dev"
        roster={[row('hotel-dev', 's-hotel')]}
      />
    )

    await waitFor(() => expect(screen.getByText('Settled')).toBeTruthy())
  })
})
