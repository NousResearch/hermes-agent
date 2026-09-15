/**
 * Per-bot reconnect (#104074): a local bot's context menu offers Reconnect
 * that tears down its pooled backend without restarting the app.
 *
 * Remote/ghost rows have no local process to recycle, so the item is disabled.
 * The handler calls window.hermesDesktop.recycleBackend(profile) and invalidates
 * the roster query.
 */
import type * as HermesSdk from '@hermes/plugin-sdk'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BotRow } from './bot-row'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const { ensureAgent, ensureBotMetadata, notify, notifyError, openRosterBot, requestProfile, warmAgent, warmProfile } =
  vi.hoisted(() => ({
    ensureAgent: vi.fn(),
    ensureBotMetadata: vi.fn(),
    notify: vi.fn(),
    notifyError: vi.fn(),
    openRosterBot: vi.fn(),
    requestProfile: vi.fn(),
    warmAgent: vi.fn(),
    warmProfile: vi.fn()
  }))

const queryInvalidate = vi.fn(async () => undefined)

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  return {
    ...sdk,
    host: {
      ...sdk.host,
      ensureAgent,
      notify,
      notifyError,
      requestProfile,
      warmAgent,
      warmProfile
    },
    queryClient: {
      ...sdk.queryClient,
      invalidateQueries: queryInvalidate
    },
    usePluginI18n: () => translateBots
  }
})

vi.mock('./canonical-chat', () => ({
  ensureBotMetadata,
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn(),
  PROFILE_SESSION_LIST_LIMIT: 200
}))

vi.mock('./roster-actions', () => ({ openRosterBot }))

const noop = () => undefined

function renderRow(bot: RosterRow) {
  render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)
  return screen.getByRole('button')
}

beforeEach(() => {
  vi.clearAllMocks()
  ensureBotMetadata.mockResolvedValue({ pinned: false })
  openRosterBot.mockResolvedValue(true)
  requestProfile.mockResolvedValue({})
  queryInvalidate.mockResolvedValue(undefined)
  // Provide a fresh desktop bridge per test — individual tests override as needed.
  ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
    recycleBackend: vi.fn(async () => ({ ok: true }))
  }
})

describe('per-bot reconnect menu (#104074)', () => {
  it('shows Reconnect and calls recycleBackend with the bot profile for a local bot', async () => {
    const recycleBackend = vi.fn(async () => ({ ok: true }))
    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { recycleBackend }

    const bot = { name: 'alpha' } as RosterRow
    fireEvent.contextMenu(renderRow(bot))
    const item = await screen.findByText('Reconnect')
    expect(item).toBeTruthy()
    // Enabled for local bots.
    expect(item.closest('[aria-disabled="true"]')).toBeNull()

    fireEvent.click(item)

    await waitFor(() => expect(recycleBackend).toHaveBeenCalledWith('alpha'))
    expect(queryInvalidate).toHaveBeenCalled()
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'success', message: expect.stringContaining('alpha') })
    )
  })

  it('uses route targetProfile for alias routes', async () => {
    const recycleBackend = vi.fn(async () => ({ ok: true }))
    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { recycleBackend }

    const bot = {
      name: 'moxie',
      route: { connectionId: 'local', mode: 'local', profile: 'moxie', targetProfile: 'backend-moxie' }
    } as unknown as RosterRow

    fireEvent.contextMenu(renderRow(bot))
    fireEvent.click(await screen.findByText('Reconnect'))
    await waitFor(() => expect(recycleBackend).toHaveBeenCalledWith('backend-moxie'))
  })

  it('disables Reconnect for remote and ghost rows', async () => {
    const remote = {
      name: 'research',
      connectionId: 'work',
      remoteSource: true,
      sourceScoped: true
    } as RosterRow
    fireEvent.contextMenu(renderRow(remote))
    const remoteItem = await screen.findByText('Reconnect')
    // Disabled items render with aria-disabled or data-disabled — either signals non-interactive.
    const disabled =
      remoteItem.getAttribute('aria-disabled') === 'true' ||
      remoteItem.hasAttribute('data-disabled') ||
      remoteItem.getAttribute('data-disabled') === 'true'
    expect(disabled).toBe(true)
  })

  it('disables Reconnect for ghost rows', async () => {
    const ghost = { name: 'beta', ghost: true } as RosterRow
    fireEvent.contextMenu(renderRow(ghost))
    const item = await screen.findByText('Reconnect')
    const disabled =
      item.getAttribute('aria-disabled') === 'true' ||
      item.hasAttribute('data-disabled') ||
      item.getAttribute('data-disabled') === 'true'
    expect(disabled).toBe(true)
  })

  it('shows error when recycleBackend is unavailable', async () => {
    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {}
    const bot = { name: 'alpha' } as RosterRow
    fireEvent.contextMenu(renderRow(bot))
    fireEvent.click(await screen.findByText('Reconnect'))
    await waitFor(() => expect(notifyError).toHaveBeenCalled())
  })

  it('surfaces recycleBackend failures', async () => {
    const recycleBackend = vi.fn(async () => {
      throw new Error('pool busy')
    })
    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = { recycleBackend }
    const bot = { name: 'alpha' } as RosterRow
    fireEvent.contextMenu(renderRow(bot))
    fireEvent.click(await screen.findByText('Reconnect'))
    await waitFor(() => expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Reconnect failed'))
  })
})
