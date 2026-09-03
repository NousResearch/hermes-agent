/**
 * The bot row's two side effects: pre-warming and opening.
 *
 * Pre-warm is per-row and hover-scoped. Warming the whole roster on paint
 * spun up every profile backend the moment the Bots rail rendered, so the row
 * warms exactly one bot and only once a pointer is actually over it — and a
 * source-scoped row pre-dials its OWN source rather than the active gateway.
 *
 * Opening is delegated whole: the row hands its exact roster row to
 * openRosterBot and does nothing else. It never activates a connection
 * itself, which is what keeps a remote row from resolving into the same-named
 * local bot.
 *
 * Ported from tests/profile-prewarm.test.mjs, which sliced BotRow out of the
 * old plugin.js bundle and rendered it against a hand-built jsx stub.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BotRow, GroupRow } from './bot-row'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'

const { ensureAgent, ensureBotMetadata, notifyError, openRosterBot, requestProfile, warmAgent, warmProfile } =
  vi.hoisted(() => ({
    ensureAgent: vi.fn(),
    ensureBotMetadata: vi.fn(),
    notifyError: vi.fn(),
    openRosterBot: vi.fn(),
    requestProfile: vi.fn(),
    warmAgent: vi.fn(),
    warmProfile: vi.fn()
  }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, ensureAgent, notifyError, requestProfile, warmAgent, warmProfile },
    // The plugin bundle normally lands via `ctx.i18n.register` at load, so
    // without this every localized label in the row renders empty.
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
  ensureBotMetadata.mockResolvedValue({ pinned: true })
  openRosterBot.mockResolvedValue(true)
  requestProfile.mockResolvedValue({})
})

describe('pre-warm is hover-scoped, never roster-wide', () => {
  it('warms nothing on paint and exactly the hovered bot on pointer entry', async () => {
    const row = renderRow({ name: 'alpha' } as RosterRow)

    expect(warmProfile).not.toHaveBeenCalled()

    fireEvent.pointerEnter(row)

    expect(warmProfile.mock.calls).toEqual([['alpha']])
    expect(warmAgent).not.toHaveBeenCalled()
  })

  it('pre-dials a source-scoped row on its own source', async () => {
    const row = renderRow({
      connectionId: 'work',
      connectionLabel: 'Work',
      name: 'research',
      remoteSource: true,
      sourceScoped: true
    } as RosterRow)

    fireEvent.pointerEnter(row)

    expect(warmAgent.mock.calls).toEqual([['work', 'research']])
    expect(warmProfile).not.toHaveBeenCalled()
  })
})

describe('the row delegates the open and claims no activation authority', () => {
  it('hands a remote Connections row to openRosterBot without activating it', async () => {
    const bot = {
      connectionId: 'work',
      connectionLabel: 'Work',
      name: 'research',
      remoteSource: true,
      sourceScoped: true
    } as RosterRow

    fireEvent.click(renderRow(bot))

    expect(ensureAgent).not.toHaveBeenCalled()
    expect(openRosterBot.mock.calls).toEqual([[bot]])
  })

  it('never resolves a remote default into the same-named local bot', async () => {
    const bot = {
      connectionId: 'mac-mini',
      connectionLabel: 'Mac Mini',
      name: 'default',
      remoteSource: true,
      sourceScoped: true
    } as RosterRow

    fireEvent.click(renderRow(bot))

    expect(ensureAgent).not.toHaveBeenCalled()
    expect(openRosterBot.mock.calls[0][0].connectionId).toBe('mac-mini')
    expect(notifyError).not.toHaveBeenCalled()
  })
})

describe('the menu opens the same forever-chat a row click does', () => {
  it('opens the canonical chat', async () => {
    const bot = { name: 'alpha' } as RosterRow

    fireEvent.contextMenu(renderRow(bot))
    fireEvent.click(await screen.findByText('Open Bot Chat'))

    expect(openRosterBot.mock.calls).toEqual([[bot]])
  })
})

describe('context-menu mutations hydrate the alias first', () => {
  it('reads the backend row before toggling pin, and writes to the alias target', async () => {
    // A non-identity alias (Desktop calls it `worker`, the backend calls it
    // `backend-worker`) must have its CURRENT state hydrated from its own
    // source before the toggle — flipping a locally-assumed value would
    // fight whatever the backend actually holds.
    const bot = {
      connectionId: 'remote-a',
      name: 'worker',
      remoteSource: true,
      route: { connectionId: 'remote-a', mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' },
      sourceScoped: true
    } as RosterRow

    fireEvent.contextMenu(renderRow(bot))
    // The label reads from LOCAL meta (unpinned here); the toggle reads from
    // the hydrated backend row, which says pinned. That divergence is the
    // point — an alias whose state lives elsewhere must not be flipped
    // against a locally-assumed value.
    fireEvent.click(await screen.findByText('Pin to top'))
    await vi.waitFor(() =>
      expect(requestProfile.mock.calls.some(([, method]) => method === 'profiles.configure')).toBe(true)
    )

    expect(ensureBotMetadata).toHaveBeenCalledWith(bot)

    const [route, , params] = requestProfile.mock.calls.find(([, method]) => method === 'profiles.configure')!

    expect(route.profile).toBe('worker')
    expect(params).toMatchObject({ name: 'backend-worker', ui_meta: { 'hermes-bots': { pinned: false } } })
  })
})

describe('GroupRow renders its needs-you badge from the needsYou prop', () => {
  const noopGroup = () => undefined

  it('shows the question badge when needsYou is true', () => {
    render(
      <GroupRow active={false} group="Core" members={[]} needsYou={true} onDisband={noopGroup} onOpen={noopGroup} />
    )

    expect(screen.getByLabelText('Needs your input')).toBeTruthy()
  })

  it('hides the question badge when needsYou is false', () => {
    render(
      <GroupRow active={false} group="Core" members={[]} needsYou={false} onDisband={noopGroup} onOpen={noopGroup} />
    )

    expect(screen.queryByLabelText('Needs your input')).toBeNull()
  })
})
