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

import { BotRow } from './bot-row'
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

describe('the menu toggles mesh visibility (private/public)', () => {
  it('flips `private` from the HYDRATED backend row and persists it via profiles.configure', async () => {
    // Same divergence the pin test pins: the LABEL reads local meta (seeded private
    // here, so it offers "Make public"), but the TOGGLE reads the hydrated backend
    // row, which says public. The write must flip the backend truth (public -> private),
    // never the locally-assumed value — a bot whose state lives elsewhere must not be
    // flipped against a stale local copy.
    ensureBotMetadata.mockResolvedValue({ private: false })

    const bot = {
      connectionId: 'remote-a',
      name: 'lucky',
      remoteSource: true,
      route: { connectionId: 'remote-a', mode: 'remote', profile: 'lucky', targetProfile: 'backend-lucky' },
      sourceScoped: true
    } as RosterRow

    const { $botMeta, botMetaKey } = await import('./data')

    $botMeta.set({ ...$botMeta.get(), [botMetaKey(bot)]: { private: true } })

    fireEvent.contextMenu(renderRow(bot))
    fireEvent.click(await screen.findByText('Make public'))
    await vi.waitFor(() =>
      expect(requestProfile.mock.calls.some(([, method]) => method === 'profiles.configure')).toBe(true)
    )

    expect(ensureBotMetadata).toHaveBeenCalledWith(bot)

    const [route, , params] = requestProfile.mock.calls.find(([, method]) => method === 'profiles.configure')!

    expect(route.profile).toBe('lucky')
    expect(params).toMatchObject({ name: 'backend-lucky', ui_meta: { 'hermes-bots': { private: true } } })
  })

  it('offers "Make public" for a bot that is already private, and clears the flag', async () => {
    // Same remote/source-scoped shape as the pin test, so the write takes the
    // profiles.configure path this suite asserts on. Local meta is seeded under the
    // row's REAL meta key so the label reads "Make public"; the hydrated backend row
    // agrees, so the toggle clears the flag.
    ensureBotMetadata.mockResolvedValue({ private: true })

    const bot = {
      connectionId: 'remote-a',
      name: 'lucky',
      remoteSource: true,
      route: { connectionId: 'remote-a', mode: 'remote', profile: 'lucky', targetProfile: 'backend-lucky' },
      sourceScoped: true
    } as RosterRow

    const { $botMeta, botMetaKey } = await import('./data')

    $botMeta.set({ ...$botMeta.get(), [botMetaKey(bot)]: { private: true } })

    fireEvent.contextMenu(renderRow(bot))
    fireEvent.click(await screen.findByText('Make public'))
    await vi.waitFor(() =>
      expect(requestProfile.mock.calls.some(([, method]) => method === 'profiles.configure')).toBe(true)
    )

    const [, , params] = requestProfile.mock.calls.find(([, method]) => method === 'profiles.configure')!

    expect(params).toMatchObject({ name: 'backend-lucky', ui_meta: { 'hermes-bots': { private: false } } })
  })
})

describe('the row shows the bot\'s circle beside its handle', () => {
  const bot = { connectionId: 'local', name: 'lucky', route: { connectionId: 'local', mode: 'local', profile: 'lucky' } } as RosterRow

  it('renders "· <circle>" when a circle is set and the handle is shown', async () => {
    const { $botMeta, botMetaKey } = await import('./data')
    $botMeta.set({ ...$botMeta.get(), [botMetaKey(bot)]: { circle: 'hobby' } })

    render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} showHandle />)

    expect(screen.getByText('hobby')).toBeTruthy()
    expect(document.querySelector('[data-slot="bot-circle"]')?.textContent).toBe('hobby')
  })

  it('renders no circle tag when the circle is unset (the shared circle is the quiet default)', async () => {
    const { $botMeta, botMetaKey } = await import('./data')
    $botMeta.set({ ...$botMeta.get(), [botMetaKey(bot)]: {} })

    render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} showHandle />)

    expect(document.querySelector('[data-slot="bot-circle"]')).toBeNull()
  })

  it('shows the circle even when the handle is not needed for disambiguation', async () => {
    // showHandle is only on when two bots share a title. The circle is a membership fact and
    // must not ride on that: a bot whose title equals its handle still shows its circle, and
    // the details row opens for it even without a handle or a session preview.
    const { $botMeta, botMetaKey } = await import('./data')
    $botMeta.set({ ...$botMeta.get(), [botMetaKey(bot)]: { circle: 'hobby' } })

    render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    expect(document.querySelector('[data-slot="bot-circle"]')?.textContent).toBe('hobby')
  })

  it('displays a hand-edited circle in the same case the mesh matches it in', async () => {
    // The dialog lower-cases on save and the gateway lower-cases on read, so a value that
    // arrived any other way (hand-edited profile.yaml, an older client's sync) must not render
    // as a different-looking circle from the one it actually belongs to.
    const { $botMeta, botMetaKey } = await import('./data')
    $botMeta.set({ ...$botMeta.get(), [botMetaKey(bot)]: { circle: '  Work  ' } })

    render(<BotRow bot={bot} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />)

    expect(document.querySelector('[data-slot="bot-circle"]')?.textContent).toBe('work')
  })
})

