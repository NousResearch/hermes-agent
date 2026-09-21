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
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BotRow, GroupRow } from './bot-row'
import { $groupChats } from './group-chat'
import { translateBotsIn } from './i18n-test-helper'
import type { GroupMember, RosterRow } from './types'

// Which shipped bundle the rendered rows resolve their strings against. `en`
// matches the literals this file once carried, so a case that has to prove a
// string comes from the catalog reads the same row under `ja`.
const locale = vi.hoisted(() => ({ current: 'en' as 'en' | 'ja' | 'zh' }))

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
    usePluginI18n: () => translateBotsIn(locale.current)
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

describe('group-turn presence', () => {
  it('updates only the exact member face and clears it when the room stops', () => {
    const local: RosterRow = { name: 'default', connectionId: 'local' }
    const remote: RosterRow = { name: 'default', connectionId: 'remote', remoteSource: true }

    const { container } = render(
      <>
        {[local, remote].map(bot => (
          <BotRow bot={bot} key={bot.connectionId} onDelete={noop} onEdit={noop} onGroup={noop} onNewSection={noop} />
        ))}
      </>
    )

    const moods = () => [...container.querySelectorAll('[data-hb-mood]')].map(el => el.getAttribute('data-hb-mood'))
    act(() => $groupChats.set({ Room: { log: [], watermarks: {}, running: true, turn: remote } }))
    expect(moods()).toEqual(['idle', 'think'])
    act(() => $groupChats.set({ Room: { log: [], watermarks: {}, running: true, turn: local } }))
    expect(moods()).toEqual(['think', 'idle'])
    act(() => $groupChats.set({}))
    expect(moods()).toEqual(['idle', 'idle'])
  })
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

describe('the bot row context menu speaks the active language', () => {
  afterEach(() => {
    locale.current = 'en'
  })

  it('renders the pin/hide toggles and the groups entry from the catalog, not English literals', async () => {
    // Regression guard for the roster menu items that stayed hardcoded after
    // the bundle landed: under `zh` no English label may survive.
    locale.current = 'zh'
    fireEvent.contextMenu(renderRow({ name: 'worker', connectionId: 'local' }))

    const menu = await screen.findByRole('menu')

    expect(within(menu).getByText('置顶')).toBeTruthy()
    expect(within(menu).getByText('隐藏')).toBeTruthy()
    expect(within(menu).getByText('管理群聊…')).toBeTruthy()
    expect(within(menu).queryByText('Pin to top')).toBeNull()
    expect(within(menu).queryByText('Hide')).toBeNull()
    expect(within(menu).queryByText('Manage groups…')).toBeNull()
  })
})

describe('a group row', () => {
  const members = [{ name: 'alpha' }, { name: 'beta' }, { name: 'gamma' }] as GroupMember[]
  const row = (
    <GroupRow
      active={false}
      group="crew"
      members={members}
      needsYou={false}
      onDisband={noop}
      onNewSection={noop}
      onOpen={noop}
    />
  )

  beforeEach(() => {
    locale.current = 'en'
  })

  it('previews an empty room and describes it to assistive tech in the active language', () => {
    const english = render(row)

    expect(english.getByText('3 bots')).toBeTruthy()
    expect(english.getByRole('button', { name: 'crew, 3 bots, 3 of 3 available' })).toBeTruthy()
    english.unmount()

    locale.current = 'ja'
    const japanese = render(row)

    expect(japanese.getByText('ボット3体')).toBeTruthy()
    expect(japanese.getByRole('button', { name: 'crew, ボット3体, 3体中3体が利用可能' })).toBeTruthy()
  })

  it('names the reader in the active language when their line is the latest, without touching the log marker', () => {
    // 'You' is the persisted author sentinel on the log entry; only its rendering localizes.
    act(() =>
      $groupChats.set({
        crew: { log: [{ at: 1, from: { kind: 'user', name: 'You' }, text: 'ship it' }], running: false, watermarks: {} }
      })
    )
    locale.current = 'zh'

    const chinese = render(row)

    expect(chinese.getByText('你: ship it')).toBeTruthy()
    expect(chinese.queryByText(/^You:/)).toBeNull()
    expect($groupChats.get().crew.log[0].from.name).toBe('You')

    act(() => $groupChats.set({}))
  })
})

describe('age label reflects the last worker run, not only the last conversation (#105874)', () => {
  const nowSec = () => Date.now() / 1000

  it('shows the worker-run age for a delegate-only bot whose worker is past the liveness window', () => {
    // A specialist driven only via delegate_task: its newest human conversation is 11 days old,
    // but it ran a `tool`/`kanban` worker 2h ago (well past the 150s liveness window). The label
    // must read "2h", not "11d" — the busiest bot in the system used to read as the most idle.
    renderRow({
      name: 'auswerter',
      last_session: { last_active: nowSec() - 11 * 86400 },
      worker_session: { last_active: nowSec() - 2 * 3600 }
    } as RosterRow)

    expect(screen.getByText('2h')).toBeTruthy()
    expect(screen.queryByText('11d')).toBeNull()
  })

  it('falls back to conversation age when there is no worker session', () => {
    // worker_session can be absent (None past the 20-row window); the max degrades to the chat age.
    renderRow({
      name: 'chatty',
      last_session: { last_active: nowSec() - 3 * 86400 }
    } as RosterRow)

    expect(screen.getByText('3d')).toBeTruthy()
  })
})
