import { focusManager, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { HermesGateway } from '@/api/client'
import { StatusbarControls } from '@/app/shell/statusbar-controls'
import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { I18nProvider } from '@/i18n'
import { createClientSessionState } from '@/lib/chat-runtime'
import {
  closeSecondaryGateways,
  configureGatewayRegistry,
  openGatewayForAgent,
  reportPrimaryGatewayState,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} from '@/store/gateway'
import { setInterfaceMode } from '@/store/interface-mode'
import {
  $activeSessionId,
  $connection,
  $currentModel,
  $currentProvider,
  $currentUsage,
  $selectedStoredSessionId,
  $sessions,
  _resetSessionOwnerHintsForTests,
  setSessionOwnerHint
} from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'
import { $statusbarHiddenIds, $statusbarVisible, STATUSBAR_HIDDEN_BY_DEFAULT } from '@/store/statusbar-prefs'
import { AccountUsagePeer, type UsageRequest } from '@/test/account-usage-peer'
import { stubMenuDomApis } from '@/test/jsdom'
import type { AccountUsageResponse, AccountUsageSnapshot } from '@/types/account-usage'

import { useStatusbarItems } from './use-statusbar-items'

const REFRESH_MS = 180_000
const BACKOFF_MS = 900_000
const GROUP_ID = 'account-usage-scenario'

let primary: HermesGateway
let queryClient: QueryClient

function reading(provider: AccountUsageSnapshot['provider'] = 'openrouter', remaining = 64): AccountUsageResponse {
  return {
    provider,
    reason: null,
    snapshot: {
      credits_balance: provider === 'openrouter' ? 123.45 : null,
      fetched_at: '2030-01-02T03:04:00Z',
      plan: provider === 'openai-codex' ? 'Pro' : null,
      provider,
      rows:
        provider === 'openrouter'
          ? [
              { args: { currency: 'USD', value: 123.45 }, key: 'credits_balance' },
              { args: { daily: 0.5, monthly: 8, total: 12.5, weekly: 2 }, key: 'api_key_usage' }
            ]
          : provider === 'openai-codex'
            ? [{ args: { count: 3 }, key: 'banked_resets' }]
            : [{ args: { currency: 'USD', limit: 50, used: 12.5 }, key: 'extra_usage' }],
      windows: [
        {
          label: 'Provider-side label',
          label_key: provider === 'openrouter' ? 'api_key_quota' : 'session',
          limit: provider === 'openrouter' ? 100 : null,
          limit_remaining: provider === 'openrouter' ? remaining : null,
          remaining_percent: remaining,
          reset_at: '2030-01-03T04:05:00Z',
          reset_interval: provider === 'openrouter' ? 'monthly' : null,
          scope: 'account',
          used_percent: 100 - remaining
        },
        ...(provider === 'anthropic'
          ? [{ label: 'Opus week', label_key: 'opus_week' as const, remaining_percent: 0, scope: 'model' as const, used_percent: 100 }]
          : [])
      ]
    },
    status: 'ok'
  }
}

function choosePrimary(provider = 'openrouter', storedId = 'primary-chat', runtimeId = 'primary-runtime'): void {
  const state = createClientSessionState(storedId)

  state.model = 'fixture-model'
  state.provider = provider
  state.usage = { calls: 1, input: 1200, output: 300, total: 1500 }
  $sessionStates.set({ [runtimeId]: state })
  $activeSessionId.set(runtimeId)
  $selectedStoredSessionId.set(storedId)
  $currentModel.set(state.model)
  $currentProvider.set(provider)
  $currentUsage.set(state.usage)
  setSessionOwnerHint(storedId, { connectionId: 'local', profile: 'default' })
  $layoutTree.set(group(['workspace'], { active: 'workspace', id: GROUP_ID }))
  noteActiveTreeGroup(GROUP_ID)
}

/** Exercise the shell's actual item registry and dropdown, with the peer at the network boundary. */
function StatusbarScenario() {
  const { statusbarItems } = useStatusbarItems({
    agentsOpen: false,
    changeSessionCwd: async () => {},
    chatOpen: true,
    commandCenterOpen: false,
    extraLeftItems: [],
    extraRightItems: [],
    freshDraftReady: false,
    gatewayState: 'open',
    inferenceStatus: null,
    openAgents: () => {},
    openCommandCenterSection: () => {},
    requestGateway: (method, params) => primary.request(method, params),
    statusSnapshot: null,
    toggleCommandCenter: () => {}
  })

  return <StatusbarControls items={statusbarItems} />
}

function mount() {
  return render(
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale="en">
        <MemoryRouter>
          <StatusbarScenario />
        </MemoryRouter>
      </I18nProvider>
    </QueryClientProvider>
  )
}

async function advance(milliseconds = 1): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds)
  })
}

function chip(name = 'OpenRouter'): HTMLElement {
  return screen.getByRole('button', { name: new RegExp(`^${name}(?:[: ]|$)`) })
}

async function openPanel(name?: string): Promise<HTMLElement> {
  fireEvent.keyDown(chip(name), { key: 'ArrowDown' })
  await advance()

  const panel = window.document.querySelector<HTMLElement>('[data-slot="account-usage-panel"]')

  expect(panel).not.toBeNull()

  return panel!
}

beforeEach(async () => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2030-01-02T03:04:00Z'))
  vi.stubGlobal('WebSocket', AccountUsagePeer)
  stubMenuDomApis()
  AccountUsagePeer.requests = []
  AccountUsagePeer.instances = []
  AccountUsagePeer.onUsage = ({ frame, peer }) => peer.answer(frame, reading())
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  focusManager.setFocused(true)
  setInterfaceMode('advanced')
  $statusbarHiddenIds.set([...STATUSBAR_HIDDEN_BY_DEFAULT])
  $statusbarVisible.set(true)
  $sessionTiles.set([])
  $sessions.set([])
  _resetSessionOwnerHintsForTests({ storage: true })
  configureGatewayRegistry({ activeConnectionId: () => 'local', foregroundScopes: () => new Set(), onEvent: () => {} })
  primary = new HermesGateway()
  await primary.connect('wss://local.example.invalid/api/ws?ticket=fixture')
  setPrimaryGateway(primary, 'default')
  setPrimaryGatewayConnectionId('local', 'local')
  reportPrimaryGatewayState('open')
  choosePrimary()
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  closeSecondaryGateways()
  primary.close()
  setPrimaryGateway(null)
  setPrimaryGatewayConnectionId(null)
  $connection.set(null)
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessionStates.set({})
  $sessionTiles.set([])
  $sessions.set([])
  _resetSessionOwnerHintsForTests({ storage: true })
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  focusManager.setFocused(undefined)
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('provider account usage through the Desktop statusbar', () => {
  it.each([
    { provider: 'openrouter' as const, name: 'OpenRouter' },
    { provider: 'openai-codex' as const, name: 'Codex' },
    { provider: 'anthropic' as const, name: 'Anthropic' }
  ])('opens normalized $provider limits and the focused session tokens', async ({ name, provider }) => {
    // * Catch an unwired statusbar entry, a used/remaining inversion, and model-only depletion of an account.
    choosePrimary(provider)
    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.answer(frame, reading(provider, 64))
    mount()
    await advance()

    expect(chip(name).textContent).toContain('64%')

    if (provider === 'openrouter') {
      expect(chip(name).textContent).toContain('$123.45')
    }

    expect(window.document.querySelector('[data-slot="account-usage-panel"]')).toBeNull()
    const panel = await openPanel(name)
    const text = panel.textContent ?? ''

    expect(text).toContain('64%')
    expect(text).toContain('36%')
    expect(text).toMatch(/1\.2k.*300.*1\.5k/)
    expect(text).toMatch(/2030/)
    expect(text).not.toContain('Provider-side label')

    if (provider === 'openrouter') {
      expect(text).toMatch(/\$64\.00.*\$100\.00/)
      expect(text).toMatch(/\$12\.50.*\$0\.50.*\$2\.00.*\$8\.00/)
    } else if (provider === 'openai-codex') {
      expect(text).toContain('Pro')
      expect(text).toMatch(/\b3\b/)
    } else {
      expect(within(panel).getAllByRole('listitem').some(row => /0%.*100%/.test(row.textContent ?? ''))).toBe(true)
      expect(text).toMatch(/\$12\.50.*\$50\.00/)
    }

    expect(AccountUsagePeer.requests.map(({ frame }) => frame.params)).toEqual([
      { profile: 'default', session_id: 'primary-runtime' }
    ])
  })

  it('polls while the panel is closed and the existing visibility menu stops hidden polling', async () => {
    // * Catch polling coupled to popover mount, or a hidden item that keeps spending provider requests.
    mount()
    await advance()
    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.answer(frame, reading('openrouter', 19))
    await advance(REFRESH_MS)
    expect(chip().textContent).toContain('19%')
    expect(AccountUsagePeer.requests).toHaveLength(2)
    expect(window.document.querySelector('[data-slot="account-usage-panel"]')).toBeNull()

    fireEvent.contextMenu(window.document.querySelector('[data-slot="statusbar"]')!)
    await advance()
    const entry = screen.getAllByRole('menuitemcheckbox').find(item => /account/i.test(item.textContent ?? ''))!

    expect(entry).toBeDefined()
    fireEvent.click(entry)
    await advance()
    expect(entry.getAttribute('aria-checked')).toBe('false')
    expect(screen.queryByRole('button', { name: /^OpenRouter/ })).toBeNull()
    await advance(REFRESH_MS * 3)
    expect(AccountUsagePeer.requests).toHaveLength(2)
  })

  it('clears failed readings, preserves per-key backoff through remount, and recovers on manual refresh', async () => {
    // * Catch stale success after failures and a remount that resets the failure budget and floods the provider.
    const view = mount()
    await advance()
    expect(chip().textContent).toContain('$123.45')
    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.reject(frame, 5001)
    await advance(REFRESH_MS)
    expect(chip().textContent).not.toContain('$123.45')
    await advance(REFRESH_MS)
    await advance(REFRESH_MS)
    expect(AccountUsagePeer.requests).toHaveLength(4)
    view.unmount()
    mount()
    await advance(REFRESH_MS)
    expect(AccountUsagePeer.requests).toHaveLength(4)

    const panel = await openPanel()

    expect(panel.textContent).not.toContain('$123.45')
    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.answer(frame, reading('openrouter', 4.9))
    fireEvent.click(within(panel).getByRole('button'))
    await advance()
    expect(AccountUsagePeer.requests).toHaveLength(5)
    expect(panel.textContent).toContain('4%')
    await advance(REFRESH_MS)
    expect(AccountUsagePeer.requests).toHaveLength(6)
  })

  it('waits fifteen minutes after three consecutive failures without a mount reset', async () => {
    // * Catch an error path that keeps the normal three-minute cadence after its retry budget is spent.
    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.reject(frame, 5001)
    mount()
    await advance()
    await advance(REFRESH_MS)
    await advance(REFRESH_MS)
    expect(AccountUsagePeer.requests).toHaveLength(3)
    const failedAt = AccountUsagePeer.requests[2].receivedAt

    await advance(BACKOFF_MS - (Date.now() - failedAt) - 1)
    expect(AccountUsagePeer.requests).toHaveLength(3)
    await advance(2)
    expect(AccountUsagePeer.requests).toHaveLength(4)
  })

  it.each(['missing-rpc', 'unsupported'] as const)('hides a %s backend until its socket changes', async failure => {
    // * Catch permanent capability failures retried every poll or reused after a backend upgrade.
    AccountUsagePeer.onUsage = ({ frame, peer }) => {
      if (failure === 'missing-rpc') {
        peer.reject(frame, -32601)
      } else {
        peer.answer(frame, { provider: 'openrouter', reason: 'provider_unsupported', snapshot: null, status: 'unsupported' })
      }
    }

    const view = mount()
    await advance()
    expect(screen.queryByRole('button', { name: /^OpenRouter/ })).toBeNull()
    view.unmount()
    mount()
    await advance(BACKOFF_MS)
    expect(AccountUsagePeer.requests).toHaveLength(1)

    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.answer(frame, reading('openrouter', 81))
    await act(async () => {
      primary.close()

      if (failure === 'unsupported') {
        primary = new HermesGateway()
      }

      await primary.connect('wss://local.example.invalid/api/ws?ticket=upgraded-fixture')
      setPrimaryGateway(primary, 'default')
      setPrimaryGatewayConnectionId('local', 'local')
      reportPrimaryGatewayState('open')
    })
    await advance()
    expect(AccountUsagePeer.requests).toHaveLength(2)
    expect(chip().textContent).toContain('81%')
  })

  it('discards the old socket response when the same session moves to a replacement transport', async () => {
    // * Catch a cached or late result crossing a connection re-home with identical session, provider and model.
    let pending: UsageRequest | undefined

    AccountUsagePeer.onUsage = request => { pending = request }
    mount()
    await advance()
    expect(AccountUsagePeer.requests).toHaveLength(1)
    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.answer(frame, reading('openrouter', 88))
    const previous = primary

    await act(async () => {
      primary = new HermesGateway()
      await primary.connect('wss://replacement.example.invalid/api/ws?ticket=fixture')
      setPrimaryGateway(primary, 'default')
      setPrimaryGatewayConnectionId('local', 'local')
      reportPrimaryGatewayState('open')
    })
    await advance()
    expect(AccountUsagePeer.requests).toHaveLength(2)
    expect(chip().textContent).toContain('88%')
    await act(async () => pending!.peer.answer(pending!.frame, reading('openrouter', 2)))
    await advance()
    expect(chip().textContent).toContain('88%')
    previous.close()
  })

  it('keeps a focused remote profile on its owning socket and ignores its late reply after focus changes', async () => {
    // * Catch cross-account reads on the ambient gateway and an old tile response replacing the new selection.
    window.hermesDesktop = {
      getConnectionFor: async () => ({
        authMode: 'token',
        baseUrl: 'https://remote.example.invalid',
        connectionId: 'remote-usage',
        mode: 'remote',
        profile: 'worker',
        token: 'synthetic-test-token',
        wsUrl: 'wss://remote.example.invalid/api/ws?ticket=fixture'
      }),
      getGatewayWsUrlFor: async () => 'wss://remote.example.invalid/api/ws?ticket=fixture'
    } as never
    await openGatewayForAgent('remote-usage', 'worker')
    const remote = createClientSessionState('remote-stored')

    remote.provider = 'openrouter'
    remote.model = 'fixture-model'
    remote.usage = { calls: 1, input: 88, output: 11, total: 99 }
    $sessionStates.set({ ...$sessionStates.get(), 'remote-runtime': remote })
    $sessionTiles.set([{
      ownerRoute: { connectionId: 'remote-usage', profile: 'worker', targetProfile: 'backend-worker' },
      runtimeId: 'remote-runtime',
      storedSessionId: 'remote-stored'
    }])
    $layoutTree.set(group(['workspace', 'session-tile:remote-stored'], { active: 'session-tile:remote-stored', id: GROUP_ID }))
    let pending: UsageRequest | undefined

    AccountUsagePeer.onUsage = request => { pending = request }
    mount()
    await advance()
    expect(pending?.peer.url).toContain('remote.example.invalid')
    expect(pending?.frame.params).toEqual({ profile: 'backend-worker', session_id: 'remote-runtime' })

    AccountUsagePeer.onUsage = ({ frame, peer }) => peer.answer(frame, reading('openrouter', 72))
    act(() => $layoutTree.set(group(['workspace', 'session-tile:remote-stored'], { active: 'workspace', id: GROUP_ID })))
    await advance()
    expect(chip().textContent).toContain('72%')
    await act(async () => pending!.peer.answer(pending!.frame, reading('openrouter', 1)))
    await advance()
    expect(chip().textContent).toContain('72%')
    const panel = await openPanel()

    expect(panel.textContent).toMatch(/1\.2k.*300.*1\.5k/)
    expect(panel.textContent).not.toMatch(/88.*11.*99/)
    expect(AccountUsagePeer.requests.at(-1)?.peer.url).toContain('local.example.invalid')
  })

  it('does not query an unsupported provider or a session whose owner is unknown', async () => {
    // * Catch an ownerless session probing the active account, including the draft-to-owned transition.
    choosePrimary('custom')
    const view = mount()
    await advance(REFRESH_MS)
    expect(AccountUsagePeer.requests).toHaveLength(0)
    act(() => {
      choosePrimary('openrouter', 'unknown-stored', 'unknown-runtime')
      _resetSessionOwnerHintsForTests({ storage: true })
      $sessionStates.set({ ...$sessionStates.get() })
    })
    await advance(REFRESH_MS)
    expect(AccountUsagePeer.requests).toHaveLength(0)
    act(() => choosePrimary('openrouter', 'unknown-stored', 'unknown-runtime'))
    await advance()
    expect(AccountUsagePeer.requests).toHaveLength(1)
    expect(chip().textContent).toContain('64%')
    view.unmount()
  })
})
