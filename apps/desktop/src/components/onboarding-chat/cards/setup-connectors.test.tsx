import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { onComposerSubmitRequest } from '@/app/chat/composer/focus'
import { PRIMARY_SESSION_VIEW, type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { HermesGateway } from '@/hermes'
import { setPrimaryGateway, setPrimaryGatewayConnectionId } from '@/store/gateway'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'
import { forgetSessionOwnerHintsForSession, setSessionOwnerHint } from '@/store/session'
import { deferred } from '@/test/deferred'

import { ConnectorsCard } from './setup'

class CatalogGateway extends HermesGateway {
  override get connectionState() {
    return 'open' as const
  }
  readonly rpc = vi.fn<HermesGateway['request']>()

  override request<T>(...args: Parameters<HermesGateway['request']>): Promise<T> {
    // SAFETY: these fixtures supply the catalog response requested by the card's RPC caller.
    return this.rpc(...args) as Promise<T>
  }
}

const $storedId = atom<string | null>('guide')
const $runtimeId = atom<string | null>('guide-runtime')
const view: SessionView = {
  ...PRIMARY_SESSION_VIEW,
  kind: 'tile',
  $storedId,
  $runtimeId
}

const submit = vi.fn()
let gateway: CatalogGateway
let unsubscribe: () => void

beforeEach(() => {
  $storedId.set('guide')
  $runtimeId.set('guide-runtime')
  gateway = new CatalogGateway()
  setPrimaryGateway(gateway, 'hermes-setup')
  setPrimaryGatewayConnectionId('remote')
  setSessionOwnerHint('guide', { connectionId: 'remote', profile: 'hermes-setup' })
  unsubscribe = onComposerSubmitRequest(submit)
})

afterEach(() => {
  cleanup()
  unsubscribe()
  setPrimaryGateway(null)
  forgetSessionOwnerHintsForSession('guide')
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
  setOnboardingAnswers({ connectors: [] })
})

it('disables pending picks, routes the catalog to the guide owner, and commits enabled slugs as titles', async () => {
  const catalog = deferred<unknown>()
  gateway.rpc.mockReturnValue(catalog.promise)
  render(
    <SessionViewProvider value={view}>
      <div data-composer-surface-id="guide-composer" data-composer-target="tile:guide">
        <ConnectorsCard attrs={{}} locked={false} />
      </div>
    </SessionViewProvider>
  )

  const gmail = screen.getByRole('button', { name: /Gmail$/ })
  expect(gmail.closest('fieldset')?.disabled).toBe(true)
  expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Continue' }).disabled).toBe(true)
  await act(async () =>
    catalog.resolve({
      available: true,
      connectors: [
        { connector: 'googlecalendar', enabled: true },
        { connector: 'gmail', enabled: true },
        { connector: 'notion', enabled: false },
        { connector: 'stripe_mcp', enabled: true }
      ]
    })
  )

  expect(gateway.rpc.mock.calls[0]?.slice(0, 3)).toEqual(['connectors.list', { session_id: 'guide-runtime' }, 15000])
  expect(screen.queryByRole('button', { name: 'Notion' })).toBeNull()
  expect(screen.queryByRole('button', { name: 'Stripe' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: /Gmail$/ }))
  fireEvent.click(screen.getByRole('button', { name: /Google Calendar$/ }))
  fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
  expect($onboardingAnswers.get().connectors).toEqual(['gmail', 'googlecalendar'])
  expect(submit).toHaveBeenCalledWith({
    text: '[setup] apps I use, not connected yet: Gmail, Google Calendar',
    displayKind: 'hidden',
    target: 'tile:guide',
    surfaceId: 'guide-composer'
  })
})

it.each(['stored', 'runtime', 'both'])('allows continuing with a missing %s id and probes when it arrives', async missing => {
  if (missing !== 'runtime') {
    $storedId.set(null)
  }
  if (missing !== 'stored') {
    $runtimeId.set(null)
  }
  const catalog = deferred<unknown>()
  gateway.rpc.mockReturnValue(catalog.promise)
  render(<SessionViewProvider value={view}><ConnectorsCard attrs={{}} locked={false} /></SessionViewProvider>)

  expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Continue' }).disabled).toBe(false)
  expect(screen.getByText("Connections aren't reachable right now, so this can wait.")).toBeTruthy()
  expect(gateway.rpc).not.toHaveBeenCalled()

  await act(async () => {
    $storedId.set('guide')
    $runtimeId.set('guide-runtime')
  })
  expect(gateway.rpc).toHaveBeenCalledTimes(1)
  expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Continue' }).disabled).toBe(true)
  await act(async () => catalog.resolve({ available: true, connectors: [{ connector: 'gmail', enabled: true }] }))
  expect(screen.getByRole('button', { name: /Gmail$/ })).toBeTruthy()
  expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Continue' }).disabled).toBe(false)
})

it('allows continuing when the real gateway request times out after 15 seconds', async () => {
  vi.useFakeTimers()
  const socket = new class extends EventTarget {
    readyState = 1
    send = vi.fn()
    close() { this.dispatchEvent(new Event('close')) }
  }()
  vi.stubGlobal('WebSocket', Object.assign(vi.fn(function () { return socket }), { OPEN: 1 }))
  const client = new HermesGateway()
  const connected = client.connect('ws://catalog.example')
  socket.dispatchEvent(new Event('open'))
  await connected
  setPrimaryGateway(client, 'hermes-setup')
  setPrimaryGatewayConnectionId('remote')

  await act(async () => {
    render(
      <SessionViewProvider value={view}>
        <div data-composer-surface-id="guide-composer" data-composer-target="tile:guide">
          <ConnectorsCard attrs={{}} locked={false} />
        </div>
      </SessionViewProvider>
    )
  })
  expect(socket.send).toHaveBeenCalledTimes(1)
  expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Continue' }).disabled).toBe(true)
  await act(async () => vi.advanceTimersByTimeAsync(15000))
  expect(screen.getByText("Connections aren't reachable right now, so this can wait.")).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
  expect(submit).toHaveBeenCalledWith(expect.objectContaining({
    text: '[setup] apps I use: none for now (connections unreachable)'
  }))
  client.close()
})

it('continues without stale picks when the gateway cannot supply any offered app', async () => {
  for (const result of [
    { available: false, connectors: [] },
    { available: true, connectors: [{ connector: 'stripe_mcp', enabled: true }] },
    new Error('Gateway offline')
  ]) {
    setOnboardingAnswers({ connectors: ['gmail'] })
    gateway.rpc.mockImplementation(() =>
      result instanceof Error ? Promise.reject(result) : Promise.resolve(result)
    )
    await act(async () => {
      render(
        <SessionViewProvider value={view}>
          <div data-composer-surface-id="guide-composer" data-composer-target="tile:guide">
            <ConnectorsCard attrs={{}} locked={false} />
          </div>
        </SessionViewProvider>
      )
    })
    expect(screen.queryByRole('button', { name: /Gmail$/ })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
    expect($onboardingAnswers.get().connectors).toEqual([])
    expect(submit).toHaveBeenLastCalledWith({
      text: '[setup] apps I use: none for now (connections unreachable)',
      displayKind: 'hidden',
      target: 'tile:guide',
      surfaceId: 'guide-composer'
    })
    cleanup()
  }
})
