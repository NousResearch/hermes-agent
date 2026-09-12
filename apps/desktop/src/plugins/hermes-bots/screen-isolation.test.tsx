import { act, fireEvent, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    Codicon: () => null,
    useValue: useStore,
    resolveSiblingWsUrl: vi.fn(),
    host: { onEvent: onGatewayEvent, requestProfile: vi.fn() }
  }
})
vi.mock('./data', async () => {
  const { atom } = await import('nanostores')

  return {
    $lastRoster: atom<RosterRow[]>([]),
    botSelectionKey: (bot: RosterRow) =>
      bot.sourceScoped || bot.remoteSource ? `${bot.connectionId}::${bot.name}` : bot.name
  }
})
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      portalTitle: 'Screen',
      portalWatching: 'Live',
      portalYouControl: 'You control',
      portalOtherControls: 'Other viewer',
      heroOpenLive: 'Open live',
      heroStale: 'Last seen',
      heroConnecting: 'Connecting'
    }
  })
}))
vi.mock('./screen-open', () => ({ openBotScreen: vi.fn() }))

import { host } from '@hermes/plugin-sdk'

// Exercise the real event bus in this integration test, not a copied dispatcher.
// eslint-disable-next-line no-restricted-imports
import { emitGatewayEvent } from '../../contrib/events'

import { $lastRoster } from './data'
import { type DisplayStatus, VIEWER_ID } from './screen-connection'
import { ScreenHero } from './screen-hero'
import { openBotScreen } from './screen-open'
import { ProfileGroupScreenPortal, useScreenPortalState } from './screen-portal'
import { $screenState, setScreenStatus } from './screen-state'

const botA: RosterRow = { name: 'default', sourceScoped: true, connectionId: 'host-a', connectionKind: 'remote' }
const botB: RosterRow = { ...botA, connectionId: 'host-b' }

const status: DisplayStatus = {
  profile: 'default',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: true,
  missing: [],
  running: true,
  pid: 42,
  display: ':20',
  socket: '/tmp/rfb.sock',
  geometry: '1440x900',
  install_command: null,
  lease: { holder: 'agent', viewer_id: null, pending_handoff: null, since: 1, reason: '' }
}

beforeEach(() => {
  $screenState.set({})
  $lastRoster.set([])
  vi.mocked(host.requestProfile).mockReset()
  vi.mocked(openBotScreen).mockClear()
  vi.spyOn(globalThis.document, 'hidden', 'get').mockReturnValue(false)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

it('applies lease events only from the owning host even when profile paths match', () => {
  setScreenStatus(botA, status)
  const view = renderHook(() => useScreenPortalState(botA))
  const human = { ...status.lease, holder: 'human' as const, viewer_id: VIEWER_ID }

  const emit = (connectionId: string, profileKey = status.profile_key) =>
    act(() =>
      emitGatewayEvent({
        type: 'display.lease',
        connectionId,
        profile: 'default',
        payload: { profile_key: profileKey, lease: human }
      })
    )

  emit('host-b')
  expect(view.result.current.lease?.holder).toBe('agent')
  emit('host-a', '/another/profile')
  expect(view.result.current.lease?.holder).toBe('agent')
  emit('host-a')
  expect(view.result.current.lease).toEqual(human)
  view.unmount()
})

it('does not match a legacy profile group to a same-named remote bot', () => {
  const legacy: RosterRow = { name: 'default' }
  $lastRoster.set([botB, legacy])
  setScreenStatus(botB, status)
  setScreenStatus(legacy, status)
  const view = render(<ProfileGroupScreenPortal route={{ connectionId: null, profile: 'default' }} />)

  fireEvent.click(view.getByRole('button'))
  expect(openBotScreen).toHaveBeenCalledWith(legacy, null)
  view.rerender(<ProfileGroupScreenPortal route={{ connectionId: 'host-b', profile: 'default' }} />)
  fireEvent.click(view.getByRole('button'))
  expect(openBotScreen).toHaveBeenLastCalledWith(botB, null)
  view.unmount()
})

it.each(['pending', 'rejected'])('never displays host A pixels under host B while B is %s', async state => {
  setScreenStatus(botA, status)
  setScreenStatus(botB, status)
  vi.mocked(host.requestProfile)
    .mockResolvedValueOnce({ data_url: 'data:image/jpeg;base64,HOST_A' })
    .mockImplementationOnce(() => (state === 'pending' ? new Promise(() => {}) : Promise.reject(new Error('offline'))))
  const view = render(<ScreenHero bot={botA} />)
  await act(async () => {})
  expect(view.container.querySelector('img')?.getAttribute('src')).toContain('HOST_A')

  view.rerender(<ScreenHero bot={botB} />)
  await act(async () => {})
  expect(view.container.querySelector('img')).toBeNull()
  view.unmount()
})

it('discards a late thumbnail from the previous owner and retains a same-owner frame on refresh failure', async () => {
  vi.useFakeTimers()
  setScreenStatus(botA, status)
  setScreenStatus(botB, status)
  let finishA!: (value: unknown) => void
  vi.mocked(host.requestProfile)
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finishA = resolve
        })
    )
    .mockResolvedValueOnce({ data_url: 'data:image/jpeg;base64,HOST_B' })
    .mockRejectedValue(new Error('offline'))
  const view = render(<ScreenHero bot={botA} />)
  view.rerender(<ScreenHero bot={botB} />)
  await act(async () => {})
  await act(async () => {
    finishA({ data_url: 'data:image/jpeg;base64,HOST_A' })
  })
  expect(view.container.querySelector('img')?.getAttribute('src')).toContain('HOST_B')
  await act(async () => {
    await vi.advanceTimersByTimeAsync(12_000)
  })
  expect(view.container.querySelector('img')?.getAttribute('src')).toContain('HOST_B')
  expect(view.getByRole('button').getAttribute('aria-label')).toContain('Last seen')
  view.unmount()
})
