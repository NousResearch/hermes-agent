/**
 * The portal's one-shot `display.status` fetch is a network reply, not an
 * event: a snapshot it returns can predate a newer authoritative write that
 * landed while it was in flight. Like the pane's fetch, it must carry a request
 * token so the store drops the superseded answer instead of moving `running`
 * and `display` backwards (#110037).
 */

import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    Codicon: () => null,
    useValue: useStore,
    resolveSiblingWsUrl: vi.fn(),
    host: { onEvent: vi.fn(onGatewayEvent), requestProfile: vi.fn() }
  }
})
vi.mock('./data', async () => {
  const { atom } = await import('nanostores')

  return { $lastRoster: atom<RosterRow[]>([]), botSelectionKey: (bot: RosterRow) => bot.name }
})
vi.mock('./i18n', () => ({ useBots: () => ({ screen: {} }) }))
vi.mock('./screen-open', () => ({ openBotScreen: vi.fn() }))

import { host } from '@hermes/plugin-sdk'

import type { DisplayStatus } from './screen-connection'
import { useScreenPortalState } from './screen-portal'
import { $screenState, beginScreenStatusRequest, screenStateFor, setScreenStatus } from './screen-state'

const bot: RosterRow = { name: 'ops' }

const running: DisplayStatus = {
  profile: 'ops',
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
  lease: { holder: 'agent', viewer_id: null, viewer_hash: null, since: 1, reason: '', epoch: 0 }
}

const stopped: DisplayStatus = { ...running, running: false, pid: null, display: null, socket: null }

beforeEach(() => {
  $screenState.set({})
  vi.mocked(host.requestProfile).mockReset()
})

afterEach(() => vi.restoreAllMocks())

it('a portal status reply superseded by a newer status request does not move running/display backwards', async () => {
  let reply: (status: DisplayStatus) => void = () => undefined
  vi.mocked(host.requestProfile).mockImplementation(
    () =>
      new Promise<DisplayStatus>(resolve => {
        reply = resolve
      }) as Promise<never>
  )

  const view = renderHook(() => useScreenPortalState(bot))
  expect(host.requestProfile).toHaveBeenCalledWith('ops', 'display.status', {})

  // The pane opens while the portal's fetch is still in flight and asks again (newer request).
  const paneRequest = beginScreenStatusRequest(bot)

  // The portal's slower reply is a snapshot from before the stop; it must neither land as truth
  // nor invalidate the pane's newer request, whose reply then carries the stop.
  await act(async () => {
    reply(running)
    await Promise.resolve()
  })
  act(() => setScreenStatus(bot, stopped, paneRequest))

  const status = screenStateFor($screenState.get(), bot)?.status
  expect(status?.running).toBe(false)
  expect(status?.display).toBeNull()
  view.unmount()
})
