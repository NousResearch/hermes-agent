/**
 * A group header with no routable owner must not paint a Screen entry. `unknown` is the state
 * of a status request that has not landed — and a profile-only `display.*` request is refused on
 * a multi-source desktop, so for a group with no connection that request can never land. The
 * entry then stays on the sidebar forever as an empty button (`aria-label="屏幕: "`).
 */

import { render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    Codicon: () => null,
    useValue: useStore,
    resolveSiblingWsUrl: vi.fn(),
    host: {
      onEvent: vi.fn(onGatewayEvent),
      requestProfile: vi.fn(),
      openWorkspace: vi.fn(),
      notify: vi.fn(),
      revealPane: vi.fn()
    }
  }
})
vi.mock('./data', async () => {
  const { atom } = await import('nanostores')

  return { $lastRoster: atom<RosterRow[]>([]), botSelectionKey: (bot: RosterRow) => bot.name }
})
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      portalTitle: '屏幕',
      portalOpen: '打开',
      portalWatching: '直播 · 机器人控制中',
      portalYouControl: '直播 · 你在控制',
      portalOtherControls: '直播 · 其他查看者控制中',
      portalStopped: '已停止',
      portalNotInstalled: '主机未安装',
      portalUnsupported: '此主机不可用',
      portalUnavailable: '更新机器人的 Hermes 以使用屏幕',
      portalUnavailableManaged: '此托管 Hermes 版本尚不支持屏幕'
    }
  })
}))
vi.mock('./screen-open', () => ({ openBotScreen: vi.fn() }))

import { host } from '@hermes/plugin-sdk'

import { $lastRoster } from './data'
import type { DisplayStatus } from './screen-connection'
import { ProfileGroupScreenPortal, ScreenPortal } from './screen-portal'
import { $screenState, setScreenStatus } from './screen-state'

const bot: RosterRow = { name: 'ops' }

const stopped: DisplayStatus = {
  profile: 'ops',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: true,
  missing: [],
  running: false,
  pid: null,
  display: null,
  socket: null,
  geometry: '1440x900',
  install_command: null,
  lease: { holder: 'agent', viewer_id: null, viewer_hash: null, since: 1, reason: '', epoch: 0 }
}

/** A request that never settles: the portal stays in `unknown` for the whole test. */
const pending = () => new Promise<never>(() => undefined)

beforeEach(() => {
  $screenState.set({})
  $lastRoster.set([])
  vi.mocked(host.requestProfile).mockReset()
  vi.mocked(host.requestProfile).mockImplementation(() => pending() as Promise<never>)
})

afterEach(() => vi.restoreAllMocks())

it('paints nothing while the status is unknown', () => {
  const { container } = render(<ScreenPortal bot={bot} />)

  expect(container.querySelector('button')).toBeNull()
  expect(container.textContent).toBe('')
})

it('paints nothing for a group with no connection and no roster owner, and asks nothing', () => {
  const { container } = render(<ProfileGroupScreenPortal route={{ connectionId: null, profile: 'chatbot' }} />)

  expect(container.querySelector('button')).toBeNull()
  expect(host.requestProfile).not.toHaveBeenCalled()
})

it('settles a route-missing failure as unavailable instead of leaving the portal unknown', async () => {
  vi.mocked(host.requestProfile).mockRejectedValue(
    new Error(
      'Profile "ops" requires a route descriptor from host.profileRoutes(); profile-only routing is limited to legacy/local profiles.'
    )
  )

  const { container } = render(<ScreenPortal bot={bot} />)

  await waitFor(() => expect($screenState.get().ops?.unavailable).toBe(true))
  expect(container.querySelector('button')).toBeNull()
})

it('still paints the portal for a routable group once a status lands', () => {
  $lastRoster.set([
    {
      name: 'chatbot',
      sourceScoped: true,
      route: { connectionId: 'remote-laptop', mode: 'remote', profile: 'chatbot', targetProfile: 'chatbot' }
    }
  ])
  setScreenStatus({ name: 'chatbot' }, stopped)

  const { container } = render(
    <ProfileGroupScreenPortal route={{ connectionId: 'remote-laptop', profile: 'chatbot' }} />
  )

  expect(container.querySelector('button')?.getAttribute('aria-label')).toBe('屏幕: 已停止')
})
