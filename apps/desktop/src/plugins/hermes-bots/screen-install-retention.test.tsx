/**
 * An INACTIVE registry-routed bot's pooled socket is disposed by the SDK as soon
 * as its request count hits zero — so `display.install.log/done` (and the
 * pane's `display.lease` events) would never arrive. The install card must hold
 * a retention from before `display.install` until the done event lands.
 */

import { act, fireEvent, render } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type { RosterRow } from './types'

const calls = vi.hoisted(() => [] as string[])

vi.mock('@hermes/plugin-sdk', async () => {
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props}>{children}</button>,
    Codicon: () => null,
    GlyphSpinner: () => null,
    resolveSiblingWsUrl: vi.fn(),
    host: {
      onEvent: onGatewayEvent,
      requestProfile: vi.fn(async (_route: unknown, method: string) => {
        calls.push(`request:${method}`)

        return {}
      }),
      retainProfile: vi.fn(async () => {
        calls.push('retain')

        return () => {
          calls.push('release')
        }
      })
    }
  }
})
vi.mock('./routing', () => ({
  botConnectionRoute: () => ({ connectionId: 'host-a', profile: 'ops', targetProfile: 'ops' })
}))
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      notInstalledTitle: 'Missing',
      notInstalledBody: 'Body',
      installHint: 'Hint',
      install: 'Install on host',
      installing: 'Installing',
      installCancelled: 'Cancelled',
      installFailed: 'Failed',
      noPackageManager: 'None'
    }
  })
}))

// eslint-disable-next-line no-restricted-imports
import { emitGatewayEvent } from '../../contrib/events'

import { ScreenInstallCard } from './screen-install'

const bot: RosterRow = { name: 'ops', sourceScoped: true, connectionId: 'host-a', connectionKind: 'remote' }

const status: DisplayStatus = {
  profile: 'ops',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: false,
  missing: ['tigervnc'],
  running: false,
  pid: null,
  display: null,
  socket: null,
  geometry: '1440x900',
  install_command: 'sudo apt-get install -y tigervnc-standalone-server',
  lease: { holder: 'agent', viewer_id: null, pending_handoff: null, since: 1, reason: '' }
}

beforeEach(() => {
  calls.length = 0
})

it('retains the bot socket before display.install and releases it when the done event lands', async () => {
  const onInstalled = vi.fn()
  const view = render(<ScreenInstallCard bot={bot} onInstalled={onInstalled} status={status} />)

  await act(async () => {
    fireEvent.click(view.getByText('Install on host'))
  })
  expect(calls).toEqual(['retain', 'request:display.install'])

  act(() =>
    emitGatewayEvent({
      type: 'display.install.done',
      connectionId: 'host-a',
      profile: 'ops',
      payload: { profile_key: status.profile_key, code: 0, status: { ...status, installed: true } }
    })
  )
  expect(calls).toEqual(['retain', 'request:display.install', 'release'])
  expect(onInstalled).toHaveBeenCalledTimes(1)
  view.unmount()
  // Unmount after done must not double-release.
  expect(calls.filter(call => call === 'release')).toHaveLength(1)
})
