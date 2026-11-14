/**
 * Missing screen packages on a managed backend are a property of its size, not
 * something to install: only a self-managed host gets the install card.
 */
import { act, render } from '@testing-library/react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'
import type { DisplayStatus } from './screen-connection'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  // The SDK mock is hoisted before static imports initialize.
  const { atom } = await import('nanostores')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
    atom,
    Button: (props: ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props} />,
    Codicon: () => null,
    GlyphSpinner: () => null,
    Tip: ({ children }: { children: ReactNode }) => <>{children}</>,
    EmptyState: ({ title, description }: { title: string; description: string }) => (
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
    ),
    useValue: useStore,
    usePluginI18n: () => translateBotsIn('en'),
    host: { onEvent: onGatewayEvent, requestProfile: vi.fn() }
  }
})
vi.mock('./routing', () => {
  const route = { connectionId: 'cloud-a', mode: 'remote', profile: 'default', targetProfile: 'default' }

  return { botConnectionRoute: () => route, resolveBotConnectionRoute: () => ({ status: 'resolved', route }) }
})
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))

import { host } from '@hermes/plugin-sdk'

import { BotScreenPane } from './screen-pane'
import { $screenState } from './screen-state'

const text = translateBotsIn('en')

const cloudBot: RosterRow = { name: 'default', connectionId: 'cloud-a', connectionKind: 'cloud' }
const gitBot: RosterRow = { name: 'default', connectionId: 'host-a', connectionKind: 'remote' }

beforeEach(() => {
  $screenState.set({})
  vi.mocked(host.requestProfile).mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
})

const packagesMissing: DisplayStatus = {
  profile: 'default',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: false,
  missing: ['tigervnc-standalone-server'],
  running: false,
  pid: null,
  display: null,
  socket: null,
  geometry: '1440x900',
  install_command: 'sudo apt-get install -y tigervnc-standalone-server',
  lease: { holder: 'agent', viewer_id: null, viewer_hash: null, since: 1, reason: '', epoch: 0 }
}

it('a managed Cloud backend without screen packages offers no install', async () => {
  vi.mocked(host.requestProfile).mockResolvedValue(packagesMissing)
  const view = render(<BotScreenPane bot={cloudBot} />)
  await act(async () => {})

  expect(view.queryByRole('button', { name: text('screen.install') })).toBeNull()
  view.unmount()
})

it('a self-managed host without screen packages keeps the install card', async () => {
  vi.mocked(host.requestProfile).mockResolvedValue(packagesMissing)
  const view = render(<BotScreenPane bot={gitBot} />)
  await act(async () => {})

  expect(view.getByRole('button', { name: text('screen.install') })).toBeTruthy()
  view.unmount()
})
