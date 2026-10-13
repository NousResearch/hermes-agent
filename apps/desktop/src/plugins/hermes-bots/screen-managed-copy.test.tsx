/**
 * A `display.*` method-not-found from a Portal-managed (Hermes Cloud) backend must not be
 * rendered as "Update the bot's Hermes": the user cannot update a managed release, and the
 * managed Cloud tab already reports it is on the latest release (#120852). A self-upgradable
 * (git/remote) backend keeps the update instruction.
 *
 * Likewise, missing screen packages on a managed backend are a property of its size, not
 * something to install: only a self-managed host gets the install card.
 */
import { act, render } from '@testing-library/react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { DisplayStatus } from './screen-connection'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { onGatewayEvent } = await import('../../contrib/events')

  return {
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
    host: { onEvent: onGatewayEvent, requestProfile: vi.fn() }
  }
})
vi.mock('./routing', () => {
  const route = { connectionId: 'cloud-a', mode: 'remote', profile: 'default', targetProfile: 'default' }

  return { botConnectionRoute: () => route, resolveBotConnectionRoute: () => ({ status: 'resolved', route }) }
})
vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))
vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      title: 'Screen',
      portalUnavailable: 'Update the bot’s Hermes to use Screen',
      portalUnavailableManaged: 'Screen is not available on this managed Hermes release yet',
      unavailableTitle: 'Screen needs a newer Hermes',
      managedNotIncludedTitle: 'Screen isn’t included on this bot',
      managedNotIncludedBody: 'Hermes Cloud includes Screen on bots with 8 GB of memory or more.',
      notInstalledTitle: 'Screen packages missing',
      install: 'Install on host'
    }
  })
}))

import { host } from '@hermes/plugin-sdk'

import { BotScreenPane } from './screen-pane'
import { $screenState } from './screen-state'

const cloudBot: RosterRow = { name: 'default', connectionId: 'cloud-a', connectionKind: 'cloud' }
const gitBot: RosterRow = { name: 'default', connectionId: 'host-a', connectionKind: 'remote' }

beforeEach(() => {
  $screenState.set({})
  vi.mocked(host.requestProfile).mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
})

const methodNotFound = () =>
  Promise.reject(Object.assign(new Error('Method not found: display.status'), { code: -32601 }))

it('a managed Cloud backend gets the managed-release copy, not a self-update instruction', async () => {
  vi.mocked(host.requestProfile).mockImplementation(() => methodNotFound())
  const view = render(<BotScreenPane bot={cloudBot} />)
  await act(async () => {})

  expect(view.getByText('Screen is not available on this managed Hermes release yet')).toBeTruthy()
  expect(view.queryByText('Update the bot’s Hermes to use Screen')).toBeNull()
  view.unmount()
})

it('a self-upgradable backend keeps the update instruction', async () => {
  vi.mocked(host.requestProfile).mockImplementation(() => methodNotFound())
  const view = render(<BotScreenPane bot={gitBot} />)
  await act(async () => {})

  expect(view.getByText('Update the bot’s Hermes to use Screen')).toBeTruthy()
  expect(view.queryByText('Screen is not available on this managed Hermes release yet')).toBeNull()
  view.unmount()
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

it('a managed Cloud backend without screen packages says Screen is not included, with no install', async () => {
  vi.mocked(host.requestProfile).mockResolvedValue(packagesMissing)
  const view = render(<BotScreenPane bot={cloudBot} />)
  await act(async () => {})

  expect(view.getByText('Screen isn’t included on this bot')).toBeTruthy()
  expect(view.queryByText('Install on host')).toBeNull()
  view.unmount()
})

it('a self-managed host without screen packages keeps the install card', async () => {
  vi.mocked(host.requestProfile).mockResolvedValue(packagesMissing)
  const view = render(<BotScreenPane bot={gitBot} />)
  await act(async () => {})

  expect(view.getByText('Install on host')).toBeTruthy()
  expect(view.queryByText('Screen isn’t included on this bot')).toBeNull()
  view.unmount()
})
