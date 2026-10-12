/**
 * A `display.*` method-not-found from a Portal-managed (Hermes Cloud) backend must not be
 * rendered as "Update the bot's Hermes": the user cannot update a managed release, and the
 * managed Cloud tab already reports it is on the latest release (#120852). A self-upgradable
 * (git/remote) backend keeps the update instruction.
 */
import { act, render } from '@testing-library/react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

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
      unavailableTitle: 'Screen needs a newer Hermes'
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
