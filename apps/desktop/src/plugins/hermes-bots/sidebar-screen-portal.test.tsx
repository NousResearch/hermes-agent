import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { useStore } = await import('@nanostores/react')
  const { desktopSettings } = await import('../../sdk/settings')

  return {
    Codicon: () => null,
    useValue: useStore,
    resolveSiblingWsUrl: vi.fn(),
    host: {
      onEvent: vi.fn(() => vi.fn()),
      requestProfile: vi.fn(() => new Promise(() => {})),
      settings: desktopSettings
    }
  }
})
vi.mock('./data', async () => {
  const { atom } = await import('nanostores')

  return { $lastRoster: atom<RosterRow[]>([]), botSelectionKey: (bot: RosterRow) => bot.name }
})
vi.mock('./i18n', () => ({ useBots: () => ({ screen: { portalTitle: 'Screen', portalOpen: 'Open' } }) }))
vi.mock('./screen-open', () => ({ openBotScreen: vi.fn() }))

import { host } from '@hermes/plugin-sdk'

import { openBotScreen } from './screen-open'
import { ProfileGroupScreenPortal, ScreenPortal } from './screen-portal'
import { $screenState } from './screen-state'

beforeEach(() => {
  $screenState.set({})
  vi.clearAllMocks()
})
afterEach(() => {
  cleanup()
  host.settings.set('sidebar.screenPortalVisible', true)
})

it('shows the Sessions Screen portal by default and opens its owner screen', () => {
  render(<ProfileGroupScreenPortal route={{ connectionId: null, profile: 'ops' }} />)
  fireEvent.click(screen.getByRole('button', { name: /Screen/ }))
  expect(openBotScreen).toHaveBeenCalledWith(expect.objectContaining({ name: 'ops' }), null)
  expect(host.requestProfile).toHaveBeenCalledWith('ops', 'display.status', { profile: 'ops' })
})

it('hides and restores the Sessions portal live without disabling the Bots portal', () => {
  render(<ProfileGroupScreenPortal route={{ connectionId: null, profile: 'ops' }} />)
  expect(screen.queryByRole('button', { name: /Screen/ })).not.toBeNull()

  act(() => host.settings.set('sidebar.screenPortalVisible', false))
  expect(screen.queryByRole('button', { name: /Screen/ })).toBeNull()

  act(() => host.settings.set('sidebar.screenPortalVisible', true))
  expect(screen.queryByRole('button', { name: /Screen/ })).not.toBeNull()

  cleanup()
  act(() => host.settings.set('sidebar.screenPortalVisible', false))
  vi.clearAllMocks()
  render(<ProfileGroupScreenPortal route={{ connectionId: null, profile: 'ops' }} />)
  expect(screen.queryByRole('button', { name: /Screen/ })).toBeNull()
  expect(host.requestProfile).not.toHaveBeenCalled()
  expect(host.onEvent).not.toHaveBeenCalled()

  render(<ScreenPortal bot={{ name: 'ops' }} />)
  fireEvent.click(screen.getByRole('button', { name: /Screen/ }))
  expect(openBotScreen).toHaveBeenCalledWith(expect.objectContaining({ name: 'ops' }), null)
  expect(host.requestProfile).toHaveBeenCalledWith('ops', 'display.status', { profile: 'ops' })
})
