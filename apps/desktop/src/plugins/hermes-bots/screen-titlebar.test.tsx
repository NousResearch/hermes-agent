/**
 * Screen titlebar entry contract: exactly one always-on titleBar.right
 * contribution; clicking it opens the focused chat's bot screen as a global
 * workspace pane docked right (not a main-area tab); the button tracks the
 * pane's visibility and a second click closes it; with no focused bot chat
 * the pane falls back to the window's active profile; the registration
 * disposer unregisters the button and closes an open pane.
 */
import type * as HermesSdk from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/** The no-inline-render rule's prescription, minus the app-only import
 *  (plugin files may import only @hermes/plugin-sdk and react): mount the
 *  render callback as a component so its hooks belong to it, not to this
 *  test's host tree. */
function MountRender({ render }: { render: () => ReactNode }) {
  return createElement(render)
}

interface TestSdk {
  __test: {
    activeProfileAtom: { get: () => string; set: (value: string) => void }
    closeWorkspace: ReturnType<typeof vi.fn>
    focusedOwnerAtom: {
      get: () => unknown
      set: (value: { authoritative?: boolean; connectionId?: string; profile?: string } | null) => void
    }
    openWorkspace: ReturnType<typeof vi.fn>
    undismissPane: ReturnType<typeof vi.fn>
    visibilityAtom: { get: () => boolean; set: (value: boolean) => void }
  }
  host: {
    notify: ReturnType<typeof vi.fn>
    openWorkspace: ReturnType<typeof vi.fn>
    paneVisibility: (paneId: string) => unknown
    undismissPane: ReturnType<typeof vi.fn>
  }
}

let contributions: { area: string; id: string; order?: number; render: unknown }[] = []

const register = (contribution: { area: string; id: string; order?: number; render: unknown }) => {
  contributions.push(contribution)

  return () => {
    contributions = contributions.filter(entry => entry !== contribution)
  }
}

const pluginCtx = { register } as unknown as PluginContext

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { atom } = await import('nanostores')

  const visibilityAtom = atom(false)
  const focusedOwnerAtom = atom<{ authoritative?: boolean; connectionId?: string; profile?: string } | null>(null)
  const activeProfileAtom = atom('default')
  const connectionIdAtom = atom('local')
  const closeWorkspace = vi.fn()
  const undismissPane = vi.fn()
  const notify = vi.fn()

  const openWorkspace = vi.fn((_key: string, _options: unknown) => {
    visibilityAtom.set(true)

    return () => {
      visibilityAtom.set(false)
      closeWorkspace()
    }
  })

  return {
    ...sdk,
    host: {
      ...sdk.host,
      notify,
      openWorkspace,
      paneVisibility: (_paneId: string) => visibilityAtom,
      state: {
        connectionId: connectionIdAtom,
        focusedSessionOwner: focusedOwnerAtom,
        profile: activeProfileAtom
      },
      undismissPane
    },
    translateNow: (key: string) => (key === 'screen.title' ? 'Screen' : String(key)),
    __test: { activeProfileAtom, closeWorkspace, focusedOwnerAtom, openWorkspace, undismissPane, visibilityAtom }
  }
})

vi.mock('./screen-pane', () => ({
  BotScreenPane: ({ bot }: { bot: { name: string } }) => <div data-bot={bot.name} data-testid="bot-screen-pane" />
}))

vi.mock('./i18n', () => ({
  useBots: () => ({
    screen: {
      collapsePanel: 'Collapse Screen panel',
      openFailed: 'Could not open the Screen panel.',
      openPanel: 'Open Screen panel'
    }
  })
}))

import { registerScreenTitlebar, resetScreenTitlebar } from './screen-titlebar'

const mod = (await import('@hermes/plugin-sdk')) as unknown as TestSdk

const { activeProfileAtom, closeWorkspace, focusedOwnerAtom, openWorkspace, undismissPane, visibilityAtom } = mod.__test

/** The focused-owner shape the SDK atom carries on newer desktops. */
const setFocusedOwner = (profile: string | null, connectionId = 'local') =>
  focusedOwnerAtom.set(profile ? { authoritative: true, connectionId, profile } : null)

function renderTitlebar() {
  const contribution = contributions[0]

  return render(<MountRender render={contribution.render as () => ReactNode} />)
}

async function openPane() {
  await act(async () => {
    fireEvent.click(screen.getAllByRole('button', { name: 'Open Screen panel' })[0])
  })

  const [key, options] = openWorkspace.mock.calls.at(-1)!

  return { key, options: options as { render: () => ReactNode } }
}

beforeEach(() => {
  contributions = []
  vi.clearAllMocks()
  // The toggle's open state is module-level and outlives unmounts; a pane a
  // previous test left open would turn the next click into a collapse.
  resetScreenTitlebar()
  visibilityAtom.set(false)
  activeProfileAtom.set('default')
  setFocusedOwner('parker')
  registerScreenTitlebar(pluginCtx)
})

afterEach(() => {
  cleanup()
})

describe('screen titlebar entry', () => {
  it('registers exactly one titleBar.right contribution with a renderable button', () => {
    expect(contributions).toHaveLength(1)
    expect(contributions[0].id).toBe('screen')
    expect(contributions[0].area).toBe('titleBar.right')
    expect(typeof contributions[0].render).toBe('function')

    renderTitlebar()
    expect(screen.getByRole('button', { name: 'Open Screen panel' })).toBeTruthy()
  })

  it('opens the focused chat bot screen docked right, and the button flips to collapse', async () => {
    renderTitlebar()

    const { key, options } = await openPane()

    expect(key).toBe('hermes-bots:screen-global')
    expect(options).toMatchObject({ dock: { pane: 'workspace', pos: 'right' }, title: 'Screen' })
    expect(undismissPane).toHaveBeenCalledWith('plugin-workspace:hermes-bots:screen-global')

    render(<MountRender render={options.render} />)
    expect(screen.getByTestId('bot-screen-pane').getAttribute('data-bot')).toBe('parker')

    renderTitlebar()
    expect(screen.getAllByRole('button', { name: 'Collapse Screen panel' }).length).toBeGreaterThan(0)
  })

  it('a second click closes the pane and the button flips back', async () => {
    renderTitlebar()
    await openPane()

    const collapse = screen.getByRole('button', { name: 'Collapse Screen panel' })

    await act(async () => {
      fireEvent.click(collapse)
    })

    expect(closeWorkspace).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: 'Open Screen panel' })).toBeTruthy()
  })

  it('with no focused bot chat the pane falls back to the window active profile', async () => {
    setFocusedOwner(null)
    renderTitlebar()

    const { options } = await openPane()

    render(<MountRender render={options.render} />)
    expect(screen.getByTestId('bot-screen-pane').getAttribute('data-bot')).toBe('default')
  })

  it('uses the roster row when the roster knows the focused bot', async () => {
    const { $lastRoster } = await import('./data')

    $lastRoster.set([{ name: 'parker', sourceScoped: true, connectionId: 'local', connectionKind: 'local' } as never])
    renderTitlebar()

    const { options } = await openPane()

    render(<MountRender render={options.render} />)
    expect(screen.getByTestId('bot-screen-pane').getAttribute('data-bot')).toBe('parker')
  })

  it('the disposer unregisters the button and closes an open pane', async () => {
    const disposer = registerScreenTitlebar(pluginCtx)
    expect(contributions).toHaveLength(2) // the beforeEach registration + this one

    renderTitlebar()
    await openPane()

    disposer()

    expect(contributions).toHaveLength(1)
    expect(closeWorkspace).toHaveBeenCalledTimes(1)

    cleanup()
    expect(screen.queryByRole('button', { name: 'Collapse Screen panel' })).toBeNull()
  })
})
