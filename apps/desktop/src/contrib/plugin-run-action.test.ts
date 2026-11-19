import { renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' as const, setMode: () => undefined })
}))

// The browser toggle is the handler the shortcut runs; spy on it so the test
// can tell "the plugin ran the same handler" from "something else happened".
const toggleBrowserTab = vi.hoisted(() => vi.fn())

vi.mock('@/store/preview', async importOriginal => ({
  ...(await importOriginal<object>()),
  toggleBrowserTab
}))

import { useKeybinds } from '@/app/hooks/use-keybinds'
import { createPluginContext } from '@/contrib/plugin'
import { KEYBIND_ACTION_IDS } from '@/lib/keybinds/actions'
import { isMacPlatform } from '@/lib/platform'
import { PLUGIN_APP_ACTIONS, type PluginAppActionId, type PluginContext, type PluginRunActionResult } from '@/sdk'
import { resetAllBindings, setBinding } from '@/store/keybinds'

const wrapper = ({ children }: { children: ReactNode }) => createElement(MemoryRouter, null, children)

const deps = {
  archiveSelectedSession: vi.fn(),
  openNewSessionTab: vi.fn(),
  requestGateway: <T>() => Promise.resolve(undefined as T),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

const pressModShift = (key: string) =>
  window.dispatchEvent(
    new KeyboardEvent('keydown', {
      bubbles: true,
      cancelable: true,
      code: `Key${key.toUpperCase()}`,
      ctrlKey: !isMacPlatform(),
      key,
      metaKey: isMacPlatform(),
      shiftKey: true
    })
  )

let unmount: (() => void) | undefined

beforeEach(() => {
  toggleBrowserTab.mockClear()
  Object.values(deps).forEach(fn => 'mockClear' in fn && fn.mockClear())
  resetAllBindings()
  unmount = renderHook(() => useKeybinds(deps), { wrapper }).unmount
})

afterEach(() => {
  unmount?.()
  unmount = undefined
  resetAllBindings()
  vi.restoreAllMocks()
})

describe('ctx.runAction — plugins run built-in app actions', () => {
  it('runs the same handler the keyboard shortcut runs', () => {
    const ctx = createPluginContext('browser-toggle')

    pressModShift('l')
    expect(toggleBrowserTab).toHaveBeenCalledTimes(1)

    expect(ctx.runAction('view.showBrowser')).toEqual({ ok: true })
    expect(toggleBrowserTab).toHaveBeenCalledTimes(2)
  })

  it('is unaffected by rebinding or unbinding the shortcut', () => {
    const ctx = createPluginContext('browser-toggle')

    setBinding('view.showBrowser', ['mod+alt+b'])

    // The old chord no longer toggles: a plugin faking Ctrl+Shift+L breaks here.
    pressModShift('l')
    expect(toggleBrowserTab).not.toHaveBeenCalled()

    expect(ctx.runAction('view.showBrowser')).toEqual({ ok: true })
    expect(toggleBrowserTab).toHaveBeenCalledTimes(1)

    setBinding('view.showBrowser', [])
    expect(ctx.runAction('view.showBrowser')).toEqual({ ok: true })
    expect(toggleBrowserTab).toHaveBeenCalledTimes(2)
  })

  it('refuses actions outside the allowlist without running them', () => {
    const ctx = createPluginContext('sneaky')
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const run = ctx.runAction as (id: string) => PluginRunActionResult

    expect(run('session.archive')).toMatchObject({ ok: false, reason: 'denied' })
    expect(run('profile.switch.1')).toMatchObject({ ok: false, reason: 'denied' })
    expect(run('composer.modelPicker')).toMatchObject({ ok: false, reason: 'denied' })
    expect(deps.archiveSelectedSession).not.toHaveBeenCalled()

    const unknown = run('app.quit')

    expect(unknown).toMatchObject({ ok: false, reason: 'unknown' })
    expect(unknown.ok ? '' : unknown.error).toContain("'app.quit'")
    expect(run(undefined as unknown as string)).toMatchObject({ ok: false, reason: 'unknown' })

    expect(toggleBrowserTab).not.toHaveBeenCalled()
    expect(warn).toHaveBeenCalledTimes(5)
    expect(String(warn.mock.calls[0]?.[0])).toContain('[plugin:sneaky]')
  })

  it('reports unavailable instead of throwing when the app shell is not mounted', () => {
    unmount?.()
    unmount = undefined
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    expect(createPluginContext('early').runAction('view.showBrowser')).toMatchObject({
      ok: false,
      reason: 'unavailable'
    })
    expect(toggleBrowserTab).not.toHaveBeenCalled()
  })

  it('lists exactly the allowlisted actions with labels', () => {
    const listed = createPluginContext('browser-toggle').listActions()

    expect(listed.map(a => a.id)).toEqual([...PLUGIN_APP_ACTIONS])
    expect(listed.find(a => a.id === 'view.showBrowser')).toMatchObject({ category: 'view', label: 'Toggle browser' })
    expect(listed.every(a => a.label && a.label !== a.id)).toBe(true)
  })

  it('allowlists only real built-in view/navigation actions', () => {
    const forbidden =
      /archive|togglePin|profile\.|modelPicker|reasoning|voice|dictate|Terminal|newWindow|openFolder|Worktree|toggleHud|closeTab/

    for (const id of PLUGIN_APP_ACTIONS) {
      expect(KEYBIND_ACTION_IDS).toContain(id)
      expect(id).not.toMatch(forbidden)
    }
  })

  it('exports the typed API from @hermes/plugin-sdk', () => {
    expectTypeOf<PluginAppActionId>().toEqualTypeOf<(typeof PLUGIN_APP_ACTIONS)[number]>()
    expectTypeOf<'view.showBrowser'>().toMatchTypeOf<PluginAppActionId>()
    expectTypeOf<'session.archive'>().not.toMatchTypeOf<PluginAppActionId>()
    expectTypeOf<PluginContext['runAction']>().parameter(0).toEqualTypeOf<PluginAppActionId>()
    expectTypeOf<PluginContext['runAction']>().returns.toEqualTypeOf<PluginRunActionResult>()
    expect(PLUGIN_APP_ACTIONS).toContain('view.showBrowser')
  })
})
