import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// This integration harness mounts the host's real dispatcher and registry;
// plugin runtime code continues to use only the SDK.
// eslint-disable-next-line no-restricted-imports
import { usePaletteContributions } from '@/app/command-palette/contrib'
// eslint-disable-next-line no-restricted-imports
import { useKeybinds } from '@/app/hooks/use-keybinds'
// eslint-disable-next-line no-restricted-imports
import { createPluginContext } from '@/contrib/plugin'
// eslint-disable-next-line no-restricted-imports
import { setRuntimeI18nLocale } from '@/i18n/runtime'
// eslint-disable-next-line no-restricted-imports
import { formatCombo } from '@/lib/keybinds/combo'
// eslint-disable-next-line no-restricted-imports
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
// eslint-disable-next-line no-restricted-imports
import { isMacPlatform } from '@/lib/platform'
// eslint-disable-next-line no-restricted-imports
import { bindingsFor, conflictsFor, resetAllBindings, setBinding } from '@/store/keybinds'

import { $boardSlug } from './api'
import { KANBAN_LOCALES } from './i18n'
import plugin from './plugin'
import { $newTaskLane } from './ui'

// Registration and keyboard dispatch are real; no board socket or theme sync.
vi.mock('./api', async importOriginal => ({
  ...(await importOriginal<object>()),
  bindApi: () => () => {}
}))
vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark', setMode: vi.fn() })
}))

const disposers: Array<() => void> = []
const rest = vi.fn()

const deps = {
  archiveSelectedSession: vi.fn(),
  openNewSessionTab: vi.fn(),
  requestGateway: vi.fn(),
  startFreshSession: vi.fn(),
  toggleCommandCenter: vi.fn(),
  toggleSelectedPin: vi.fn()
}

function enable() {
  const ctx = createPluginContext('kanban', dispose => disposers.push(dispose))
  ctx.rest = rest
  plugin.register(ctx)
}

function disable() {
  disposers.splice(0).forEach(dispose => dispose())
}

function PaletteButton({ action, label, run }: { action?: string; label: string; run: () => void }) {
  const hint = useKeybindHint(action ?? '')

  return (
    <button onClick={run} type="button">
      {label}
      {hint && <kbd>{hint}</kbd>}
    </button>
  )
}

function Shell() {
  useKeybinds(deps)
  const palette = usePaletteContributions()

  return (
    <>
      <textarea aria-label="Composer" />
      {palette.map(item => (
        <PaletteButton action={item.action} key={item.key} label={item.label} run={item.run} />
      ))}
    </>
  )
}

function press(key: string, target: EventTarget = window) {
  const event = new KeyboardEvent('keydown', {
    altKey: true,
    bubbles: true,
    cancelable: true,
    code: `Key${key.toUpperCase()}`,
    ctrlKey: !isMacPlatform(),
    key,
    metaKey: isMacPlatform()
  })

  act(() => target.dispatchEvent(event))

  return event
}

beforeEach(() => {
  resetAllBindings()
  vi.clearAllMocks()
  setRuntimeI18nLocale('en')
  window.location.hash = '#/chat'
  $boardSlug.set('remembered-board')
  $newTaskLane.set(null)
  render(
    <MemoryRouter>
      <Shell />
    </MemoryRouter>
  )
})

afterEach(() => {
  cleanup()
  disable()
  resetAllBindings()
  $boardSlug.set('')
  $newTaskLane.set(null)
  window.location.hash = ''
})

it('opens the remembered board from the keyboard and palette without requesting a new task or backend change', () => {
  act(enable)
  // This reaches the mounted app dispatcher from an editable chat target.
  expect(press('k', screen.getByRole('textbox', { name: 'Composer' })).defaultPrevented).toBe(true)
  expect(window.location.hash).toBe('#/kanban')
  expect(conflictsFor('kanban.open', 'mod+alt+k')).toEqual([])
  const button = screen.getByRole('button', { name: new RegExp(KANBAN_LOCALES.en!.openBoard as string) })
  expect(button.querySelector('kbd')?.textContent).toBe(formatCombo('mod+alt+k'))
  expect($newTaskLane.get()).toBeNull()
  expect($boardSlug.get()).toBe('remembered-board')

  window.location.hash = '#/chat'
  fireEvent.click(button)
  expect(window.location.hash).toBe('#/kanban')
  expect($newTaskLane.get()).toBeNull()
  expect($boardSlug.get()).toBe('remembered-board')
  expect(rest).not.toHaveBeenCalled()
  expect(deps.requestGateway).not.toHaveBeenCalled()
  expect(deps.startFreshSession).not.toHaveBeenCalled()

  // The existing create shortcut remains a distinct, opt-in action.
  press('n')
  expect($newTaskLane.get()).toBe('triage')
})

it('keeps an existing new-task override on the new default chord and reports the conflict', () => {
  act(enable)
  act(() => setBinding('kanban.newTask', ['mod+alt+k']))
  act(disable)
  act(enable)

  expect(bindingsFor('kanban.newTask')).toEqual(['mod+alt+k'])
  expect(conflictsFor('kanban.open', 'mod+alt+k')).toContain('kanban.newTask')
  press('k')
  expect($newTaskLane.get()).toBe('triage')
})

it('honors rebinds and unbinding, and removes dispatch and palette access while disabled', () => {
  // Disabled at startup: the shortcut must not activate Kanban implicitly.
  expect(press('k').defaultPrevented).toBe(false)
  expect(window.location.hash).toBe('#/chat')
  expect(screen.queryByRole('button')).toBeNull()

  act(enable)
  act(() => setBinding('kanban.open', ['mod+alt+y']))
  expect(bindingsFor('kanban.open')).toEqual(['mod+alt+y'])
  expect(screen.getByText(formatCombo('mod+alt+y'))).toBeTruthy()
  expect(press('k').defaultPrevented).toBe(false)
  expect(window.location.hash).toBe('#/chat')
  expect(press('y').defaultPrevented).toBe(true)
  expect(window.location.hash).toBe('#/kanban')

  act(disable)
  window.location.hash = '#/chat'
  expect(press('y').defaultPrevented).toBe(false)
  expect(screen.queryByRole('button')).toBeNull()
  expect(window.location.hash).toBe('#/chat')

  act(enable)
  expect(screen.getByText(formatCombo('mod+alt+y'))).toBeTruthy()
  expect(press('y').defaultPrevented).toBe(true)
  expect(window.location.hash).toBe('#/kanban')

  act(() => setBinding('kanban.open', []))
  act(disable)
  act(enable)
  window.location.hash = '#/chat'
  expect(press('k').defaultPrevented).toBe(false)
  expect(press('y').defaultPrevented).toBe(false)
  expect(window.location.hash).toBe('#/chat')
  const button = screen.getByRole('button', { name: KANBAN_LOCALES.en!.openBoard as string })
  expect(button.querySelector('kbd')).toBeNull()
  fireEvent.click(button)
  expect(window.location.hash).toBe('#/kanban')
  expect($newTaskLane.get()).toBeNull()
})
