// @vitest-environment jsdom
import { act, render } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' as const, setMode: () => undefined })
}))

import { useKeybinds } from '@/app/hooks/use-keybinds'
import { $historyPosition } from '@/app/shell/history-position'
import { resetBinding, setBinding } from '@/store/keybinds'
import { $shellChoice } from '@/store/shell-variant'

import { $workspaceIsPage, NEW_CHAT_ROUTE } from '../routes'

let pathname = ''

function Harness() {
  pathname = useLocation().pathname
  useKeybinds({
    archiveSelectedSession: () => undefined,
    openNewSessionTab: () => undefined,
    requestGateway: <T,>() => Promise.resolve(undefined as T),
    startFreshSession: () => undefined,
    toggleCommandCenter: () => undefined,
    toggleSelectedPin: () => undefined
  })

  // jsdom has no isContentEditable, so a textarea stands in for CodeMirror's editable content.
  return <textarea data-testid="editor" />
}

const pressBack = (target: EventTarget) =>
  act(() => {
    target.dispatchEvent(
      new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code: 'BracketLeft', ctrlKey: true, key: '[' })
    )
  })

// The redesign is opt-in until it is turned on: these cases describe it, so they select it.
beforeEach(() => $shellChoice.set(true))

beforeEach(() => {
  // Ctrl+[ off macOS is ⌘[ on it: the code editor's indentLess.
  setBinding('nav.back', ['mod+['])
  $historyPosition.set({ index: 0, last: 0 })
  $workspaceIsPage.set(true)
})

afterEach(() => {
  resetBinding('nav.back')
  $workspaceIsPage.set(false)
  vi.restoreAllMocks()
})

function mount() {
  return render(
    <MemoryRouter initialEntries={['/capabilities']}>
      <Harness />
    </MemoryRouter>
  )
}

it('takes the band Back’s way out of a workspace page with no history behind it', () => {
  const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
  const { unmount } = mount()

  pressBack(window)

  expect(pathname).toBe(NEW_CHAT_ROUTE)
  expect(back).not.toHaveBeenCalled()
  unmount()
})

it('leaves the chord to a focused code editor', () => {
  const back = vi.spyOn(window.history, 'back').mockImplementation(() => undefined)
  const { getByTestId, unmount } = mount()

  pressBack(getByTestId('editor'))

  expect(pathname).toBe('/capabilities')
  expect(back).not.toHaveBeenCalled()
  unmount()
})
