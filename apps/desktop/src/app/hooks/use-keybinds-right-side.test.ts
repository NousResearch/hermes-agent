import { renderHook } from '@testing-library/react'
import { createElement, type ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark' as const, setMode: () => undefined })
}))

const toggleTerminalPane = vi.hoisted(() => vi.fn())

vi.mock('@/app/right-sidebar/terminal/reveal-focus', async importOriginal => ({
  ...(await importOriginal<object>()),
  toggleTerminalPane
}))

import { useKeybinds } from '@/app/hooks/use-keybinds'
import { findGroupOfPane, group, type LayoutNode, split } from '@/components/pane-shell/tree/model'
import { $hiddenTreePanes, $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { isMacPlatform } from '@/lib/platform'
import { $shellChoice } from '@/store/shell-variant'

// ⌘J dispatch must resolve the SAME physical right target the toggle mutates:
// a Browser column registers `placement: 'main'`, so the semantic side gate
// saw no right side and sent ⌘J to the terminal instead of folding it.

const BROWSER = 'preview-tile:url:browser'
const wrapper = ({ children }: { children: ReactNode }) => createElement(MemoryRouter, null, children)

let unmountKeybinds: (() => void) | undefined

beforeAll(() => {
  const placements = { sessions: 'left', files: 'right', [BROWSER]: 'main', workspace: 'main' }

  for (const [id, placement] of Object.entries(placements)) {
    registry.register({ id, area: 'panes', title: id, data: { placement }, render: () => null })
  }
})

beforeEach(() => {
  toggleTerminalPane.mockClear()
  unmountKeybinds = renderHook(
    () =>
      useKeybinds({
        archiveSelectedSession: () => undefined,
        openNewSessionTab: () => undefined,
        requestGateway: <T>() => Promise.resolve(undefined as T),
        startFreshSession: () => undefined,
        toggleCommandCenter: () => undefined,
        toggleSelectedPin: () => undefined
      }),
    { wrapper }
  ).unmount
})

afterEach(() => {
  unmountKeybinds?.()
  $layoutTree.set(null)
  $hiddenTreePanes.set(new Set())
  $shellChoice.set(null)
})

const pressToggle = () =>
  window.dispatchEvent(
    new KeyboardEvent('keydown', {
      bubbles: true,
      cancelable: true,
      code: 'KeyJ',
      ctrlKey: !isMacPlatform(),
      key: 'j',
      metaKey: isMacPlatform()
    })
  )

describe('⌘J (view.toggleRightSidebar)', () => {
  it('folds a Browser column on the right instead of toggling the terminal', () => {
    $shellChoice.set(false)
    $layoutTree.set(split('row', [group(['sessions', 'files']), group(['workspace']), group([BROWSER])]))

    pressToggle()

    expect(findGroupOfPane($layoutTree.get() as LayoutNode, BROWSER)?.minimized).toBe(true)
    expect(toggleTerminalPane).not.toHaveBeenCalled()
  })

  // The redesign's right side is its tool panel. A Browser tab tiled into a zone of its own is the
  // user's, not the side: the key leaves it where it is, shown.
  it('leaves a Browser zone of its own alone in the redesign', () => {
    $shellChoice.set(true)
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace']), group([BROWSER])]))

    pressToggle()

    expect($hiddenTreePanes.get().has(BROWSER)).toBe(false)
    expect(findGroupOfPane($layoutTree.get() as LayoutNode, BROWSER)?.minimized).toBeFalsy()
    expect(toggleTerminalPane).not.toHaveBeenCalled()
  })
})
