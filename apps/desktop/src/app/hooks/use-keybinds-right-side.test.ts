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
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { isMacPlatform } from '@/lib/platform'

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
        isActiveRunBusy: () => false,
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
})

describe('⌘J (view.toggleRightSidebar)', () => {
  it('folds a Browser column on the right instead of toggling the terminal', () => {
    $layoutTree.set(split('row', [group(['sessions', 'files']), group(['workspace']), group([BROWSER])]))

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

    expect(findGroupOfPane($layoutTree.get() as LayoutNode, BROWSER)?.minimized).toBe(true)
    expect(toggleTerminalPane).not.toHaveBeenCalled()
  })
})
