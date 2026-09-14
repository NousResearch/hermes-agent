import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { isMacPlatform } from '@/lib/platform'
import { $bindings, $capture } from '@/store/keybinds'

import { group, split } from './model'
import { $activeTreeGroup, $hoveredTreeGroup, $layoutTree } from './store'
import { $heldTabModifier, useTabKeyHints } from './tab-key-hint-state'
import { TabKeyHint } from './tab-key-hints'

const disposers: Array<() => void> = []
const container = globalThis.document.createElement('div')
let root: ReturnType<typeof createRoot> | undefined

function Fixture() {
  useTabKeyHints()

  return (
    <>
      <TabKeyHint groupId="left" slot={1}>
        <span>left dot</span>
      </TabKeyHint>
      <TabKeyHint groupId="right" slot={1}>
        <span>right dot</span>
      </TabKeyHint>
    </>
  )
}

function modifier(held: boolean) {
  window.dispatchEvent(
    new KeyboardEvent(held ? 'keydown' : 'keyup', {
      key: isMacPlatform() ? 'Meta' : 'Control',
      code: isMacPlatform() ? 'MetaLeft' : 'ControlLeft',
      metaKey: isMacPlatform() && held,
      ctrlKey: !isMacPlatform() && held
    })
  )
}

afterEach(() => {
  act(() => root?.unmount())
  container.remove()
  disposers.splice(0).forEach(dispose => dispose())
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  $hoveredTreeGroup.set(null)
  $capture.set(null)
  vi.useRealTimers()
})

it('reveals only the shortcut target after a hold and clears on release, blur or capture', () => {
  vi.useFakeTimers()

  for (const id of ['workspace', 'a', 'b', 'c']) {
    disposers.push(registry.register({ area: 'panes', id, title: id, data: { placement: 'main' }, render: () => null }))
  }

  $layoutTree.set(split('row', [group(['workspace', 'a'], { id: 'left' }), group(['b', 'c'], { id: 'right' })]))
  $activeTreeGroup.set('left')
  $hoveredTreeGroup.set('right')
  globalThis.document.body.append(container)
  root = createRoot(container)
  act(() => root!.render(<Fixture />))
  act(() => {
    modifier(true)
    vi.advanceTimersByTime(399)
  })
  expect(container.querySelector('[data-tab-key-hint]')).toBeNull()
  act(() => vi.advanceTimersByTime(1))
  expect(container.querySelector('[data-tab-key-hint]')?.parentElement?.textContent).toBe('right dot1')
  act(() => $hoveredTreeGroup.set(null))
  expect(container.querySelector('[data-tab-key-hint]')?.parentElement?.textContent).toBe('left dot1')

  const originalBindings = $bindings.get()
  act(() => $bindings.set({ ...originalBindings, 'profile.switch.1': [] }))
  expect(container.querySelector('[data-tab-key-hint]')).toBeNull()
  act(() => $bindings.set(originalBindings))

  act(() => modifier(false))
  expect($heldTabModifier.get()).toBe(false)
  act(() => {
    modifier(true)
    vi.advanceTimersByTime(400)
    window.dispatchEvent(new Event('blur'))
  })
  expect(container.querySelector('[data-tab-key-hint]')).toBeNull()
  act(() => {
    modifier(true)
    vi.advanceTimersByTime(400)
    $capture.set('profile.switch.1')
  })
  expect(container.querySelector('[data-tab-key-hint]')).toBeNull()
})
