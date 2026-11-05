// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'

import { $activeTreeGroup } from '@/components/pane-shell/tree/store'

import { $floatingComposerOwner } from './floating-state'
import { claimFloatingComposerForTarget, registerFloatingComposer } from './floating-target'
import { getActiveComposer } from './focus'

const unregister: Array<() => void> = []

afterEach(() => {
  unregister.splice(0).forEach(dispose => dispose())
  document.body.innerHTML = ''
  $floatingComposerOwner.set(null)
  $activeTreeGroup.set(null)
})

const mountZone = (id: string, target: string) => {
  const zone = document.createElement('div')
  zone.dataset.treeGroup = id
  const host = document.createElement('div')
  host.dataset.composerOwner = id
  const editor = document.createElement('div')
  editor.dataset.slot = 'composer-rich-input'
  editor.tabIndex = 0
  host.append(editor)
  zone.append(host)
  document.body.append(zone)
  unregister.push(registerFloatingComposer(id, { groupId: id, target }))

  return editor
}

describe('claimFloatingComposerForTarget', () => {
  it('lets a programmatic focus reach a split the pointer did not pick', () => {
    const main = mountZone('main-zone', 'main')
    const tile = mountZone('tile-zone', 'tile:abc')

    // The pointer last chose main: a bare focus() into the tile is redirected back.
    main.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, buttons: 0, clientX: 10, clientY: 10 }))
    expect(document.activeElement).toBe(main)
    tile.focus()
    expect(document.activeElement).toBe(main)

    expect(claimFloatingComposerForTarget('tile:abc')).toBe(true)
    expect($floatingComposerOwner.get()?.id).toBe('tile-zone')
    expect($activeTreeGroup.get()).toBe('tile-zone')
    expect(getActiveComposer()).toBe('tile:abc')

    tile.focus()
    expect(document.activeElement).toBe(tile)
  })

  it('is false and changes nothing when no mounted surface renders the target', () => {
    mountZone('main-zone', 'main')
    const before = $floatingComposerOwner.get()

    expect(claimFloatingComposerForTarget('tile:gone')).toBe(false)
    expect($floatingComposerOwner.get()).toBe(before)
  })
})
