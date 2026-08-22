import { afterEach, describe, expect, it } from 'vitest'

import { $activeTreeGroup, $hoveredTreeGroup } from '@/components/pane-shell/tree/store'

import {
  blurComposerInput,
  getActiveComposer,
  markActiveComposer,
  onComposerFocusRequest,
  onComposerModelMenuRequest,
  releaseActiveComposer,
  requestComposerFocus,
  requestModelMenuToggle
} from './focus'
import { RICH_INPUT_SLOT } from './rich-editor'

/**
 * Inactive tabs keep their composer mounted, so an unscoped lookup can blur a
 * background input and leave the one the user is typing in focused.
 */

/** A composer input inside its own pane layer, hidden or not. */
function mountInput(hidden = false) {
  const layer = document.createElement('div')
  const input = document.createElement('div')
  input.dataset.slot = RICH_INPUT_SLOT
  input.tabIndex = 0
  layer.toggleAttribute('data-pane-hidden', hidden)
  layer.append(input)
  document.body.append(layer)

  return input
}

/** A chat surface stamp — the same `data-composer-target` ChatView hangs. */
function mountSurface(target: string, hidden = false) {
  const layer = document.createElement('div')
  layer.toggleAttribute('data-pane-hidden', hidden)
  const surface = document.createElement('div')
  surface.dataset.composerTarget = target
  layer.append(surface)
  document.body.append(layer)

  return surface
}

afterEach(() => {
  document.body.innerHTML = ''
  // `activeTarget` is module-level — a case that leaves a stale claim behind
  // would otherwise decide the next one.
  markActiveComposer('main')
  $hoveredTreeGroup.set(null)
  $activeTreeGroup.set(null)
})

describe('blurComposerInput', () => {
  it('blurs the foreground composer while a hidden tab matches first', () => {
    const background = mountInput(true)
    const foreground = mountInput()

    foreground.focus()
    blurComposerInput()

    expect(document.activeElement).not.toBe(foreground)
    expect(document.activeElement).not.toBe(background)
  })

  it('leaves focus alone when the composer does not hold it', () => {
    const outside = document.createElement('button')
    document.body.append(outside)
    mountInput()

    outside.focus()
    blurComposerInput()

    expect(document.activeElement).toBe(outside)
  })

  it('blurs the focused input when a split leaves several visible', () => {
    // Both zones of a split are visible, so matching the first one and then
    // testing it against activeElement no-ops for every zone but the leftmost.
    const left = mountInput()
    const right = mountInput()

    right.focus()
    blurComposerInput()

    expect(document.activeElement).not.toBe(right)
    expect(document.activeElement).not.toBe(left)
  })
})

/**
 * `markActiveComposer` has four call sites and, unguarded, no counterpart: an
 * unmounting or keep-alive-buried composer left `activeTarget` pointing at
 * itself, so every `'active'`-routed request was delivered to a target with no
 * on-screen subscriber. Type-to-focus preventDefaults the keystroke BEFORE the
 * request, so a dead target swallows the character and focuses nothing.
 */
describe('releaseActiveComposer', () => {
  it('falls back to the main composer when the claimant releases', () => {
    const root = document.createElement('div')
    root.dataset.slot = 'aui_edit-composer-root'
    document.body.append(root)

    markActiveComposer('edit')
    expect(getActiveComposer()).toBe('edit')

    root.remove()
    releaseActiveComposer('edit')

    expect(getActiveComposer()).toBe('main')
  })

  it('leaves the key with the live claimant when a stale composer releases late', () => {
    markActiveComposer('edit')
    markActiveComposer('tile:abc')

    releaseActiveComposer('edit')

    expect(getActiveComposer()).toBe('tile:abc')
  })

  it('prefers the visible chat surface over a hard main default', () => {
    const root = document.createElement('div')
    root.dataset.slot = 'aui_edit-composer-root'
    document.body.append(root)
    mountSurface('tile:visible')
    markActiveComposer('edit')

    root.remove()
    releaseActiveComposer('edit')

    expect(getActiveComposer()).toBe('tile:visible')
  })

  it('routes an active-target request to the main composer once the edit composer closes', async () => {
    // Mirrors the per-composer filter in use-composer-draft / user-edit-composer:
    // a composer ignores any request not addressed to its own target.
    const mainComposerSaw: string[] = []

    const off = onComposerFocusRequest(({ target }) => {
      if (target === 'main') {
        mainComposerSaw.push(target)
      }
    })

    const root = document.createElement('div')
    root.dataset.slot = 'aui_edit-composer-root'
    document.body.append(root)
    markActiveComposer('edit')
    root.remove()
    releaseActiveComposer('edit')
    requestComposerFocus('active')

    // `dispatch` defers to a macrotask so click/keydown handlers settle first.
    await new Promise(resolve => window.setTimeout(resolve, 0))
    off()

    expect(mainComposerSaw).toEqual(['main'])
  })
})

describe('resolveActive / keep-alive tab heal', () => {
  it('heals type-to-focus onto the visible main tab when a tile is buried', async () => {
    // Repro for the reported main-tab miss: user typed in a session tile, then
    // clicked the main/workspace tab without focusing its input. The tile stays
    // mounted under data-pane-hidden, so activeTarget still reads tile:… and
    // every type-to-focus request is dropped by the visible main composer.
    mountSurface('tile:buried', true)
    mountSurface('main')
    markActiveComposer('tile:buried')

    expect(getActiveComposer()).toBe('main')

    const mainSaw: string[] = []
    const tileSaw: string[] = []

    const off = onComposerFocusRequest(({ target }) => {
      if (target === 'main') {
        mainSaw.push(target)
      }

      if (target === 'tile:buried') {
        tileSaw.push(target)
      }
    })

    requestComposerFocus('active', { typeChar: 'h' })
    await new Promise(resolve => window.setTimeout(resolve, 0))
    off()

    expect(mainSaw).toEqual(['main'])
    expect(tileSaw).toEqual([])
    // Cache stays honest so dict/insert/Esc path all agree thereafter.
    expect(getActiveComposer()).toBe('main')
  })

  it('keeps a live tile claim while that tile is the visible surface', () => {
    mountSurface('main', true)
    mountSurface('tile:front')
    markActiveComposer('tile:front')

    expect(getActiveComposer()).toBe('tile:front')
  })

  it('heals an edit claim once the edit root is gone (no release site needed)', async () => {
    mountSurface('main')
    markActiveComposer('edit')
    // No edit root in the document → claim is dead. getActiveComposer heals.
    expect(getActiveComposer()).toBe('main')

    const mainSaw: string[] = []

    const off = onComposerFocusRequest(({ target }) => {
      if (target === 'main') {
        mainSaw.push(target)
      }
    })

    requestComposerFocus('active', { typeChar: 'a' })
    await new Promise(resolve => window.setTimeout(resolve, 0))
    off()

    expect(mainSaw).toEqual(['main'])
  })

  it('does not adopt a present-but-empty composer target as a routing key', () => {
    // `[data-composer-target]` matches on attribute presence, so a surface
    // stamped with an empty value is a candidate. Routing to '' would address a
    // composer that cannot exist and drop the keystroke — fall back to main.
    mountSurface('')
    markActiveComposer('tile:closed')

    expect(getActiveComposer()).toBe('main')
  })

  it('holds an edit claim while the edit composer root is mounted', () => {
    const root = document.createElement('div')
    root.dataset.slot = 'aui_edit-composer-root'
    document.body.append(root)
    mountSurface('main')
    markActiveComposer('edit')

    expect(getActiveComposer()).toBe('edit')
  })
})

/** A chat surface inside a layout zone, mirroring ChatView-in-tree-group. */
function mountZonedSurface(target: string, zone: string, hidden = false) {
  const group = document.createElement('div')
  group.dataset.treeGroup = zone
  const layer = document.createElement('div')
  layer.toggleAttribute('data-pane-hidden', hidden)
  const surface = document.createElement('div')
  surface.dataset.composerTarget = target
  layer.append(surface)
  group.append(layer)
  document.body.append(group)

  return surface
}

/**
 * A split renders one chat surface per zone and BOTH are visible, so resolving
 * "the" visible surface by first DOM match answers document order — always the
 * leftmost zone. Work in the right-hand zone and the composer routing key is
 * handed to the left one: Enter, type-to-focus, Esc, voice and soft `/` are all
 * preventDefault'd and dropped in the pane the user is actually looking at.
 */
describe('visibleChatTarget / split-zone identity', () => {
  /** Two visible chat surfaces, one per split zone. */
  function mountSplit() {
    mountZonedSurface('tile:left', 'zone-a')
    mountZonedSurface('tile:right', 'zone-b')
  }

  it('releases the routing key into the active zone, not the leftmost surface', () => {
    mountSplit()
    $activeTreeGroup.set('zone-b')
    markActiveComposer('edit')

    releaseActiveComposer('edit')

    expect(getActiveComposer()).toBe('tile:right')
  })

  it('releases into zone-a when zone-a is the active zone', () => {
    // The mirror of the case above: proves the resolver reads the zone rather
    // than simply preferring the last-mounted surface.
    mountSplit()
    $activeTreeGroup.set('zone-a')
    markActiveComposer('edit')

    releaseActiveComposer('edit')

    expect(getActiveComposer()).toBe('tile:left')
  })

  it('prefers the hovered zone over the active one (ladder order)', () => {
    mountSplit()
    $activeTreeGroup.set('zone-a')
    $hoveredTreeGroup.set('zone-b')
    markActiveComposer('edit')

    releaseActiveComposer('edit')

    expect(getActiveComposer()).toBe('tile:right')
  })

  it('falls through to the active zone when the hovered zone holds no chat surface', () => {
    // Pointer parked on the sidebar / titlebar — a zone that cannot serve the
    // verb must hand off to the next rung, not swallow the key.
    mountSplit()
    $hoveredTreeGroup.set('zone-sidebar')
    $activeTreeGroup.set('zone-b')
    markActiveComposer('edit')

    releaseActiveComposer('edit')

    expect(getActiveComposer()).toBe('tile:right')
  })

  it('still answers a visible surface when neither rung resolves', () => {
    // Nothing hovered and nothing activated yet: there is no identity to
    // resolve, so the fallback stays document order rather than null. Answering
    // null would route the key at a possibly-buried 'main' and drop every
    // keystroke — strictly worse than naming a surface the user can see.
    mountSplit()
    markActiveComposer('tile:closed')

    expect(getActiveComposer()).toBe('tile:left')
  })

  it('heals a stale claim into the active zone rather than document order', () => {
    // The reported repro: user works in the right zone, its claim goes stale
    // (tab switch / tile close), and type-to-focus heals onto the left zone.
    mountSplit()
    $activeTreeGroup.set('zone-b')
    markActiveComposer('tile:closed')

    expect(getActiveComposer()).toBe('tile:right')
  })

  it('answers the only visible surface whatever the zone state says', () => {
    // The non-split path must not start depending on the layout store.
    mountZonedSurface('tile:only', 'zone-a')
    $hoveredTreeGroup.set('zone-elsewhere')
    $activeTreeGroup.set('zone-elsewhere')
    markActiveComposer('tile:closed')

    expect(getActiveComposer()).toBe('tile:only')
  })
})

const collectModelMenuTargets = async (): Promise<string[]> => {
  const saw: string[] = []
  const off = onComposerModelMenuRequest(target => saw.push(target))

  await new Promise(resolve => window.setTimeout(resolve, 0))
  off()

  return saw
}

describe('requestModelMenuToggle', () => {
  it('targets the pane under the pointer over the focused one (#74447 convention)', async () => {
    mountZonedSurface('main', 'zone-a')
    mountZonedSurface('tile:hovered', 'zone-b')
    markActiveComposer('main')
    $hoveredTreeGroup.set('zone-b')

    expect(requestModelMenuToggle()).toBe(true)
    expect(await collectModelMenuTargets()).toEqual(['tile:hovered'])
  })

  it('falls back to the active composer when the pointer is off every zone', async () => {
    mountZonedSurface('main', 'zone-a')
    mountZonedSurface('tile:other', 'zone-b')
    markActiveComposer('tile:other')

    expect(requestModelMenuToggle()).toBe(true)
    expect(await collectModelMenuTargets()).toEqual(['tile:other'])
  })

  it('skips a hidden keep-alive tab in the hovered zone (targets its visible sibling)', async () => {
    mountZonedSurface('main', 'zone-a', true)
    mountZonedSurface('tile:front', 'zone-a')
    markActiveComposer('main')
    $hoveredTreeGroup.set('zone-a')

    expect(requestModelMenuToggle()).toBe(true)
    expect(await collectModelMenuTargets()).toEqual(['tile:front'])
  })

  it('returns false with no chat surface on screen so the caller can open the dialog', async () => {
    // Settings/profiles routes: no [data-composer-target] anywhere.
    expect(requestModelMenuToggle()).toBe(false)
    expect(await collectModelMenuTargets()).toEqual([])
  })
})
