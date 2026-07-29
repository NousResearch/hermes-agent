import { afterEach, describe, expect, it } from 'vitest'

import {
  blurComposerInput,
  getActiveComposer,
  markActiveComposer,
  onComposerFocusRequest,
  releaseActiveComposer,
  requestComposerFocus
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

  it('holds an edit claim while the edit composer root is mounted', () => {
    const root = document.createElement('div')
    root.dataset.slot = 'aui_edit-composer-root'
    document.body.append(root)
    mountSurface('main')
    markActiveComposer('edit')

    expect(getActiveComposer()).toBe('edit')
  })
})
