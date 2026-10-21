import { afterEach, describe, expect, it } from 'vitest'

import { $hoveredTreeGroup } from '@/components/pane-shell/tree/store'

import {
  ackComposerInsert,
  blurComposerInput,
  focusComposerInput,
  getActiveComposer,
  markActiveComposer,
  onComposerDictationRequest,
  onComposerDraftRequests,
  onComposerFocusRequest,
  onComposerInsertRequest,
  onComposerModelMenuRequest,
  releaseActiveComposer,
  requestComposerDictation,
  requestComposerFocus,
  requestComposerGetDraft,
  requestComposerInsert,
  requestComposerInsertAcked,
  requestComposerSetDraft,
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
})

describe('focusComposerInput', () => {
  it('does not steal the caret from another live composer', () => {
    const foreground = mountInput()
    const background = mountInput()

    foreground.focus()
    focusComposerInput(background)

    expect(document.activeElement).toBe(foreground)
  })

  it('still focuses when the caret is not already in a composer', () => {
    const input = mountInput()
    const outside = document.createElement('button')

    document.body.append(outside)
    outside.focus()
    focusComposerInput(input)

    expect(document.activeElement).toBe(input)
  })

  it('takes the caret from a hidden keep-alive composer on tab switch', () => {
    const hidden = mountInput(true)
    const visible = mountInput()

    hidden.focus()
    focusComposerInput(visible)

    expect(document.activeElement).toBe(visible)
  })
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

describe('requestComposerDictation', () => {
  it('delivers the request only to the active visible composer', async () => {
    mountSurface('main', true)
    mountSurface('tile:front')
    markActiveComposer('tile:front')
    const targets: string[] = []
    const off = onComposerDictationRequest(target => targets.push(target))

    requestComposerDictation('active')
    await new Promise(resolve => window.setTimeout(resolve, 0))
    off()

    expect(targets).toEqual(['tile:front'])
  })
})

/**
 * The plugin SDK's draft read/write bus (`host.composer`). A mounted composer
 * answers for its own sessions and the active composer answers `active`
 * requests; unaddressed surfaces must stay silent so a plugin addressing one
 * session never reads or writes another's draft.
 */
describe('composer draft requests', () => {
  const disposer: (() => void)[] = []

  afterEach(() => {
    disposer.splice(0).forEach(off => off())
  })

  function mountDraft(id: string, text: string, active = false) {
    const state = { text, wrote: null as null | string }
    disposer.push(
      onComposerDraftRequests(
        { getIds: () => [id], isActive: () => active },
        {
          read: () => state.text,
          write: next => {
            if (next.trim() === '') {
              return false
            }

            state.wrote = next

            return true
          }
        }
      )
    )

    return state
  }

  it('reads the draft of the addressed session', async () => {
    mountDraft('sess-a', 'draft A')
    mountDraft('sess-b', 'draft B')

    expect(await requestComposerGetDraft(['sess-b'])).toEqual({ text: 'draft B' })
  })

  it('answers an active request only from the composer the bus routes to', async () => {
    markActiveComposer('main')
    mountSurface('main')
    const active = mountDraft('sess-live', 'on screen', true)
    const dead = mountDraft('sess-dead', 'buried', false)

    expect(active.wrote).toBeNull()
    expect(dead.wrote).toBeNull()
    expect(await requestComposerSetDraft([], 'type here', { active: true })).toBe(true)
    expect(active.wrote).toBe('type here')
    // Without the isActive() gate the buried surface also writes (registered
    // after active so its reply loses the race, but the write still fires).
    expect(dead.wrote).toBeNull()
  })

  it('reads from the bus-routed composer on active access, not the first registered', async () => {
    markActiveComposer('main')
    mountSurface('main')
    // Register the inactive surface FIRST — without the isActive() gate its
    // listener answered by registration order and the wrong draft won.
    mountDraft('sess-dead', 'buried', false)
    mountDraft('sess-live', 'on screen', true)

    expect(await requestComposerGetDraft([], { active: true })).toEqual({ text: 'on screen' })
  })

  it('writes only the addressed session, through exactly one owner, and reports success', async () => {
    const a = mountDraft('sess-a', 'x')
    // A second owner of the same id (primary pane + keep-alive tile showing
    // one session) must not paint too: the first claim wins.
    const aTwin = mountDraft('sess-a', 'x')
    const b = mountDraft('sess-b', 'y')

    expect(await requestComposerSetDraft(['sess-a'], 'new text')).toBe(true)
    expect(a.wrote).toBe('new text')
    expect(aTwin.wrote).toBeNull()
    expect(b.wrote).toBeNull()
  })

  it('resolves null / false when no surface answers', async () => {
    mountDraft('sess-a', 'x')

    expect(await requestComposerGetDraft(['nobody'])).toBeNull()
    expect(await requestComposerSetDraft(['nobody'], 'hi')).toBe(false)
  })

  it('surfaces a write refused by the owner (a blank paint is a no-op)', async () => {
    mountDraft('sess-a', 'x')

    expect(await requestComposerSetDraft(['sess-a'], '   ')).toBe(false)
  })
})

describe('insert acknowledgement', () => {
  const disposer: (() => void)[] = []

  afterEach(() => {
    disposer.splice(0).forEach(off => off())
  })

  function mountInsert(target: string, seen: string[]) {
    disposer.push(
      onComposerInsertRequest(detail => {
        if (detail.target === target) {
          seen.push(detail.text)

          if (detail.token !== undefined) {
            ackComposerInsert(detail.token, true)
          }
        }
      })
    )
  }

  it('resolves true when the addressed composer claims the insert', async () => {
    const seen: string[] = []

    mountInsert('tile:sess-a', seen)
    await expect(requestComposerInsertAcked(' snippet ', { target: 'tile:sess-a' })).resolves.toBe(true)
    expect(seen).toEqual(['snippet'])
  })

  it('resolves false when no surface claims it', async () => {
    await expect(requestComposerInsertAcked('hello', { target: 'tile:gone' })).resolves.toBe(false)
  })

  it('resolves false for blank text without dispatching', async () => {
    const seen: string[] = []

    mountInsert('main', seen)
    await expect(requestComposerInsertAcked('   ')).resolves.toBe(false)
    expect(seen).toEqual([])
  })

  it('leaves the internal fire-and-forget insert path untracked', async () => {
    const seen: string[] = []

    mountInsert('main', seen)
    requestComposerInsert('untracked', { target: 'main' })
    await new Promise(resolve => window.setTimeout(resolve, 5))

    expect(seen).toEqual(['untracked'])
  })
})
