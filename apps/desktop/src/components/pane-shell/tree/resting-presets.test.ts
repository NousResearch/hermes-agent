import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'

import { allPaneIds, findGroupOfPane, group, split } from './model'
import {
  applyLayoutPreset,
  deleteUserPreset,
  layoutPresetTier,
  registerBundledPresets,
  saveCurrentLayoutAs
} from './presets'
import { $dismissedPanes, $hiddenTreePanes, $layoutTree, bindPaneVisibility, bindToolPaneCollapse } from './store'

// Basic used to be `sessions | workspace`, and picking it produced Focus:
// `applyTree` adopts every registered pane the tree omits back in as
// workspace tabs. A preset that wants the tooling OFF SCREEN has to place it
// and rest it, so the arrangement survives adoption and only what's open
// differs from the deck that shows it.

const ARRANGEMENT = split('row', [
  group(['sessions']),
  group(['workspace']),
  split('column', [group(['review', 'files']), group(['terminal'])])
])

const disposers: (() => void)[] = []

beforeEach(() => {
  window.localStorage.clear()
  $dismissedPanes.set(new Set())
  $hiddenTreePanes.set(new Set())

  for (const [id, data] of [
    ['sessions', { placement: 'left' }],
    ['workspace', { placement: 'main', uncloseable: true }],
    ['files', { placement: 'right' }],
    ['review', { placement: 'right' }],
    ['terminal', { placement: 'bottom', revealOnPreset: true }]
  ] as const) {
    disposers.push(registry.register({ area: 'panes', data, id, render: () => null, title: id }))
  }
})

afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
})

describe('resting presets', () => {
  it('keeps the arrangement and closes the resting panes through their stores', () => {
    const $terminal = atom(true)
    const $files = atom(true)
    const $review = atom(true)

    bindToolPaneCollapse(
      'terminal',
      $terminal,
      () => $terminal.set(false),
      () => $terminal.set(true)
    )
    bindPaneVisibility(
      'files',
      $files,
      () => $files.set(false),
      () => $files.set(true)
    )
    bindPaneVisibility(
      'review',
      $review,
      () => $review.set(false),
      () => $review.set(true)
    )

    disposers.push(
      registerBundledPresets([
        { id: 'shows', title: 'Shows', order: 0, tree: ARRANGEMENT, tier: 'advanced' },
        { id: 'rests', title: 'Rests', order: 1, tree: ARRANGEMENT, resting: ['terminal', 'files', 'review'] }
      ])
    )

    applyLayoutPreset('rests', ARRANGEMENT)

    expect(allPaneIds($layoutTree.get()!).sort()).toEqual(allPaneIds(ARRANGEMENT).sort())
    expect([$terminal.get(), $files.get(), $review.get()]).toEqual([false, false, false])
    expect([...$hiddenTreePanes.get()].sort()).toEqual(['files', 'review'])
    expect(findGroupOfPane($layoutTree.get()!, 'terminal')?.minimized).toBe(true)

    // The deck that shows the tooling opens EVERYTHING it places.
    applyLayoutPreset('shows', ARRANGEMENT)

    expect([$terminal.get(), $files.get(), $review.get()]).toEqual([true, true, true])
    expect(allPaneIds($layoutTree.get()!).sort()).toEqual(allPaneIds(ARRANGEMENT).sort())

    // Resting again with the store ALREADY closed still collapses the rail:
    // a same-value closer is a no-op and the fresh tree carries no flag.
    $terminal.set(false)
    applyLayoutPreset('rests', ARRANGEMENT)

    expect(findGroupOfPane($layoutTree.get()!, 'terminal')?.minimized).toBe(true)
  })

  it('a deck saved from the live layout remembers what was closed', () => {
    const $terminal = atom(false)
    const $files = atom(true)

    bindToolPaneCollapse(
      'terminal',
      $terminal,
      () => $terminal.set(false),
      () => $terminal.set(true)
    )
    bindPaneVisibility(
      'files',
      $files,
      () => $files.set(false),
      () => $files.set(true)
    )

    disposers.push(
      registerBundledPresets([{ id: 'seed', title: 'Seed', order: 0, tree: ARRANGEMENT, resting: ['terminal'] }])
    )
    applyLayoutPreset('seed', ARRANGEMENT)
    saveCurrentLayoutAs('Mine')

    // Open the terminal by hand, then re-apply the saved deck: it rests again.
    $terminal.set(true)
    applyLayoutPreset('user-mine', $layoutTree.get()!)

    expect($terminal.get()).toBe(false)
    expect($files.get()).toBe(true)
    expect(layoutPresetTier('user-mine')).toBeUndefined()

    deleteUserPreset('user-mine')
  })

  it('a reveal on apply never steals the tab the preset put first', () => {
    const $terminal = atom(false)

    bindToolPaneCollapse(
      'terminal',
      $terminal,
      () => $terminal.set(false),
      () => $terminal.set(true)
    )

    // Focus: the terminal is a tab BEHIND the chat, and opening it on apply
    // used to front it over the chat.
    const focus = split('row', [group(['sessions']), group(['workspace', 'files', 'review', 'terminal'])])

    disposers.push(registerBundledPresets([{ id: 'focus', title: 'Focus', order: 0, tree: focus }]))
    applyLayoutPreset('focus', focus)

    expect($terminal.get()).toBe(true)
    expect(findGroupOfPane($layoutTree.get()!, 'workspace')?.active).toBe('workspace')
  })

  it('files bundled presets by their declared tier; user decks have none', () => {
    disposers.push(
      registerBundledPresets([{ id: 'deck', title: 'Deck', order: 0, tree: ARRANGEMENT, tier: 'advanced' }])
    )

    expect(layoutPresetTier('deck')).toBe('advanced')
    expect(layoutPresetTier('user-mine')).toBeUndefined()
  })
})
