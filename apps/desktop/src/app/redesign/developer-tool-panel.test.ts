import { atom } from 'nanostores'
import { afterEach, expect, it } from 'vitest'

import { DEFAULT_TREE } from '@/app/contrib/layout-presets'
import { allPaneIds, findGroupOfPane, group, split } from '@/components/pane-shell/tree/model'
import {
  $collapsedTreeSides,
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  $userPlacedPanes,
  bindPaneVisibility,
  bindToolPaneCollapse,
  dismissTreePane,
  isPaneVisible,
  moveTreePane,
  paneRootSide,
  registerPaneCloser,
  revealTreePane
} from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'
import { $shellChoice } from '@/store/shell-variant'

import {
  $developerToolPanelWidth,
  hideDeveloperToolPanel,
  inDeveloperToolPanel,
  installDeveloperToolPanelWidth,
  prepareDeveloperToolPanel
} from './developer-tool-panel'

afterEach(() => {
  $layoutTree.set(null)
  $collapsedTreeSides.set(new Set())
  $hiddenTreePanes.set(new Set())
  $userPlacedPanes.set(new Set())
  $narrowViewport.set(false)
  $shellChoice.set(null)
})

it('turns the legacy bottom-right tools into one side panel without losing chats or plugins', () => {
  $shellChoice.set(true)
  $layoutTree.set(split('row', [DEFAULT_TREE, group(['plugin:notes', 'session-tile:other'])]))
  const before = allPaneIds($layoutTree.get()!)
  prepareDeveloperToolPanel('terminal')
  const tree = $layoutTree.get()!
  const panel = findGroupOfPane(tree, 'terminal')!
  expect(panel.panes).toEqual(['files', 'review', 'terminal'])
  expect(allPaneIds(tree).sort()).toEqual(before.sort())
  expect(panel.active).toBe('terminal')
  expect(isPaneVisible('terminal')).toBe(true)
  expect(isPaneVisible('files')).toBe(false)
  prepareDeveloperToolPanel('files')
  expect($layoutTree.get()).toBe(tree)
  revealTreePane('files')
  expect(isPaneVisible('files')).toBe(true)
  expect(isPaneVisible('terminal')).toBe(false)
  expect(hideDeveloperToolPanel('files')).toBe(true)
  expect(isPaneVisible('files')).toBe(false)
  expect(isPaneVisible('terminal')).toBe(false)
  revealTreePane('terminal')
  expect(isPaneVisible('terminal')).toBe(true)
})

it('joins a browser to the existing panel without moving conversation tabs', () => {
  $shellChoice.set(true)
  $layoutTree.set(split('row', [DEFAULT_TREE, group(['preview-tile:browser', 'session-tile:other'])]))
  prepareDeveloperToolPanel('terminal')
  const panel = findGroupOfPane($layoutTree.get()!, 'terminal')!
  prepareDeveloperToolPanel('preview-tile:browser')
  expect(findGroupOfPane($layoutTree.get()!, 'preview-tile:browser')?.id).toBe(panel.id)
  expect(findGroupOfPane($layoutTree.get()!, 'session-tile:other')?.id).not.toBe(panel.id)
  const withBrowser = $layoutTree.get()
  prepareDeveloperToolPanel('terminal')
  expect($layoutTree.get()).toBe(withBrowser)
  expect(hideDeveloperToolPanel('preview-tile:browser')).toBe(true)
  expect(isPaneVisible('terminal')).toBe(false)
  revealTreePane('terminal')
  expect(isPaneVisible('terminal')).toBe(true)
  expect(isPaneVisible('preview-tile:browser')).toBe(false)
})

it('hides only the built-in tools: a plugin pane stacked into the panel stays shown, with its panel', () => {
  $shellChoice.set(true)
  $layoutTree.set(DEFAULT_TREE)
  prepareDeveloperToolPanel('terminal')
  const panel = findGroupOfPane($layoutTree.get()!, 'terminal')!
  moveTreePane('plugin:notes', { groupId: panel.id, pos: 'center' })
  revealTreePane('terminal')

  expect(hideDeveloperToolPanel('terminal')).toBe(true)

  for (const tool of ['files', 'review', 'terminal']) {
    expect(isPaneVisible(tool)).toBe(false)
  }

  expect($hiddenTreePanes.get().has('plugin:notes')).toBe(false)
  expect(isPaneVisible('plugin:notes')).toBe(true)
  // A tool tile still brings its tool back beside the plugin pane.
  revealTreePane('files')
  expect(isPaneVisible('files')).toBe(true)
  expect(findGroupOfPane($layoutTree.get()!, 'plugin:notes')?.id).toBe(panel.id)
})

it.each(['classic', 'narrow'])('keeps the %s layout unchanged', mode => {
  $shellChoice.set(mode !== 'classic')
  $narrowViewport.set(mode === 'narrow')
  $layoutTree.set(DEFAULT_TREE)
  prepareDeveloperToolPanel('terminal')
  expect($layoutTree.get()).toBe(DEFAULT_TREE)
  expect(hideDeveloperToolPanel('terminal')).toBe(false)
})

it('folds a resize of whichever tool shows into the one width every tool panel shares', () => {
  $shellChoice.set(true)
  const root = document.createElement('div')
  const stop = installDeveloperToolPanelWidth(root)

  setPaneWidthOverride('terminal', 480)

  expect($developerToolPanelWidth.get()).toBe(480)
  expect($paneStates.get().terminal?.widthOverride).toBeUndefined()
  expect(root.style.getPropertyValue('--developer-tool-panel-width')).toBe('480px')

  // Classic keeps sizing each pane on its own.
  $shellChoice.set(false)
  setPaneWidthOverride('files', 300)
  expect($paneStates.get().files?.widthOverride).toBe(300)
  expect($developerToolPanelWidth.get()).toBe(480)

  stop()
  setPaneWidthOverride('files', undefined)
  $developerToolPanelWidth.set(null)
})

it('hides store-bound tools through their stores, so a later Show brings them back', () => {
  $shellChoice.set(true)
  $layoutTree.set(DEFAULT_TREE)
  const $files = atom(true)
  const $terminal = atom(true)
  bindPaneVisibility('files', $files, () => $files.set(false), () => $files.set(true))
  bindToolPaneCollapse('terminal', $terminal, () => $terminal.set(false), () => $terminal.set(true), atom(false))
  prepareDeveloperToolPanel('terminal')
  revealTreePane('terminal')

  expect(hideDeveloperToolPanel('terminal')).toBe(true)
  expect($files.get()).toBe(false)
  expect($terminal.get()).toBe(false)
  expect(isPaneVisible('files')).toBe(false)
  expect(isPaneVisible('terminal')).toBe(false)

  // "Show files" (view.showFiles) writes the store.
  $files.set(true)
  expect(isPaneVisible('files')).toBe(true)

  registerPaneCloser('files')
  registerPaneCloser('terminal')
})

// A Browser tab registers as a main surface (it can stand as a tile), so the pane tree no longer
// reads the panel it joined as the window's right side, and Files' closer (contrib/controller.tsx,
// mirrored here) took its "dragged beside main" branch: it dismissed Files instead of hiding it.
// With Files gone the tools no longer counted as gathered, and each one opened next made a panel of
// its own beside the last (HADESK-90: one to three panels, the chat squeezed between them).
it('keeps one panel when it is closed and reopened with a Browser tab in it', () => {
  $shellChoice.set(true)
  $layoutTree.set(DEFAULT_TREE)
  const browser = 'preview-tile:browser'

  const unregister = [
    registry.register({ id: 'workspace', area: 'panes', data: { placement: 'main' }, render: () => null }),
    registry.register({ id: browser, area: 'panes', data: { placement: 'main' }, render: () => null })
  ]

  const $files = atom(true)
  bindPaneVisibility(
    'files',
    $files,
    () => (paneRootSide('files') === 'right' || inDeveloperToolPanel('files') ? $files.set(false) : dismissTreePane('files')),
    () => $files.set(true)
  )

  prepareDeveloperToolPanel('files')
  const panel = findGroupOfPane($layoutTree.get()!, 'files')!
  moveTreePane(browser, { groupId: $layoutTree.get() ? findGroupOfPane($layoutTree.get()!, 'workspace')!.id : '', pos: 'right' })
  prepareDeveloperToolPanel(browser)
  expect(findGroupOfPane($layoutTree.get()!, browser)?.id).toBe(panel.id)

  expect(hideDeveloperToolPanel(browser)).toBe(true)
  expect(findGroupOfPane($layoutTree.get()!, 'files')?.id).toBe(panel.id)

  // Reopening any tool brings it back in that panel, not in a new one.
  revealTreePane('review')
  prepareDeveloperToolPanel('review')
  expect(findGroupOfPane($layoutTree.get()!, 'review')?.id).toBe(panel.id)
  expect(findGroupOfPane($layoutTree.get()!, browser)?.id).toBe(panel.id)

  registerPaneCloser('files')
  unregister.forEach(stop => stop())
})

// Files can still leave the tree (its tab's Close, from a layout where it stands beside main). The
// tools that remain are still the panel: the next tool joins them.
it('gathers into the panel that is left when Files has been dismissed', () => {
  $shellChoice.set(true)
  $layoutTree.set(DEFAULT_TREE)
  prepareDeveloperToolPanel('terminal')
  const panel = findGroupOfPane($layoutTree.get()!, 'terminal')!
  dismissTreePane('files')
  moveTreePane('chat-items', { groupId: findGroupOfPane($layoutTree.get()!, 'workspace')!.id, pos: 'right' })

  prepareDeveloperToolPanel('chat-items')

  expect(findGroupOfPane($layoutTree.get()!, 'chat-items')?.id).toBe(panel.id)
  expect(hideDeveloperToolPanel('chat-items')).toBe(true)
})
