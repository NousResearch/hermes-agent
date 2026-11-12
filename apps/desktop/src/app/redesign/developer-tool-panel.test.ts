import { afterEach, expect, it } from 'vitest'

import { DEFAULT_TREE } from '@/app/contrib/layout-presets'
import { allPaneIds, findGroupOfPane, group, split } from '@/components/pane-shell/tree/model'
import {
  $collapsedTreeSides,
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  $userPlacedPanes,
  isPaneVisible,
  moveTreePane,
  revealTreePane
} from '@/components/pane-shell/tree/store'
import { $paneStates, setPaneWidthOverride } from '@/store/panes'
import { $shellChoice } from '@/store/shell-variant'

import {
  $developerToolPanelWidth,
  hideDeveloperToolPanel,
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
