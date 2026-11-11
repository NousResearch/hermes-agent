import { afterEach, expect, it } from 'vitest'

import { DEFAULT_TREE } from '@/app/contrib/layout-presets'
import { allPaneIds, findGroupOfPane, group, split } from '@/components/pane-shell/tree/model'
import { $collapsedTreeSides, $hiddenTreePanes, $layoutTree, $narrowViewport, $userPlacedPanes, isPaneVisible, revealTreePane } from '@/components/pane-shell/tree/store'
import { $shellChoice } from '@/store/shell-variant'

import { hideDeveloperToolPanel, prepareDeveloperToolPanel } from './developer-tool-panel'

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

it.each(['classic', 'narrow'])('keeps the %s layout unchanged', mode => {
  $shellChoice.set(mode !== 'classic')
  $narrowViewport.set(mode === 'narrow')
  $layoutTree.set(DEFAULT_TREE)
  prepareDeveloperToolPanel('terminal')
  expect($layoutTree.get()).toBe(DEFAULT_TREE)
  expect(hideDeveloperToolPanel('terminal')).toBe(false)
})
