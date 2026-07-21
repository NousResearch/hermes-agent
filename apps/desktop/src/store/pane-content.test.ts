import { afterEach, describe, expect, it } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $dismissedPanes, $hiddenTreePanes, $layoutTree } from '@/components/pane-shell/tree/store'

import {
  $focusedPaneContent,
  $focusedPaneId,
  $focusedStoredSessionId,
  $visibleSessionIds,
  clearPaneContent,
  setPaneContent,
} from './pane-content'

afterEach(() => {
  clearPaneContent()
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  $hiddenTreePanes.set(new Set())
  $dismissedPanes.set(new Set())
})

describe('$visibleSessionIds', () => {
  it('includes sessions in separate visible split panes', () => {
    setPaneContent('main', { kind: 'chat', storedSessionId: 'session-a' })
    setPaneContent('session-tile:b', { kind: 'chat', storedSessionId: 'session-b' })
    $layoutTree.set(split('row', [
      group(['main'], { id: 'main-group' }),
      group(['session-tile:b'], { id: 'tile-group' }),
    ]))

    expect($visibleSessionIds.get()).toEqual(['session-a', 'session-b'])
  })

  it('excludes a chat session in an inactive stacked tab', () => {
    setPaneContent('main', { kind: 'chat', storedSessionId: 'session-a' })
    setPaneContent('session-tile:b', { kind: 'chat', storedSessionId: 'session-b' })
    $layoutTree.set(group(['main', 'session-tile:b'], {
      active: 'main',
      id: 'stack-group',
    }))

    expect($visibleSessionIds.get()).toEqual(['session-a'])
  })

  it('ignores visible non-chat pane content', () => {
    setPaneContent('main', { kind: 'chat', storedSessionId: 'session-a' })
    setPaneContent('skills', { kind: 'page', page: 'skills' })
    $layoutTree.set(split('row', [
      group(['main'], { id: 'main-group' }),
      group(['skills'], { id: 'skills-group' }),
    ]))

    expect($visibleSessionIds.get()).toEqual(['session-a'])
  })

  it('resolves the focused pane and its chat session from the active tree group', () => {
    setPaneContent('main', { kind: 'chat', storedSessionId: 'session-a' })
    setPaneContent('session-tile:b', { kind: 'chat', storedSessionId: 'session-b' })
    $layoutTree.set(split('row', [
      group(['main'], { id: 'main-group' }),
      group(['session-tile:b'], { id: 'tile-group' }),
    ]))
    $activeTreeGroup.set('tile-group')

    expect($focusedPaneId.get()).toBe('session-tile:b')
    expect($focusedPaneContent.get()).toEqual({ kind: 'chat', storedSessionId: 'session-b' })
    expect($focusedStoredSessionId.get()).toBe('session-b')
  })
})
