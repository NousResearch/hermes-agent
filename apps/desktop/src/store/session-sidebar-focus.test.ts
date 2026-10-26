// Loaded for its primary-follow wiring: a new primary selection re-fronts the
// workspace, which the last case asserts through the focus derivation.
import '@/store/session-states'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import {
  $activeTreeGroup,
  $layoutTree,
  noteActiveTreeGroup,
  trackActiveTreeGroup
} from '@/components/pane-shell/tree/store'
import { setWorkspaceScope } from '@/components/pane-shell/workspace-scope'
import { $selectedStoredSessionId } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'

const pane = (id: string) => `session-tile:${id}`

function target(groupId: string) {
  const button = document.createElement('button')
  button.dataset.treeGroup = groupId
  document.body.append(button)

  return button
}

describe('session focus while interacting with the sidebar', () => {
  let stopTracking: () => void

  beforeEach(() => {
    setWorkspaceScope('sessions')
    $selectedStoredSessionId.set('primary')
    noteActiveTreeGroup(null)
    $layoutTree.set(
      split('row', [
        group(['sessions'], { active: 'sessions', id: 'sidebar' }),
        group(['workspace', pane('main')], { active: pane('main'), id: 'main' }),
        group([pane('split')], { active: pane('split'), id: 'split' })
      ])
    )
    stopTracking = trackActiveTreeGroup()
  })

  afterEach(() => {
    stopTracking()
    document.body.replaceChildren()
    $layoutTree.set(null)
    noteActiveTreeGroup(null)
    $selectedStoredSessionId.set(null)
  })

  it('retains the active chat through sidebar clicks and keyboard focus while still switching between chats', () => {
    const sidebar = target('sidebar')
    const main = target('main')
    const split = target('split')

    main.focus()
    expect($focusedStoredSessionId.get()).toBe('main')
    sidebar.dispatchEvent(new Event('pointerdown', { bubbles: true }))
    expect($activeTreeGroup.get()).toBe('sidebar')
    expect($focusedStoredSessionId.get()).toBe('main')

    split.focus()
    expect($focusedStoredSessionId.get()).toBe('split')
    sidebar.focus()
    expect($activeTreeGroup.get()).toBe('sidebar')
    expect($focusedStoredSessionId.get()).toBe('split')

    main.focus()
    expect($focusedStoredSessionId.get()).toBe('main')
  })

  it('keeps the remembered chat while Files or Terminal own focus', () => {
    const tree = $layoutTree.get()!
    $layoutTree.set(split('row', [tree, group(['files', 'terminal'], { active: 'files', id: 'tools' })]))
    const tools = target('tools')

    target('split').focus()
    tools.focus()
    expect($activeTreeGroup.get()).toBe('tools')
    expect($focusedStoredSessionId.get()).toBe('split')

    // The remembered group fronting another chat while chrome owns focus (⌘1..9, drag-to-split) is followed.
    $layoutTree.set(
      split('row', [
        group(['sessions'], { active: 'sessions', id: 'sidebar' }),
        group(['workspace', pane('main')], { active: pane('main'), id: 'main' }),
        group([pane('split'), pane('other')], { active: pane('other'), id: 'split' }),
        group(['files', 'terminal'], { active: 'files', id: 'tools' })
      ])
    )
    expect($focusedStoredSessionId.get()).toBe('other')

    // A preview later covering the followed chat keeps it, not the stale remembered tile.
    $layoutTree.set(
      split('row', [
        group(['sessions'], { active: 'sessions', id: 'sidebar' }),
        group(['workspace', pane('main')], { active: pane('main'), id: 'main' }),
        group([pane('split'), pane('other'), 'preview-tile:file:test'], {
          active: 'preview-tile:file:test',
          id: 'split'
        }),
        group(['files', 'terminal'], { active: 'files', id: 'tools' })
      ])
    )
    expect($focusedStoredSessionId.get()).toBe('other')
  })

  it('retains the chat when a preview replaces its active tab in the same group', () => {
    $layoutTree.set(
      group(['workspace', pane('a'), pane('b'), 'preview-tile:file:test'], { active: pane('a'), id: 'main' })
    )
    target('main').focus()
    expect($focusedStoredSessionId.get()).toBe('a')

    $layoutTree.set(
      group(['workspace', pane('a'), pane('b'), 'preview-tile:file:test'], {
        active: 'preview-tile:file:test',
        id: 'main'
      })
    )
    expect($focusedStoredSessionId.get()).toBe('a')

    $layoutTree.set(
      group(['workspace', pane('a'), pane('b'), 'preview-tile:file:test'], { active: pane('b'), id: 'main' })
    )
    $layoutTree.set(
      group(['workspace', pane('a'), pane('b'), 'preview-tile:file:test'], {
        active: 'preview-tile:file:test',
        id: 'main'
      })
    )
    expect($focusedStoredSessionId.get()).toBe('b')
  })

  it('uses the visible main tab on restore and after the remembered split closes', () => {
    const sidebar = target('sidebar')
    sidebar.focus()
    expect($focusedStoredSessionId.get()).toBe('main')

    target('split').focus()
    sidebar.focus()
    $layoutTree.set(
      split('row', [
        group(['sessions'], { active: 'sessions', id: 'sidebar' }),
        group(['workspace', pane('main')], { active: pane('main'), id: 'main' })
      ])
    )
    expect($focusedStoredSessionId.get()).toBe('main')

    $selectedStoredSessionId.set('new-primary')
    expect($focusedStoredSessionId.get()).toBe('new-primary')
  })
})
