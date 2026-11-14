import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { findGroupOfPane, group, type LayoutNode, split } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'

import { handleSidePaneEscape } from './side-pane-escape'

// Esc with focus in a side pane used to fall through to the chat's global
// Esc-to-cancel and abort the running turn; it must fold the pane instead and
// leave the key claimed so the cancel listener stands down.

beforeAll(() => {
  const placements = { sessions: 'left', files: 'right', workspace: 'main' }

  for (const [id, placement] of Object.entries(placements)) {
    registry.register({ id, area: 'panes', title: id, data: { placement }, render: () => null })
  }
})

afterEach(() => {
  $layoutTree.set(null)
  document.body.innerHTML = ''
})

function pressEscapeIn(zoneId: string): KeyboardEvent {
  const zone = document.createElement('div')
  zone.dataset.treeGroup = zoneId
  const button = document.createElement('button')
  zone.append(button)
  document.body.append(zone)

  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Escape' })
  document.addEventListener('keydown', handleSidePaneEscape)
  button.dispatchEvent(event)
  document.removeEventListener('keydown', handleSidePaneEscape)

  return event
}

describe('Escape in a side pane', () => {
  it('folds the files zone and claims the key', () => {
    const files = group(['files'])
    $layoutTree.set(split('row', [group(['sessions']), group(['workspace']), files]))

    const event = pressEscapeIn(files.id)

    expect(event.defaultPrevented).toBe(true)
    expect(findGroupOfPane($layoutTree.get() as LayoutNode, 'files')?.minimized).toBe(true)
  })

  it('leaves the chat zone and the sessions column to their old Escape', () => {
    const sessions = group(['sessions'])
    const main = group(['workspace', 'files'])
    $layoutTree.set(split('row', [sessions, main]))

    expect(pressEscapeIn(main.id).defaultPrevented).toBe(false)
    expect(pressEscapeIn(sessions.id).defaultPrevented).toBe(false)
    expect(findGroupOfPane($layoutTree.get() as LayoutNode, 'workspace')?.minimized).toBeFalsy()
  })
})
