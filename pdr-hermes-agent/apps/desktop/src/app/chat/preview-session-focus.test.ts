import { afterEach, beforeAll, expect, it, vi } from 'vitest'

vi.mock('./right-rail/preview', () => ({ PreviewTilePane: () => null }))
vi.mock('./right-rail/preview-console-store', () => ({ forgetPreviewConsole: () => undefined }))

import '@/store/session-states'

import { findGroupOfPane, group } from '@/components/pane-shell/tree/model'
import {
  $layoutTree,
  activateTreePane,
  declareDefaultTree,
  noteActiveTreeGroup,
  revealTreePane,
  watchContributedPanes
} from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $previewTabs, $visiblePreviewTabs, openPreview } from '@/store/preview'
import { $selectedStoredSessionId } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $sessionTiles } from '@/store/session-states'

import { paneMirror } from './pane-mirror'
import { watchPreviewTiles } from './preview-tile'

beforeAll(() => {
  watchContributedPanes()
  paneMirror({
    source: $sessionTiles,
    key: tile => tile.storedSessionId,
    prefix: 'session-tile',
    title: id => id,
    minWidth: '22vw',
    dir: () => 'center',
    anchor: () => 'workspace',
    render: () => null,
    close: () => undefined
  })()
  watchPreviewTiles()
})

afterEach(() => {
  noteActiveTreeGroup(null)
  $previewTabs.set([])
  $sessionTiles.set([])
  $layoutTree.set(null)
  $selectedStoredSessionId.set(null)
})

it('settles session-owned preview reveals in the same group without losing chat ownership', () => {
  registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'workspace'
  })
  declareDefaultTree(group(['workspace'], { active: 'workspace', id: 'main' }))
  $selectedStoredSessionId.set('primary')
  $sessionTiles.set([{ storedSessionId: 'tile-a' }, { storedSessionId: 'tile-b' }])

  let focusChanges = 0

  const stop = $focusedStoredSessionId.listen(() => {
    if (++focusChanges > 20) {
      throw new Error('Session/preview focus did not settle')
    }
  })

  try {
    revealTreePane('session-tile:tile-a')
    noteActiveTreeGroup('main')
    openPreview({ kind: 'url', label: 'Browser', source: 'about:blank', url: 'about:blank' })
    const tabId = $previewTabs.get()[0]!.id
    $layoutTree.set(
      group(['workspace', 'session-tile:tile-a', 'session-tile:tile-b', `preview-tile:${tabId}`], {
        active: 'session-tile:tile-a',
        id: 'main'
      })
    )
    openPreview({ kind: 'url', label: 'Browser', source: 'about:blank', url: 'about:blank' })
    expect($focusedStoredSessionId.get()).toBe('tile-a')
    expect($visiblePreviewTabs.get().map(tab => tab.id)).toContain(tabId)

    const tree = $layoutTree.get()!
    expect(findGroupOfPane(tree, `preview-tile:${tabId}`)?.id).toBe('main')
    expect(findGroupOfPane(tree, `preview-tile:${tabId}`)?.active).toBe(`preview-tile:${tabId}`)

    for (let index = 0; index < 3; index++) {
      activateTreePane('main', 'session-tile:tile-b')
      expect($focusedStoredSessionId.get()).toBe('tile-b')
      expect($visiblePreviewTabs.get()).toEqual([])
      activateTreePane('main', 'session-tile:tile-a')
      expect($focusedStoredSessionId.get()).toBe('tile-a')
      expect($visiblePreviewTabs.get().map(tab => tab.id)).toContain(tabId)
    }
  } finally {
    stop()
  }
})
