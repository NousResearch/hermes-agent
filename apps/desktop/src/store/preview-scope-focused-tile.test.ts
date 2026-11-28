// Regression: the preview rail was keyed on the PRIMARY runtime's profile,
// not on the chat on screen. With another bot's chat focused as a session tile
// and the primary empty (Bot Mode), the rail fell back to the launch profile,
// so a preview the bot opened was stored in the launch profile's bucket. The
// next visit resolved the tile correctly, found no tab there, and the preview
// "vanished"; the agent re-opened it, and the same chat collected one tab per
// bucket.
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { paneMirror } from '@/app/chat/pane-mirror'
import * as model from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $previewTabs, openPreview, type PreviewTarget } from '@/store/preview'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import * as session from '@/store/session'
import * as states from '@/store/session-states'

const TABS_KEY = 'hermes.desktop.previewTabs.v2'

beforeAll(() => {
  const dispose = registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })

  tree.watchContributedPanes()
  paneMirror({
    source: states.$sessionTiles,
    key: tile => tile.storedSessionId,
    prefix: 'session-tile',
    dir: () => 'center',
    minWidth: '20rem',
    title: id => id,
    render: () => null,
    close: states.closeSessionTile
  })()

  return dispose
})

const urlTarget = (url: string): PreviewTarget => ({ kind: 'url', label: url, source: url, url })

function storedBuckets(): Record<string, { sessionId?: string }[]> {
  const raw = window.localStorage.getItem(TABS_KEY)

  return raw ? JSON.parse(raw) : {}
}

function openBotTile(storedId: string, profile: string) {
  states.openSessionTile(storedId, 'center', 'workspace', undefined, {
    ownerRoute: { connectionId: 'local', mode: 'local', profile },
    workspaceMode: 'bots',
    workspaceOwnerKey: `local::${profile}`,
    workspaceTabTitle: 'Bot Chat'
  })
  // Front and focus the tile's pane, as a click on its tab does.
  const paneId = `session-tile:${storedId}`
  const group = model.findGroupOfPane(tree.$layoutTree.get()!, paneId)!.id
  tree.activateTreePane(group, paneId)
  tree.noteActiveTreeGroup(group)
}

function reset() {
  for (const tile of states.$sessionTiles.get()) {
    states.discardSessionTile(tile.storedSessionId)
  }

  tree.$layoutTree.set(null)
  tree.$activeTreeGroup.set(null)
  session.$selectedStoredSessionId.set(null)
  session.$activeSessionId.set(null)
  $previewTabs.set([])
  window.localStorage.clear()
}

describe('the preview rail follows a focused bot tile', () => {
  beforeEach(() => {
    reset()
    $activeGatewayProfile.set('default')
    tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'main' }))
  })

  afterEach(reset)

  it("stores a tile's preview in the tile's own profile, not the launch profile", () => {
    openBotTile('web-chat', 'web')

    openPreview(urlTarget('http://localhost:3100/'), 'web-chat')

    const buckets = storedBuckets()

    expect(buckets[normalizeProfileKey('web')]?.map(tab => tab.sessionId)).toEqual(['web-chat'])
    expect(buckets.default).toBeUndefined()
  })

  it('finds the same tab again after visiting another bot', () => {
    openBotTile('web-chat', 'web')
    openPreview(urlTarget('http://localhost:3100/'), 'web-chat')
    const first = $previewTabs.get().map(tab => tab.id)

    openBotTile('jobs-chat', 'jobs')
    expect($previewTabs.get()).toEqual([])

    openBotTile('web-chat', 'web')
    expect($previewTabs.get().map(tab => tab.id)).toEqual(first)
  })
})
