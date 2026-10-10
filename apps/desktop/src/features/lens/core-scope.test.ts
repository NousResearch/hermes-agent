import { backendScopeKey } from '@hermes/shared'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $previewComposerTarget, $previewTabs, openPreview } from '@/store/preview'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection, $selectedStoredSessionId, setSessionOwnerHint } from '@/store/session'
import { $sessionTiles, dropTilesForProfile, migrateTilesForProfile } from '@/store/session-states'

import { $lensCards, $lensScope, pinLensCapture } from './store'

const source = {
  url: 'https://example.com/item',
  title: 'Source',
  text: 'Private research',
  selector: '#item',
  tag: 'P',
  truncated: false
}

const preview = { kind: 'url' as const, label: 'Source', source: source.url, url: source.url }

function activate(id: string, connectionId: string, profile = 'research') {
  setSessionOwnerHint(id, { connectionId, profile })
  $selectedStoredSessionId.set(id)
  $activeSessionId.set(id)
}

beforeEach(() => {
  localStorage.clear()
})
afterEach(() => {
  $layoutTree.set(null)
  $activeTreeGroup.set(null)
  $sessionTiles.set([])
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $connection.set(null)
})

it('isolates browser tabs and evidence for same-name profiles across A → B → A', () => {
  activate('scope-a', 'server-a')
  expect($lensScope.get()).toBe(backendScopeKey('server-a', 'research'))
  const card = pinLensCapture(source, $lensScope.get())
  openPreview(preview)
  activate('scope-b', 'server-b')
  expect($lensCards.get()).toEqual([])
  expect($previewTabs.get()).toEqual([])
  pinLensCapture({ ...source, text: 'B only' }, $lensScope.get())
  activate('scope-a', 'server-a')
  expect($lensCards.get()).toEqual([card])
  expect($previewTabs.get()[0].target.url).toBe(source.url)
})

it('deletes only the routed connection and migrates only local profile data', () => {
  activate('remote-delete', 'remote', 'lifecycle')
  const remote = pinLensCapture(source, $lensScope.get())
  openPreview(preview)
  activate('local-rename', 'local', 'lifecycle')
  const local = pinLensCapture(source, $lensScope.get())
  migrateTilesForProfile('lifecycle', 'renamed')
  activate('local-renamed', 'local', 'renamed')
  expect($lensCards.get()[0]).toMatchObject({ id: local.id, scope: 'conn:local::renamed' })
  dropTilesForProfile('lifecycle', { connectionId: 'remote', profile: 'lifecycle' })
  expect($lensCards.get()[0].id).toBe(local.id)
  activate('remote-delete', 'remote', 'lifecycle')
  expect($lensCards.get()).toEqual([])
  expect($previewTabs.get()).toEqual([])
  expect(localStorage.getItem('hermes.desktop.lens.card.v1.' + remote.id)).toBeNull()
})

it('keeps ambiguous earlier profile-only captures saved but does not assign them to a connection', () => {
  const legacy = pinLensCapture(source, 'legacy-owner')
  activate('legacy-local', 'local', 'legacy-owner')
  expect($lensCards.get()).toEqual([])
  activate('legacy-remote', 'remote', 'legacy-owner')
  expect($lensCards.get()).toEqual([])
  expect(localStorage.getItem('hermes.desktop.lens.card.v1.' + legacy.id)).not.toBeNull()
})

it('follows a tile owner and retains it while its browser gains focus', () => {
  activate('primary-focus', 'server-main', 'focus')
  $sessionTiles.set([{ storedSessionId: 'tile-focus', ownerRoute: { connectionId: 'server-tile', profile: 'focus' } }])
  $layoutTree.set(
    group(['workspace', 'session-tile:tile-focus', 'preview-tile:url:https://example.com'], {
      active: 'session-tile:tile-focus',
      id: 'main'
    })
  )
  $activeTreeGroup.set('main')
  expect($lensScope.get()).toBe('conn:server-tile::focus')
  const card = pinLensCapture(source, $lensScope.get())
  $layoutTree.set(
    group(['workspace', 'session-tile:tile-focus', 'preview-tile:url:https://example.com'], {
      active: 'preview-tile:url:https://example.com',
      id: 'main'
    })
  )
  expect($lensCards.get()).toEqual([card])
  expect($previewComposerTarget.get()).toBe('tile:tile-focus')
  $layoutTree.set(group(['workspace', 'session-tile:tile-focus'], { active: 'workspace', id: 'main' }))
  expect($lensScope.get()).toBe('conn:server-main::focus')
  expect($previewComposerTarget.get()).toBe('main')
  expect($lensCards.get()).toEqual([])
})

it('re-homes a new draft on connection changes and keeps its scope through reconnect', () => {
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $activeGatewayProfile.set('draft-test')
  $connection.set({ mode: 'remote', connectionId: 'draft-a' } as never)
  const card = pinLensCapture(source, $lensScope.get())
  $connection.set({ mode: 'remote', connectionId: 'draft-b' } as never)
  expect($lensScope.get()).toBe('conn:draft-b::draft-test')
  expect($lensCards.get()).toEqual([])
  $connection.set(null)
  expect($lensScope.get()).toBe('conn:draft-b::draft-test')
  $connection.set({ mode: 'remote', connectionId: 'draft-a' } as never)
  expect($lensCards.get()).toEqual([card])
})
