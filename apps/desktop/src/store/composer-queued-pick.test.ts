import { afterEach, describe, expect, it } from 'vitest'

import { noteQueuedLaunchPick } from './composer-queued-pick'
import { getComposerSelectionScope, setComposerSelectionOwner, setConnection } from './session'
import { dropTilesForProfile, migrateTilesForProfile } from './session-states'

const key = (profile: string) => `hermes.desktop.composer.queued-pick.registry.local.${profile}`

function queueOn(profile: string) {
  setComposerSelectionOwner('local', profile)
  noteQueuedLaunchPick('s1', { model: 'm', provider: 'p' }, getComposerSelectionScope())
}

afterEach(() => {
  setConnection(null)
  window.localStorage.clear()
})

describe('a queued launch pick follows its profile', () => {
  it('to its new name on rename', () => {
    queueOn('work')
    migrateTilesForProfile('work', 'home')
    expect([
      window.localStorage.getItem(key('work')),
      JSON.parse(window.localStorage.getItem(key('home')) ?? 'null')
    ]).toEqual([null, { model: 'm', provider: 'p', sessionId: 's1' }])
  })

  it('out of storage on delete, leaving other profiles', () => {
    queueOn('work')
    queueOn('keep')
    dropTilesForProfile('work', { connectionId: 'local', profile: 'work' })
    expect([window.localStorage.getItem(key('work')), window.localStorage.getItem(key('keep'))]).toEqual([
      null,
      expect.any(String)
    ])
  })

  it("out of a direct remote's storage when that remote deletes it, not the same-named local one", () => {
    queueOn('work')
    setConnection({ baseUrl: 'https://box', mode: 'remote', profile: 'work' } as never)
    noteQueuedLaunchPick('s1', { model: 'm', provider: 'p' }, getComposerSelectionScope())
    const remote = `hermes.desktop.composer.queued-pick${getComposerSelectionScope()}`
    expect(window.localStorage.getItem(remote)).not.toBeNull()
    dropTilesForProfile('work')
    expect([window.localStorage.getItem(remote), window.localStorage.getItem(key('work'))]).toEqual([
      null,
      expect.any(String)
    ])
  })

  it("out of local storage for a local route, not the active remote's", () => {
    queueOn('work')
    setConnection({ baseUrl: 'https://box', connectionId: 'box', mode: 'remote', profile: 'work' } as never)
    noteQueuedLaunchPick('s1', { model: 'm', provider: 'p' }, getComposerSelectionScope())
    expect(getComposerSelectionScope()).toBe('.registry.box.work')
    dropTilesForProfile('work', { connectionId: '', profile: 'work' })
    expect([
      window.localStorage.getItem(key('work')),
      window.localStorage.getItem('hermes.desktop.composer.queued-pick.registry.box.work')
    ]).toEqual([null, expect.any(String)])
  })
})
