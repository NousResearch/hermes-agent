import { beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

it('re-homes every persisted session route after a profile rename', async () => {
  const oldName = 'webdesign_bhp'
  const newName = 'hutnik-projectmanager'
  const oldTail = JSON.stringify(['local', oldName, 'session-1'])
  const newTail = JSON.stringify(['local', newName, 'session-1'])

  window.localStorage.setItem(`hermes.desktop.lastSessionId.profile.${oldName}`, 'session-1')
  window.localStorage.setItem(`hermes.desktop.lastRoute.profile.${oldName}`, '/session-1')
  window.localStorage.setItem(
    'hermes.desktop.sessionTiles.v2',
    JSON.stringify({
      [oldName]: [
        {
          ownerProfile: oldName,
          ownerRoute: { connectionId: 'local', mode: 'local', profile: oldName, targetProfile: oldName },
          storedSessionId: 'session-1',
          workspaceMode: 'sessions'
        }
      ]
    })
  )
  window.localStorage.setItem(
    'hermes.desktop.sessionOwnerHints.v1',
    JSON.stringify([['session-1', { connectionId: 'local', mode: 'local', profile: oldName, targetProfile: oldName }]])
  )
  window.localStorage.setItem('hermes.transcript-tail.v2-index', JSON.stringify([oldTail]))
  window.localStorage.setItem(`hermes.transcript-tail.v2:${oldTail}`, JSON.stringify({ messages: [{ id: 'u1' }] }))

  const migration = await import('./profile-rename-state')

  migration.stageProfileRenameState(oldName, newName)
  expect(migration.recoverPendingProfileRenameState(newName)).toBe(true)

  expect(window.localStorage.getItem(`hermes.desktop.lastSessionId.profile.${oldName}`)).toBeNull()
  expect(window.localStorage.getItem(`hermes.desktop.lastSessionId.profile.${newName}`)).toBe('session-1')
  expect(window.localStorage.getItem(`hermes.desktop.lastRoute.profile.${newName}`)).toBe('/session-1')
  expect(window.localStorage.getItem(`hermes.transcript-tail.v2:${oldTail}`)).toBeNull()
  expect(window.localStorage.getItem(`hermes.transcript-tail.v2:${newTail}`)).not.toBeNull()

  const tiles = JSON.parse(window.localStorage.getItem('hermes.desktop.sessionTiles.v2') ?? '{}')
  expect(tiles[oldName]).toBeUndefined()
  expect(tiles[newName][0]).toMatchObject({
    ownerProfile: newName,
    ownerRoute: { profile: newName, targetProfile: newName },
    storedSessionId: 'session-1'
  })

  const { getSessionOwnerHint } = await import('./session')
  expect(getSessionOwnerHint('session-1', { connectionId: 'local', profile: newName })).toMatchObject({
    profile: newName,
    targetProfile: newName
  })
})
