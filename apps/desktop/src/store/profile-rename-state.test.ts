import { beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

it('moves only the exact profile and connection-scoped navigation keys', async () => {
  const oldName = 'work'
  const newName = 'personal'
  const oldSuffix = `.remote.${encodeURIComponent('https://gateway-a.example')}.${oldName}`
  const newSuffix = `.remote.${encodeURIComponent('https://gateway-a.example')}.${newName}`
  const foreignSuffix = `.remote.${encodeURIComponent('https://gateway-b.example')}.${oldName}`
  const oldKey = `hermes.desktop.lastRoute.profile.${oldName}${oldSuffix}`
  const newKey = `hermes.desktop.lastRoute.profile.${newName}${newSuffix}`
  const siblingKey = `hermes.desktop.lastRoute.profile.work2${oldSuffix}`
  const foreignKey = `hermes.desktop.lastRoute.profile.${oldName}${foreignSuffix}`

  window.localStorage.setItem(oldKey, '/session-a')
  window.localStorage.setItem(siblingKey, '/session-sibling')
  window.localStorage.setItem(foreignKey, '/session-b')

  const migration = await import('./profile-rename-state')

  migration.completeProfileRenameState(oldName, newName, {
    connectionId: 'gateway-a',
    oldNavigationSuffix: oldSuffix,
    newNavigationSuffix: newSuffix
  })

  expect(window.localStorage.getItem(oldKey)).toBeNull()
  expect(window.localStorage.getItem(newKey)).toBe('/session-a')
  expect(window.localStorage.getItem(siblingKey)).toBe('/session-sibling')
  expect(window.localStorage.getItem(foreignKey)).toBe('/session-b')
})

it('recovers a pending rename only for its owning connection and preserves null navigation scope', async () => {
  const oldKey = 'hermes.desktop.lastRoute.profile.work'

  window.localStorage.setItem(oldKey, '/session-local')

  const migration = await import('./profile-rename-state')

  migration.stageProfileRenameState('work', 'personal', {
    connectionId: 'gateway-a',
    oldNavigationSuffix: null,
    newNavigationSuffix: null
  })

  expect(migration.recoverPendingProfileRenameState('personal', 'gateway-b')).toBe(false)
  expect(migration.recoverPendingProfileRenameState('personal', 'gateway-a')).toBe(true)
  expect(window.localStorage.getItem(oldKey)).toBe('/session-local')
})

it('re-homes every persisted session route after a profile rename', async () => {
  const oldName = 'work'
  const newName = 'hutnik-projectmanager'
  const siblingName = 'work2'
  const oldTail = JSON.stringify(['local', oldName, 'session-1'])
  const newTail = JSON.stringify(['local', newName, 'session-1'])
  const foreignTail = JSON.stringify(['gateway-b', oldName, 'session-2'])

  window.localStorage.setItem(`hermes.desktop.lastSessionId.profile.${oldName}`, 'session-1')
  window.localStorage.setItem(`hermes.desktop.lastRoute.profile.${oldName}`, '/session-1')
  window.localStorage.setItem(`hermes.desktop.lastSessionId.profile.${siblingName}`, 'session-2')
  window.localStorage.setItem(`hermes.desktop.lastRoute.profile.${siblingName}`, '/session-2')
  window.localStorage.setItem(
    'hermes.desktop.sessionTiles.v2',
    JSON.stringify({
      [oldName]: [
        {
          ownerProfile: oldName,
          ownerRoute: { connectionId: 'local', mode: 'local', profile: oldName, targetProfile: oldName },
          storedSessionId: 'session-1',
          workspaceMode: 'sessions'
        },
        {
          ownerProfile: oldName,
          ownerRoute: { connectionId: 'gateway-b', mode: 'remote', profile: oldName, targetProfile: oldName },
          storedSessionId: 'session-2',
          workspaceMode: 'sessions'
        }
      ]
    })
  )
  window.localStorage.setItem(
    'hermes.desktop.sessionOwnerHints.v1',
    JSON.stringify([
      ['session-1', { connectionId: 'local', mode: 'local', profile: oldName, targetProfile: oldName }],
      ['session-2', { connectionId: 'gateway-b', mode: 'remote', profile: oldName, targetProfile: oldName }]
    ])
  )
  window.localStorage.setItem('hermes.transcript-tail.v2-index', JSON.stringify([oldTail, foreignTail]))
  window.localStorage.setItem(`hermes.transcript-tail.v2:${oldTail}`, JSON.stringify({ messages: [{ id: 'u1' }] }))
  window.localStorage.setItem(`hermes.transcript-tail.v2:${foreignTail}`, JSON.stringify({ messages: [{ id: 'u2' }] }))

  // Hydrate the live registries before the rename event, matching an active
  // renderer whose stores already own these persisted records.
  const [session, migration] = await Promise.all([
    import('./session'),
    import('./profile-rename-state'),
    import('./session-states')
  ]).then(([sessionStore, profileRename]) => [sessionStore, profileRename] as const)

  migration.stageProfileRenameState(oldName, newName)
  expect(migration.recoverPendingProfileRenameState(newName, 'local')).toBe(true)

  expect(window.localStorage.getItem(`hermes.desktop.lastSessionId.profile.${oldName}`)).toBeNull()
  expect(window.localStorage.getItem(`hermes.desktop.lastSessionId.profile.${newName}`)).toBe('session-1')
  expect(window.localStorage.getItem(`hermes.desktop.lastRoute.profile.${newName}`)).toBe('/session-1')
  expect(window.localStorage.getItem(`hermes.desktop.lastSessionId.profile.${siblingName}`)).toBe('session-2')
  expect(window.localStorage.getItem(`hermes.desktop.lastRoute.profile.${siblingName}`)).toBe('/session-2')
  expect(window.localStorage.getItem(`hermes.transcript-tail.v2:${oldTail}`)).toBeNull()
  expect(window.localStorage.getItem(`hermes.transcript-tail.v2:${newTail}`)).not.toBeNull()
  expect(window.localStorage.getItem(`hermes.transcript-tail.v2:${foreignTail}`)).not.toBeNull()

  const tiles = JSON.parse(window.localStorage.getItem('hermes.desktop.sessionTiles.v2') ?? '{}')
  expect(tiles[oldName]).toEqual([
    expect.objectContaining({
      ownerProfile: oldName,
      ownerRoute: expect.objectContaining({ connectionId: 'gateway-b', profile: oldName }),
      storedSessionId: 'session-2'
    })
  ])
  expect(tiles[newName][0]).toMatchObject({
    ownerProfile: newName,
    ownerRoute: { profile: newName, targetProfile: newName },
    storedSessionId: 'session-1'
  })

  expect(session.getSessionOwnerHint('session-1', { connectionId: 'local', profile: newName })).toMatchObject({
    profile: newName,
    targetProfile: newName
  })
  expect(session.getSessionOwnerHint('session-2', { connectionId: 'gateway-b', profile: oldName })).toMatchObject({
    profile: oldName,
    targetProfile: oldName
  })
})
