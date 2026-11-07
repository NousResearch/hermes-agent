import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// Every scenario boots a fresh module graph: the tile buckets, the pane tree
// and the registry are module singletons, and a relaunch is exactly "the same
// localStorage, new modules".

const remote = {
  baseUrl: 'http://100.119.19.8:9119',
  connectionId: 'mac-max',
  mode: 'remote',
  profile: 'default',
  registryScoped: true
} as unknown as HermesConnection

const SESSIONS = { workspaceMode: 'sessions' } as const

let routePrefix = () => ''

async function boot({ connect = true, search = '/' }: { connect?: boolean; search?: string } = {}) {
  window.history.replaceState(null, '', search)
  vi.resetModules()

  const { registry } = await import('@/contrib/registry')
  const routes = await import('@/app/routes')
  routePrefix = () => routes.sessionRoute('')
  const model = await import('@/components/pane-shell/tree/model')
  const tree = await import('@/components/pane-shell/tree/store')
  const { paneMirror } = await import('@/app/chat/pane-mirror')
  const states = await import('./session-states')
  const session = await import('./session')
  const profile = await import('./profile')
  const recovery = await import('./recovery-requests')

  registry.register({
    area: 'panes',
    data: { placement: 'main', uncloseable: true },
    id: 'workspace',
    render: () => null,
    title: 'Chat'
  })
  tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'main' }))
  tree.watchContributedPanes()
  paneMirror({
    source: states.$sessionTiles,
    key: tile => tile.storedSessionId,
    prefix: 'session-tile',
    dir: tile => tile.dir ?? 'center',
    anchor: tile => tile.anchor,
    before: tile => tile.before,
    minWidth: '20rem',
    title: id => id,
    render: () => null,
    close: states.closeSessionTile
  })()

  if (connect) {
    session.setConnection(remote)
  }

  const mainGroup = () => model.findGroupOfPane(tree.$layoutTree.get()!, 'workspace')!

  const open = (...ids: string[]) =>
    ids.forEach(id => states.openSessionTile(id, 'center', 'workspace', undefined, SESSIONS))

  const strip = () => mainGroup().panes.map(id => id.replace('session-tile:', ''))
  const tiles = () => states.$sessionTiles.get().map(tile => tile.storedSessionId)

  return { mainGroup, model, open, profile, recovery, session, states, strip, tiles, tree }
}

afterEach(() => {
  window.localStorage.clear()
  window.history.replaceState(null, '', '/')
})

describe('each profile view keeps its own tabs', () => {
  it('All profiles has its own tab set: filtering by a profile and coming back restores it', async () => {
    const app = await boot()
    app.profile.$activeGatewayProfile.set('default')
    app.profile.setShowAllProfiles(true)
    app.open('all-a', 'all-b')

    app.profile.setShowAllProfiles(false)
    app.profile.$activeGatewayProfile.set('infra')
    app.open('infra-a')
    expect(app.tiles()).toEqual(['infra-a'])

    app.profile.setShowAllProfiles(true)

    expect(app.tiles()).toEqual(['all-a', 'all-b'])
    expect(app.strip()).toEqual(['workspace', 'all-a', 'all-b'])
  })

  it('a session opened in another profile while in All does not swap the strip', async () => {
    const app = await boot()
    app.profile.$activeGatewayProfile.set('default')
    app.profile.setShowAllProfiles(true)
    app.open('all-a', 'all-b')

    // Shared-primary remote: opening an infra chat moves the gateway route.
    app.profile.$activeGatewayProfile.set('infra')

    expect(app.tiles()).toEqual(['all-a', 'all-b'])
  })

  it('returns to the same strip order and the same active tab', async () => {
    const app = await boot()
    app.profile.$activeGatewayProfile.set('default')
    app.profile.setShowAllProfiles(true)
    app.open('a', 'b', 'c')
    // User drags c to the front of the strip, then clicks b.
    app.tree.reorderTreePanes(app.mainGroup().id, ['session-tile:c'], 1)
    app.tree.activateTreePane(app.mainGroup().id, 'session-tile:b')
    expect(app.strip()).toEqual(['workspace', 'c', 'a', 'b'])

    app.profile.setShowAllProfiles(false)
    app.profile.$activeGatewayProfile.set('infra')
    app.open('x')
    app.tree.activateTreePane(app.mainGroup().id, 'session-tile:x')
    app.profile.setShowAllProfiles(true)

    expect(app.strip()).toEqual(['workspace', 'c', 'a', 'b'])
    expect(app.mainGroup().active).toBe('session-tile:b')

    // And the profile view kept its own active tab too.
    app.profile.setShowAllProfiles(false)
    expect(app.strip()).toEqual(['workspace', 'x'])
    expect(app.mainGroup().active).toBe('session-tile:x')
  })

  it('a split zone of tabs comes back as a split, in order', async () => {
    const app = await boot()
    app.profile.$activeGatewayProfile.set('default')
    app.profile.setShowAllProfiles(true)
    app.open('a')
    app.states.openSessionTile('r1', 'right', 'workspace', undefined, SESSIONS)
    app.states.openSessionTile('r2', 'center', 'session-tile:r1', undefined, SESSIONS)
    const zoneOf = (id: string) => app.model.findGroupOfPane(app.tree.$layoutTree.get()!, `session-tile:${id}`)
    expect(zoneOf('r1')?.panes).toEqual(['session-tile:r1', 'session-tile:r2'])

    app.profile.setShowAllProfiles(false)
    app.profile.$activeGatewayProfile.set('infra')
    app.profile.setShowAllProfiles(true)

    expect(app.strip()).toEqual(['workspace', 'a'])
    expect(zoneOf('r1')?.panes).toEqual(['session-tile:r1', 'session-tile:r2'])
    expect(zoneOf('r1')?.id).not.toBe(app.mainGroup().id)
  })
})

describe('the main tab follows the view the user picks', () => {
  it('restores each view’s main chat instead of dropping to a fresh draft', async () => {
    const app = await boot()
    const routed = () => app.recovery.$routeRequest.get()?.path
    app.profile.$activeGatewayProfile.set('infra')
    app.session.$selectedStoredSessionId.set('infra-main')

    app.profile.enterAllProfilesView()
    app.session.$selectedStoredSessionId.set('all-main')
    const fresh = app.profile.$freshSessionRequest.get()

    app.profile.selectProfile('infra')
    expect(routed()).toBe(`${routePrefix()}infra-main`)
    expect(app.profile.$freshSessionRequest.get()).toBe(fresh)

    app.session.$selectedStoredSessionId.set('infra-main')
    app.profile.enterAllProfilesView()
    expect(routed()).toBe(`${routePrefix()}all-main`)
  })

  it('a profile never visited still starts on a fresh draft', async () => {
    const app = await boot()
    app.profile.$activeGatewayProfile.set('default')
    app.profile.enterAllProfilesView()
    app.session.$selectedStoredSessionId.set('all-main')
    const fresh = app.profile.$freshSessionRequest.get()

    app.profile.selectProfile('infra')

    expect(app.profile.$freshSessionRequest.get()).toBe(fresh + 1)
  })

  it('a renamed profile keeps its main chat; a deleted one takes it along', async () => {
    const app = await boot({ connect: false })
    const routed = () => app.recovery.$routeRequest.get()?.path
    app.profile.$activeGatewayProfile.set('old')
    app.session.$selectedStoredSessionId.set('old-main')
    app.profile.enterAllProfilesView()

    app.states.migrateTilesForProfile('old', 'new')
    app.profile.selectProfile('new')
    expect(routed()).toBe(`${routePrefix()}old-main`)

    app.profile.enterAllProfilesView()
    app.states.dropTilesForProfile('new')
    const fresh = app.profile.$freshSessionRequest.get()
    app.profile.selectProfile('new')
    expect(app.profile.$freshSessionRequest.get()).toBe(fresh + 1)
  })

  it('the All toggle (filter menu, keybind) brings back each side’s main chat', async () => {
    const app = await boot()
    const routed = () => app.recovery.$routeRequest.get()?.path
    app.profile.$activeGatewayProfile.set('infra')
    app.session.$selectedStoredSessionId.set('infra-main')
    app.profile.toggleShowAllProfiles()
    app.session.$selectedStoredSessionId.set('all-main')

    app.profile.toggleShowAllProfiles()
    expect(routed()).toBe(`${routePrefix()}infra-main`)

    app.session.$selectedStoredSessionId.set('infra-main')
    app.profile.toggleShowAllProfiles()
    expect(routed()).toBe(`${routePrefix()}all-main`)
  })

  it('a rename while in All carries the owner of the tabs on screen', async () => {
    const app = await boot({ connect: false })
    app.profile.$activeGatewayProfile.set('default')
    app.profile.enterAllProfilesView()
    app.states.openSessionTile('x', 'center', 'workspace', undefined, { ...SESSIONS, ownerProfile: 'old' })

    app.states.migrateTilesForProfile('old', 'new')

    expect(app.states.$sessionTiles.get().map(tile => tile.ownerProfile)).toEqual(['new'])
  })
})

describe('tabs survive a relaunch', () => {
  it('shows the last view’s tabs before the remote connection publishes', async () => {
    const first = await boot()
    first.profile.$activeGatewayProfile.set('default')
    first.profile.setShowAllProfiles(true)
    first.open('kept-a', 'kept-b')

    const relaunch = await boot({ connect: false })
    expect(relaunch.tiles()).toEqual(['kept-a', 'kept-b'])
    expect(relaunch.strip()).toEqual(['workspace', 'kept-a', 'kept-b'])

    // A tab opened during the boot window stays with that view.
    relaunch.open('during-boot')
    relaunch.session.setConnection(remote)
    expect(relaunch.tiles()).toEqual(['kept-a', 'kept-b', 'during-boot'])
  })

  it('a HUD window never overwrites the main window’s tabs', async () => {
    const hud = await boot({ search: '/?win=hud' })
    const main = await boot()
    main.profile.$activeGatewayProfile.set('default')
    main.profile.setShowAllProfiles(true)
    main.open('main-a', 'main-b')

    hud.states.openSessionTile('hud-tab', 'center', 'workspace', undefined, SESSIONS)
    hud.states.discardSessionTile('hud-tab')

    const relaunch = await boot()
    expect(relaunch.tiles()).toEqual(['main-a', 'main-b'])
  })
})
