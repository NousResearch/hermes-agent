import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { $activeGatewayProfile, setShowAllProfiles } from '@/store/profile'
import { $currentCwd, applyConfiguredDefaultProjectDir } from '@/store/session'
import { deferred } from '@/test/deferred'

import { wipeSessionListsForGatewaySwitch } from './gateway-switch'
import { $projectScope, ALL_PROJECTS } from './project-scope'
import { $activeProjectId, $projects, $projectTree, refreshProjects, resolveNewSessionCwd } from './projects'

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  desktopDefaultCwd: vi.fn(),
  isDesktopFsRemoteMode: vi.fn(),
  selectDesktopPaths: vi.fn(),
  writeDesktopFileText: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  $gateway: atom(null),
  activeGateway: vi.fn(),
  activeGatewayConnectionId: vi.fn(() => null),
  isActivePrimary: vi.fn(() => true),
  ensureActiveGatewayOpen: vi.fn()
}))

vi.mock('@/lib/desktop-git', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  desktopGit: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getHermesConfig: vi.fn(),
  getProfiles: vi.fn(),
  hermesApi: vi.fn(),
  resetSidebarBatchCapability: vi.fn(),
  setApiRequestProfile: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))

vi.mock('@/app/contrib/hooks/use-background-sync', () => ({
  resetLiveRuntimeTracking: vi.fn()
}))

vi.mock('@/lib/query-client', () => ({
  invalidateProfileScopedQueries: vi.fn()
}))

const gw = await import('@/store/gateway')
const activeGateway = vi.mocked(gw.activeGateway)
const gatewayAtom = gw.$gateway

// #79406: each profile's projects.db is an island, but the renderer's project
// cache is app-global. Two profiles with different project roots must never
// see each other's projects through that cache.
describe('project cache per-profile boundary (#79406)', () => {
  const treeNode = (
    over: Partial<SidebarProjectTree> & Pick<SidebarProjectTree, 'id' | 'label'>
  ): SidebarProjectTree => ({ path: null, repos: [], sessionCount: 0, ...over })

  beforeEach(() => {
    window.localStorage.clear()
    applyConfiguredDefaultProjectDir('/work/configured')
    setShowAllProfiles(false)
    // Set the profile FIRST: crossing the cache boundary clears the atoms, so
    // seeding must happen on the profile that owns the data.
    $activeGatewayProfile.set('profile-a')
    $projectScope.set(ALL_PROJECTS)
    $projects.set([{ id: 'p_a', name: 'Profile A project' } as never])
    $projectTree.set([treeNode({ id: 'p_a', label: 'Profile A project', path: '/work/profile-a' })])
    $activeProjectId.set('p_a')
    $currentCwd.set('')
  })

  afterEach(() => {
    applyConfiguredDefaultProjectDir(null)
    setShowAllProfiles(false)
    $activeGatewayProfile.set('default')
    $projectScope.set(ALL_PROJECTS)
    $projectTree.set([])
    $projects.set([])
    $activeProjectId.set(null)
    $currentCwd.set('')
  })

  it("clears the previous profile's cache when the active gateway profile changes, so the next chat does not start in its project", () => {
    // The canonical repro: the user is inside Profile A's project
    // (/work/profile-a) and switches to Profile B, whose own active project is
    // /work/profile-b. The swap publishes $activeGatewayProfile once B's
    // backend is live — the cache must not carry A's catalog across it.
    $projectScope.set('p_a')
    $activeGatewayProfile.set('profile-b')

    expect($projects.get()).toEqual([])
    expect($projectTree.get()).toEqual([])
    expect($activeProjectId.get()).toBeNull()
    // The entered scope belonged to A's catalog and leaves with it.
    expect($projectScope.get()).toBe(ALL_PROJECTS)

    // The new chat resolves from Profile B's own context — the configured
    // default until B's tree lands, never A's project root.
    expect(resolveNewSessionCwd()).toBe('/work/configured')
  })

  it('leaves nothing to inherit when neither profile has project rows (empty store, #91818 variant)', () => {
    // The empty-project-store variant: no rows in either projects.db, so the
    // only things that could leak a cwd are the renderer's own scope + cache.
    $projects.set([])
    $projectTree.set([])
    $activeProjectId.set(null)
    $activeGatewayProfile.set('profile-empty')

    expect(resolveNewSessionCwd()).toBe('/work/configured')

    // With no configured default either, a new chat stays detached — never a
    // path remembered from the previous profile.
    applyConfiguredDefaultProjectDir('')
    expect(resolveNewSessionCwd()).toBe('')
  })

  it('strands a late projects.list reply from the departing profile so it cannot repopulate the cleared cache', async () => {
    const { promise: responseA, resolve: resolveA } = deferred<unknown>()

    const request = vi.fn(() => responseA)
    const gatewayA = { connectionState: 'open', request }

    activeGateway.mockReturnValue(gatewayA as never)
    gatewayAtom.set(gatewayA as never)

    const pending = refreshProjects()
    // The switch happens while Profile A's read is still in flight.
    $activeGatewayProfile.set('profile-b')
    resolveA({ active_id: 'p_a', projects: [{ id: 'p_a', name: 'Stale reply from A' }] })
    await pending

    expect($projects.get()).toEqual([])
    expect($activeProjectId.get()).toBeNull()
  })

  it("does not root a new chat in another profile's project entered from the All-profiles overview (#79003 family)", () => {
    setShowAllProfiles(true)
    // The merged overview holds every profile's projects; the user drilled
    // into Profile B's project while the gateway still serves Profile A.
    $projectTree.set([treeNode({ id: 'p_b', label: 'Profile B project', path: '/work/profile-b' })])
    $projectScope.set('p_b')

    // A plain new chat there targets the ACTIVE gateway profile (A) — it must
    // not start in Profile B's project root; the gateway resolves A's own
    // active project from its projects.db.
    expect(resolveNewSessionCwd()).toBe('/work/configured')
  })

  it('clears the project cache on a connection switch that keeps the same profile name', () => {
    // beginGatewaySwitch's wipe: a source change can serve the same profile
    // name from a different backend, so the profile-change reset never fires —
    // the wipe must drop the catalog with the scope.
    wipeSessionListsForGatewaySwitch()

    expect($projects.get()).toEqual([])
    expect($projectTree.get()).toEqual([])
    expect($activeProjectId.get()).toBeNull()
  })
})
