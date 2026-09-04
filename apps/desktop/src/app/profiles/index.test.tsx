import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type * as Nanostores from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { deleteProfile, getProfilesForScope } from '@/hermes'
import { $connectionsRegistry } from '@/store/connections'
import { retireLocalProfileGateways } from '@/store/gateway'
import { refreshProfiles, selectProfile, setActiveProfile } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

import { ProfilesView } from './index'

// These tests pin the invariant this whole area exists to hold: the Manage
// Profiles page and the sidebar rail share ONE set of profile dialogs, so both
// "New Profile" entry points render the same modal (SOUL.md included), and
// deleting the profile the gateway is on re-homes to default instead of
// stranding it on a dead backend. The drift that motivated the fix got in
// precisely because nothing rendered this view.

// Real i18n (useI18n falls back to English with no provider), so labels are the
// actual strings — no brittle key snapshot to maintain here.

// CodeEditor is CodeMirror; the detail pane's SOUL editor doesn't matter to
// these behaviors, so stub it out of the jsdom render.
vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: () => null
}))

vi.mock('@/hermes', () => ({
  createProfile: vi.fn(async () => ({ name: 'x', ok: true, path: '/x' })),
  deleteProfile: vi.fn(async () => ({ ok: true, path: '/x' })),
  getProfilesForScope: vi.fn(async () => ({ profiles: [] })),
  getProfileSoul: vi.fn(async () => ({ content: '', exists: true })),
  renameProfile: vi.fn(async () => ({ name: 'x', ok: true, path: '/x' })),
  updateProfileSoul: vi.fn(async () => ({ ok: true }))
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: vi.fn(() => null),
  retireAgentGateways: vi.fn(),
  retireLocalProfileGateways: vi.fn()
}))

const { $activeGatewayProfile: activeGateway, $profileColors } = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    $activeGatewayProfile: atom<string>('default'),
    $profileColors: atom<Record<string, string>>({})
  }
})

const connectionStores = vi.hoisted(() => {
  const { atom } = require('nanostores') as typeof Nanostores

  return {
    active: atom<null | string>(null),
    multiple: atom(false),
    registry: atom(null)
  }
})

vi.mock('@/store/connections', () => ({
  $activeConnectionId: connectionStores.active,
  $connectionsRegistry: connectionStores.registry,
  $hasMultipleConnections: connectionStores.multiple
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: activeGateway,
  $profileColors,
  normalizeProfileKey: (name: null | string | undefined) => (name ?? '').trim() || 'default',
  profileLabel: (profile: { display_name?: string; name: string }) =>
    (profile.display_name ?? '').trim() || profile.name,
  refreshProfiles: vi.fn(async () => [] as ProfileInfo[]),
  selectProfile: vi.fn(),
  setActiveProfile: vi.fn()
}))

afterEach(() => {
  cleanup()
  $connectionsRegistry.set(null)
  connectionStores.active.set(null)
  connectionStores.multiple.set(false)
  vi.clearAllMocks()
})

// The one non-default profile these tests act on. Its name doubles as the row's
// accessible name, so the delete helper queries by it rather than a literal.
const NAMED_PROFILE = 'work'

function makeProfile(name: string, isDefault = false): ProfileInfo {
  return {
    has_env: false,
    is_default: isDefault,
    model: null,
    name,
    path: `/home/user/.hermes/profiles/${name}`,
    provider: null,
    skill_count: 0
  }
}

function deferred<T>() {
  let reject!: (error: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })

  return { promise, reject, resolve }
}

// Radix's trigger opens on the pointerdown/up pair, not the synthetic click
// alone — fire the full sequence a real click produces.
function realClick(el: HTMLElement) {
  fireEvent.pointerDown(el, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(el, { button: 0, pointerType: 'mouse' })
  fireEvent.click(el)
}

// ProfilesView loads its list in a mount effect (refreshProfiles → setProfiles),
// so the first paint is the loader and the rows commit a microtask later. Flush
// that inside act() so the rows exist before anything queries them, and so the
// mount setState isn't left unwrapped.
async function renderProfilesView() {
  await act(async () => {
    render(<ProfilesView onClose={vi.fn()} />)
  })
}

// PanelListRow labels BOTH the row's select target and its kebab with the
// profile name (`menuLabel={profile.name}`), so the name alone matches two
// buttons. Only the kebab is a menu trigger, so `expanded` disambiguates.
function findRowMenu(profileName: string) {
  return screen.findByRole('button', { expanded: false, name: profileName })
}

// Open the (only non-default) row's actions menu → Delete → confirm. The
// confirm click kicks off an async chain (deleteProfile → onDeleted refresh →
// setProfiles, plus the re-home writes), so settle it inside act() to flush
// those updates deterministically instead of leaking them past the assertions.
async function deleteTheNamedProfile() {
  realClick(await findRowMenu(NAMED_PROFILE))
  fireEvent.click(await screen.findByRole('menuitem', { name: /delete/i }))
  const confirm = await screen.findByRole('button', { name: 'Delete' })
  await act(async () => {
    fireEvent.click(confirm)
  })
}

describe('ProfilesView', () => {
  it('loads and groups profiles from every registered gateway', async () => {
    connectionStores.multiple.set(true)
    $connectionsRegistry.set({
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        { id: 'cloud', kind: 'cloud', label: 'Cloud' }
      ],
      launchMode: 'primary',
      lastUsed: 'local',
      primary: 'local',
      version: 2
    } as never)
    vi.mocked(getProfilesForScope).mockImplementation(async scope => ({
      profiles:
        typeof scope === 'object' && scope?.connectionId === 'cloud'
          ? [makeProfile('cloud-work')]
          : [makeProfile('local-work')]
    }))

    await renderProfilesView()

    expect((await screen.findAllByRole('button', { name: 'local-work' })).length).toBeGreaterThan(0)
    expect((await screen.findAllByRole('button', { name: 'cloud-work' })).length).toBeGreaterThan(0)
    expect(screen.getAllByText('This device').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Cloud').length).toBeGreaterThan(0)
    expect(getProfilesForScope).toHaveBeenCalledWith({ connectionId: 'local' })
    expect(getProfilesForScope).toHaveBeenCalledWith({ connectionId: 'cloud' })
  })

  it('ignores an older fleet refresh after a newer refresh finishes', async () => {
    connectionStores.multiple.set(true)
    connectionStores.active.set('local')

    const registry = {
      connections: [
        { id: 'local', kind: 'local', label: 'This device' },
        { id: 'cloud', kind: 'cloud', label: 'Cloud' }
      ],
      launchMode: 'primary',
      lastUsed: 'local',
      primary: 'local',
      version: 2
    } as never

    $connectionsRegistry.set(registry)
    const oldLocal = deferred<{ profiles: ProfileInfo[] }>()
    const oldCloud = deferred<{ profiles: ProfileInfo[] }>()
    let request = 0
    vi.mocked(getProfilesForScope).mockImplementation(async scope => {
      const connectionId = typeof scope === 'object' ? scope?.connectionId : undefined
      request += 1

      if (request <= 2) {
        return connectionId === 'local' ? oldLocal.promise : oldCloud.promise
      }

      return { profiles: [makeProfile(`new-${connectionId}`, connectionId === 'cloud')] }
    })

    await renderProfilesView()
    await act(async () => connectionStores.active.set('cloud'))
    expect(await screen.findByRole('heading', { name: 'new-cloud' })).toBeTruthy()

    await act(async () => {
      oldLocal.resolve({ profiles: [makeProfile('stale-local', true)] })
      oldCloud.reject(new Error('offline'))
      await Promise.allSettled([oldLocal.promise, oldCloud.promise])
    })

    expect(screen.queryByText('stale-local')).toBeNull()
    expect(screen.queryByText('Cloud · unreachable')).toBeNull()
    expect(screen.getByRole('heading', { name: 'new-cloud' })).toBeTruthy()
  })

  it('opens the shared create dialog with the SOUL.md field (parity with the rail)', async () => {
    vi.mocked(refreshProfiles).mockResolvedValue([])

    await renderProfilesView()

    realClick(await screen.findByRole('button', { name: 'New profile' }))

    const soul = await screen.findByLabelText(/SOUL\.md/i)

    expect(soul.tagName).toBe('TEXTAREA')
    expect(soul.getAttribute('id')).toBe('new-profile-soul')
  })

  it('re-homes to default when the active profile is deleted', async () => {
    const deleteProfileMock = vi.mocked(deleteProfile)
    const retireLocalProfileGatewaysMock = vi.mocked(retireLocalProfileGateways)

    deleteProfileMock.mockClear()
    retireLocalProfileGatewaysMock.mockClear()
    vi.mocked(refreshProfiles).mockResolvedValue([makeProfile('default', true), makeProfile(NAMED_PROFILE)])
    activeGateway.set(NAMED_PROFILE)

    await renderProfilesView()
    await deleteTheNamedProfile()

    await waitFor(() => expect(deleteProfile).toHaveBeenCalledWith(NAMED_PROFILE))
    expect(retireLocalProfileGateways).toHaveBeenCalledWith(NAMED_PROFILE)
    expect(retireLocalProfileGatewaysMock.mock.invocationCallOrder[0]).toBeLessThan(
      deleteProfileMock.mock.invocationCallOrder[0]
    )
    await waitFor(() => expect(selectProfile).toHaveBeenCalledWith('default'))
    expect(setActiveProfile).toHaveBeenCalledWith('default')
  })

  it('leaves the active profile alone when a different profile is deleted', async () => {
    vi.mocked(selectProfile).mockClear()
    vi.mocked(setActiveProfile).mockClear()
    vi.mocked(refreshProfiles).mockResolvedValue([makeProfile('default', true), makeProfile(NAMED_PROFILE)])
    activeGateway.set('default')

    await renderProfilesView()
    await deleteTheNamedProfile()

    await waitFor(() => expect(deleteProfile).toHaveBeenCalledWith(NAMED_PROFILE))
    // The dialog closes once the delete settles; a non-active delete must not re-home.
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Delete' })).toBeNull())
    expect(selectProfile).not.toHaveBeenCalled()
    expect(setActiveProfile).not.toHaveBeenCalled()
  })
})
