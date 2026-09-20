import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

// The profile scope is the axis folders are keyed by, so drive it directly
// instead of booting the profile store's gateway machinery.
vi.mock('@/store/profile', async () => {
  const { atom, computed } = await import('nanostores')

  const $activeGatewayProfile = atom('default')
  const $showAllProfiles = atom(false)

  return {
    ALL_PROFILES: '__all__',
    $activeGatewayProfile,
    $profileScope: computed([$showAllProfiles, $activeGatewayProfile], (all: boolean, profile: string) =>
      all ? '__all__' : profile
    ),
    $showAllProfiles
  }
})

const { $activeGatewayProfile, $showAllProfiles } = await import('@/store/profile')

const {
  $sessionFolders,
  $sessionFoldersAvailable,
  $sessionFolderScope,
  $sessionFolderStore,
  createSessionFolder,
  deleteSessionFolder,
  fileSessionInFolder,
  groupSessionsByFolder,
  renameSessionFolder,
  toggleSessionFolderCollapsed
} = await import('./session-folders')

const session = (id: string): SessionInfo => ({ id, profile: 'default' }) as unknown as SessionInfo

beforeEach(() => {
  window.localStorage.clear()
  $showAllProfiles.set(false)
  $activeGatewayProfile.set('default')
  $sessionFolderStore.set({})
})

describe('session folders', () => {
  it('files a session into a folder and lists it nowhere else', () => {
    const folder = createSessionFolder('Work')
    const sessions = [session('s1'), session('s2')]

    fileSessionInFolder('s1', folder!.id)

    const byFolder = groupSessionsByFolder(sessions, $sessionFolderScope.get())

    expect(byFolder.groups.map(group => [group.folder.name, group.sessions.map(s => s.id)])).toEqual([['Work', ['s1']]])
    expect(byFolder.unfiled.map(s => s.id)).toEqual(['s2'])
  })

  it('moves a session between folders and back out to the main list', () => {
    const work = createSessionFolder('Work')
    const ideas = createSessionFolder('Ideas')
    const sessions = [session('s1')]

    fileSessionInFolder('s1', work!.id)
    fileSessionInFolder('s1', ideas!.id)

    expect(groupSessionsByFolder(sessions, $sessionFolderScope.get()).groups.map(g => g.folder.name)).toEqual([
      'Work',
      'Ideas'
    ])

    const [workGroup, ideasGroup] = groupSessionsByFolder(sessions, $sessionFolderScope.get()).groups

    expect(workGroup.sessions).toEqual([])
    expect(ideasGroup.sessions.map(s => s.id)).toEqual(['s1'])

    fileSessionInFolder('s1', null)

    expect(groupSessionsByFolder(sessions, $sessionFolderScope.get()).unfiled.map(s => s.id)).toEqual(['s1'])
  })

  it('returns a deleted folder\u2019s sessions to the main list', () => {
    const folder = createSessionFolder('Work')
    const sessions = [session('s1')]

    fileSessionInFolder('s1', folder!.id)
    deleteSessionFolder(folder!.id)

    expect($sessionFolders.get()).toEqual([])
    expect(groupSessionsByFolder(sessions, $sessionFolderScope.get()).unfiled.map(s => s.id)).toEqual(['s1'])
  })

  it('renames and collapses a folder', () => {
    const folder = createSessionFolder('Work')

    renameSessionFolder(folder!.id, ' Deep work ')

    expect($sessionFolders.get()).toEqual([{ collapsed: false, id: folder!.id, name: 'Deep work' }])

    toggleSessionFolderCollapsed(folder!.id)

    expect($sessionFolders.get()[0].collapsed).toBe(true)
  })

  it('keeps folders on the profile they were made for', () => {
    createSessionFolder('Work')

    $activeGatewayProfile.set('other')

    expect($sessionFolders.get()).toEqual([])
    // A session filed under one profile must not be grouped under another.
    expect(groupSessionsByFolder([session('s1')], $sessionFolderScope.get()).unfiled.map(s => s.id)).toEqual(['s1'])

    $activeGatewayProfile.set('default')

    expect($sessionFolders.get().map(folder => folder.name)).toEqual(['Work'])
  })

  it('groups nothing while the sidebar spans every profile', () => {
    const folder = createSessionFolder('Work')

    fileSessionInFolder('s1', folder!.id)
    $showAllProfiles.set(true)

    expect($sessionFoldersAvailable.get()).toBe(false)
    expect($sessionFolders.get()).toEqual([])
    expect(createSessionFolder('Late')).toBeNull()
    expect(groupSessionsByFolder([session('s1')], $sessionFolderScope.get()).unfiled.map(s => s.id)).toEqual(['s1'])
  })

  it('hands the list straight back when there are no folders', () => {
    const sessions = [session('s1')]

    expect(groupSessionsByFolder(sessions, $sessionFolderScope.get()).unfiled).toBe(sessions)
  })

  it('drops a persisted assignment whose folder no longer exists', async () => {
    // Another window deleted the folder: its assignments are on disk with no
    // folder to honour. Load them the way a cold boot does — through the
    // persisted codec — so the drop is proven on the real path.
    window.localStorage.setItem(
      'hermes.desktop.sessionFolders',
      JSON.stringify({
        default: {
          filed: { s1: 'f_gone', s2: 'f_kept' },
          folders: [{ collapsed: false, id: 'f_kept', name: 'Kept' }]
        }
      })
    )
    vi.resetModules()

    const fresh = await import('./session-folders')
    const scope = fresh.$sessionFolderScope.get()

    expect(scope.filed).toEqual({ s2: 'f_kept' })
    // The stranded session is still listed rather than hidden from both places.
    expect(groupSessionsByFolder([session('s1'), session('s2')], scope).unfiled.map(s => s.id)).toEqual(['s1'])
  })
})
