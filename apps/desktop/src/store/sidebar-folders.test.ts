import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $sidebarFolders,
  createFolder,
  deleteFolder,
  folderForKey,
  folderKeySet,
  moveSessionToFolder,
  removeSessionFromFolder,
  renameFolder,
  reorderFolders,
  reorderFolderSessions,
  setFolderOpen,
  toggleFolderOpen
} from './sidebar-folders'

const STORAGE_KEY = 'hermes.desktop.sidebarFolders'

function persisted() {
  const raw = window.localStorage.getItem(STORAGE_KEY)

  return raw ? JSON.parse(raw) : null
}

describe('sidebar-folders store', () => {
  beforeEach(() => {
    $sidebarFolders.set([])
    window.localStorage.clear()
  })

  afterEach(() => {
    $sidebarFolders.set([])
    window.localStorage.clear()
  })

  describe('createFolder', () => {
    it('creates an open, empty folder and returns its id', () => {
      const id = createFolder('Work')
      const folders = $sidebarFolders.get()

      expect(folders).toHaveLength(1)
      expect(folders[0]).toMatchObject({ id, name: 'Work', open: true, sessionIds: [] })
    })

    it('trims the name and persists to localStorage', () => {
      createFolder('  Reading  ')

      expect($sidebarFolders.get()[0].name).toBe('Reading')
      expect(persisted()).toHaveLength(1)
      expect(persisted()[0].name).toBe('Reading')
    })

    it('mints unique ids for repeated creates', () => {
      const a = createFolder('A')
      const b = createFolder('B')

      expect(a).not.toBe(b)
      expect($sidebarFolders.get()).toHaveLength(2)
    })
  })

  describe('renameFolder', () => {
    it('renames an existing folder', () => {
      const id = createFolder('Old')
      renameFolder(id, 'New')

      expect($sidebarFolders.get()[0].name).toBe('New')
    })

    it('ignores blank names and unknown ids', () => {
      const id = createFolder('Keep')
      renameFolder(id, '   ')
      renameFolder('missing', 'Nope')

      expect($sidebarFolders.get()[0].name).toBe('Keep')
    })
  })

  describe('deleteFolder', () => {
    it('removes the folder but preserves its sessions by unfiling them', () => {
      const id = createFolder('Temp')
      moveSessionToFolder('s1', id)
      moveSessionToFolder('s2', id)

      deleteFolder(id)

      // Folder is gone...
      expect($sidebarFolders.get()).toHaveLength(0)
      // ...and the sessions are no longer claimed by any folder (unfiled).
      expect(folderForKey($sidebarFolders.get(), 's1')).toBeUndefined()
      expect(folderForKey($sidebarFolders.get(), 's2')).toBeUndefined()
    })
  })

  describe('moveSessionToFolder', () => {
    it('adds a session to the target folder', () => {
      const id = createFolder('Inbox')
      moveSessionToFolder('s1', id)

      expect($sidebarFolders.get()[0].sessionIds).toEqual(['s1'])
    })

    it('moves a session out of its previous folder (one folder per session)', () => {
      const a = createFolder('A')
      const b = createFolder('B')
      moveSessionToFolder('s1', a)
      moveSessionToFolder('s1', b)

      const [folderA, folderB] = $sidebarFolders.get()

      expect(folderA.sessionIds).toEqual([])
      expect(folderB.sessionIds).toEqual(['s1'])
    })

    it('is a no-op for an unknown target folder', () => {
      const id = createFolder('A')
      moveSessionToFolder('s1', 'missing')

      expect($sidebarFolders.get().find(f => f.id === id)?.sessionIds).toEqual([])
    })

    it('does not duplicate a session already in the target folder', () => {
      const id = createFolder('A')
      moveSessionToFolder('s1', id)
      moveSessionToFolder('s1', id)

      expect($sidebarFolders.get()[0].sessionIds).toEqual(['s1'])
    })
  })

  describe('removeSessionFromFolder', () => {
    it('unfiles a session without touching the folder', () => {
      const id = createFolder('A')
      moveSessionToFolder('s1', id)
      moveSessionToFolder('s2', id)

      removeSessionFromFolder('s1')

      expect($sidebarFolders.get()[0].sessionIds).toEqual(['s2'])
    })
  })

  describe('open state', () => {
    it('setFolderOpen and toggleFolderOpen flip the flag', () => {
      const id = createFolder('A')

      setFolderOpen(id, false)
      expect($sidebarFolders.get()[0].open).toBe(false)

      toggleFolderOpen(id)
      expect($sidebarFolders.get()[0].open).toBe(true)
    })
  })

  describe('reorderFolders', () => {
    it('reorders by id and keeps unnamed folders at the end', () => {
      const a = createFolder('A')
      const b = createFolder('B')
      const c = createFolder('C')

      reorderFolders([c, a])

      expect($sidebarFolders.get().map(f => f.id)).toEqual([c, a, b])
    })
  })

  describe('reorderFolderSessions', () => {
    it('reorders visible keys and appends hidden members', () => {
      const id = createFolder('A')
      moveSessionToFolder('s1', id)
      moveSessionToFolder('s2', id)
      moveSessionToFolder('s3', id)

      // Only s2 + s1 are "visible"; s3 is a non-loaded member.
      reorderFolderSessions(id, ['s2', 's1'])

      expect($sidebarFolders.get()[0].sessionIds).toEqual(['s2', 's1', 's3'])
    })
  })

  describe('persistence / load resilience', () => {
    it('survives a round-trip through localStorage', () => {
      const id = createFolder('Persisted')
      moveSessionToFolder('s1', id)

      const stored = persisted()
      expect(stored[0]).toMatchObject({ id, name: 'Persisted', sessionIds: ['s1'] })
    })

    it('clears storage when the last folder is removed', () => {
      const id = createFolder('A')
      deleteFolder(id)

      expect(persisted()).toBeNull()
    })
  })

  describe('pure helpers', () => {
    it('folderKeySet collects every claimed key', () => {
      const a = createFolder('A')
      const b = createFolder('B')
      moveSessionToFolder('s1', a)
      moveSessionToFolder('s2', b)

      expect(folderKeySet($sidebarFolders.get())).toEqual(new Set(['s1', 's2']))
    })

    it('folderForKey finds the owning folder', () => {
      const a = createFolder('A')
      moveSessionToFolder('s1', a)

      expect(folderForKey($sidebarFolders.get(), 's1')?.id).toBe(a)
      expect(folderForKey($sidebarFolders.get(), 'nope')).toBeUndefined()
    })
  })
})
