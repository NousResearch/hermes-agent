import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import {
  $draggingSession,
  $sessionSectionMembership,
  $sessionSections,
  createSessionSection,
  deleteSessionSection,
  dropSessionSectionsForProfile,
  groupSessionsBySection,
  loadSessionSections,
  migrateSessionSectionsForProfile,
  moveSessionsToSection,
  normalizeMembership,
  normalizeSessionSections,
  renameSessionSection,
  sessionSectionId,
  UNASSIGNED_SESSION_KEY
} from './session-sections'

const STORAGE_KEY = 'hermes.desktop.sessionSections'

function storedProfiles(): Record<string, unknown> {
  return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}')
}

beforeEach(() => {
  window.localStorage.clear()
  $activeGatewayProfile.set('default')
  loadSessionSections()
})

describe('normalize', () => {
  it('drops blank names, blank ids and duplicate ids but keeps order', () => {
    const sections = normalizeSessionSections([
      { id: 'a', name: 'Clients' },
      { id: '', name: 'No id' },
      { id: 'b', name: '   ' },
      { id: 'a', name: 'Clients again' },
      { id: 'c', name: 'Billing' },
      'not an object'
    ])

    expect(sections).toEqual([
      { id: 'a', name: 'Clients' },
      { id: 'c', name: 'Billing' }
    ])
  })

  it('keeps only memberships that point at a live section', () => {
    const membership = normalizeMembership(
      { s1: 'a', s2: 'ghost', s3: 'b', '': 'a', s4: '' },
      [{ id: 'a', name: 'Clients' }]
    )

    expect(membership).toEqual({ s1: 'a' })
  })

  it('treats a non-object membership map as empty rather than throwing', () => {
    expect(normalizeMembership(null, [])).toEqual({})
    expect(normalizeMembership(['s1'], [])).toEqual({})
  })
})

describe('create / rename / move', () => {
  it('creates a section and files the given sessions into it', () => {
    const section = createSessionSection('Billing work', ['s1', 's2'])

    expect(section?.name).toBe('Billing work')
    expect($sessionSections.get()).toHaveLength(1)
    expect(sessionSectionId('s1')).toBe(section?.id)
    expect(sessionSectionId('s2')).toBe(section?.id)
    expect(sessionSectionId('s3')).toBeNull()
  })

  it('refuses a blank name instead of making an unnamed folder', () => {
    expect(createSessionSection('   ', ['s1'])).toBeNull()
    expect($sessionSections.get()).toEqual([])
    expect(sessionSectionId('s1')).toBeNull()
  })

  it('renames in place without touching membership', () => {
    const section = createSessionSection('Old', ['s1'])!
    const before = { ...$sessionSectionMembership.get() }

    renameSessionSection(section.id, 'New')

    expect($sessionSections.get()[0]?.name).toBe('New')
    expect($sessionSectionMembership.get()).toEqual(before)
  })

  it('moves a session between sections and back out to Unassigned', () => {
    const a = createSessionSection('A', ['s1'])!
    const b = createSessionSection('B')!

    moveSessionsToSection(['s1'], b.id)
    expect(sessionSectionId('s1')).toBe(b.id)

    moveSessionsToSection(['s1'], null)
    expect(sessionSectionId('s1')).toBeNull()
  })

  it('files a move to an unknown section as Unassigned rather than dropping the session', () => {
    const a = createSessionSection('A', ['s1'])!

    moveSessionsToSection(['s1'], 'sec-does-not-exist')

    expect(sessionSectionId('s1')).toBeNull()
    expect($sessionSections.get().map(s => s.id)).toEqual([a.id])
  })
})

describe('delete', () => {
  it('deletes the folder only: its sessions survive as Unassigned', () => {
    const section = createSessionSection('Clients', ['s1', 's2'])!

    deleteSessionSection(section.id)

    expect($sessionSections.get()).toEqual([])
    expect(sessionSectionId('s1')).toBeNull()
    expect(sessionSectionId('s2')).toBeNull()
  })

  it('undo restores the folder in its original slot and refiles the same sessions', () => {
    const first = createSessionSection('First')!
    const second = createSessionSection('Second', ['s1', 's2'])!
    const third = createSessionSection('Third')!

    const { undo } = deleteSessionSection(second.id)

    undo()

    expect($sessionSections.get().map(s => s.id)).toEqual([first.id, second.id, third.id])
    expect(sessionSectionId('s1')).toBe(second.id)
    expect(sessionSectionId('s2')).toBe(second.id)
  })

  it('undo is a no-op for a section that was never there', () => {
    createSessionSection('Only')!

    const { undo } = deleteSessionSection('sec-ghost')

    undo()

    expect($sessionSections.get()).toHaveLength(1)
  })
})

describe('grouping', () => {
  const rows = [{ id: 's1' }, { id: 's2' }, { id: 's3' }]

  it('draws sections in order and Unassigned last', () => {
    const a = createSessionSection('A', ['s1'])!
    const b = createSessionSection('B', ['s2'])!

    const groups = groupSessionsBySection(rows, $sessionSections.get(), $sessionSectionMembership.get())

    expect(groups.map(g => g.id)).toEqual([a.id, b.id, null])
    expect(groups[2]!.rows.map(r => r.id)).toEqual(['s3'])
  })

  it('keeps an empty section drawn, so a folder does not vanish while you drag into it', () => {
    const a = createSessionSection('A', ['s1'])!
    const empty = createSessionSection('Empty')!

    const groups = groupSessionsBySection([{ id: 's1' }], $sessionSections.get(), $sessionSectionMembership.get())

    expect(groups.map(g => g.id)).toEqual([a.id, empty.id])
    expect(groups[1]!.rows).toEqual([])
  })

  it('omits the Unassigned bucket when everything is filed', () => {
    const a = createSessionSection('A', ['s1', 's2', 's3'])!

    const groups = groupSessionsBySection(rows, $sessionSections.get(), $sessionSectionMembership.get())

    expect(groups.map(g => g.id)).toEqual([a.id])
  })

  it('renders exactly one flat bucket when there are no sections at all', () => {
    const groups = groupSessionsBySection(rows, [], {})

    expect(groups).toEqual([{ id: null, name: null, rows }])
    expect(UNASSIGNED_SESSION_KEY).toBe('section:unassigned')
  })

  it('never invents rows: a filed session the caller did not hand over is not drawn', () => {
    createSessionSection('A', ['s9'])

    const groups = groupSessionsBySection([{ id: 's1' }], $sessionSections.get(), $sessionSectionMembership.get())

    expect(groups[0]!.rows).toEqual([])
    expect(groups[1]!.rows.map(r => r.id)).toEqual(['s1'])
  })
})

describe('profile scope', () => {
  it('keeps each profile’s folders to itself', () => {
    const section = createSessionSection('Mine', ['s1'])!

    $activeGatewayProfile.set('other')
    loadSessionSections()

    expect($sessionSections.get()).toEqual([])
    expect(sessionSectionId('s1')).toBeNull()

    $activeGatewayProfile.set('default')
    loadSessionSections()

    expect($sessionSections.get().map(s => s.id)).toEqual([section.id])
    expect(sessionSectionId('s1')).toBe(section.id)
  })

  it('survives a reload: folders and membership come back from storage', () => {
    const section = createSessionSection('Clients', ['s1'])!

    $sessionSections.set([])
    $sessionSectionMembership.set({})
    loadSessionSections()

    expect($sessionSections.get()).toEqual([{ id: section.id, name: 'Clients' }])
    expect(sessionSectionId('s1')).toBe(section.id)
  })

  it('drops a profile’s folders on profile removal', () => {
    createSessionSection('Gone', ['s1'])

    dropSessionSectionsForProfile('default')

    expect($sessionSections.get()).toEqual([])
    expect(storedProfiles().default).toBeUndefined()
  })

  it('carries folders across a profile rename', () => {
    const section = createSessionSection('Mine', ['s1'])!

    migrateSessionSectionsForProfile('default', 'renamed')

    $activeGatewayProfile.set('renamed')
    loadSessionSections()

    expect($sessionSections.get().map(s => s.id)).toEqual([section.id])
    expect(sessionSectionId('s1')).toBe(section.id)
    expect(storedProfiles().default).toBeUndefined()
  })

  it('is a no-op when the rename is to the same name', () => {
    createSessionSection('Mine', ['s1'])

    migrateSessionSectionsForProfile('default', 'default')

    expect($sessionSections.get()).toHaveLength(1)
  })

  it('does not throw when storage rejects writes', () => {
    const setItem = vi.spyOn(window.localStorage, 'setItem').mockImplementation(() => {
      throw new Error('quota')
    })

    expect(() => createSessionSection('Still works', ['s1'])).not.toThrow()
    expect($sessionSections.get()).toHaveLength(1)

    setItem.mockRestore()
  })
})

describe('drag state', () => {
  it('starts idle and clears back to idle', () => {
    expect($draggingSession.get()).toBeNull()

    $draggingSession.set('s1')
    expect($draggingSession.get()).toBe('s1')

    $draggingSession.set(null)
    expect($draggingSession.get()).toBeNull()
  })
})
