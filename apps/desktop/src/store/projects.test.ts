import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $projects,
  addProject,
  getProject,
  PROJECTS_STORAGE_KEY,
  removeProject,
  setProjects,
  updateProject
} from './projects'

beforeEach(() => {
  setProjects([])
})

afterEach(() => {
  setProjects([])
})

describe('addProject', () => {
  it('creates a project with auto id and createdAt', () => {
    const before = Date.now()
    const project = addProject({ title: 'My Project', description: 'A test project', path: '/home/user/project' })
    const after = Date.now()

    expect(project.id).toBeTypeOf('string')
    expect(project.id.length).toBeGreaterThan(0)
    expect(project.createdAt).toBeGreaterThanOrEqual(before)
    expect(project.createdAt).toBeLessThanOrEqual(after)
    expect(project.title).toBe('My Project')
    expect(project.description).toBe('A test project')
    expect(project.path).toBe('/home/user/project')
  })

  it('adds the project to the store', () => {
    addProject({ title: 'P1', description: 'desc', path: '/p1' })
    addProject({ title: 'P2', description: 'desc2', path: '/p2' })

    expect($projects.get()).toHaveLength(2)
  })

  it('assigns unique ids', () => {
    const p1 = addProject({ title: 'A', description: '', path: '/a' })
    const p2 = addProject({ title: 'B', description: '', path: '/b' })

    expect(p1.id).not.toBe(p2.id)
  })
})

describe('updateProject', () => {
  it('changes the specified fields', () => {
    const p = addProject({ title: 'Old Title', description: 'old', path: '/old' })

    updateProject(p.id, { title: 'New Title', description: 'new' })

    const updated = getProject(p.id)

    expect(updated?.title).toBe('New Title')
    expect(updated?.description).toBe('new')
    expect(updated?.path).toBe('/old')
  })

  it('does not mutate id or createdAt', () => {
    const p = addProject({ title: 'T', description: 'd', path: '/t' })

    updateProject(p.id, { title: 'T2' })

    const updated = getProject(p.id)

    expect(updated?.id).toBe(p.id)
    expect(updated?.createdAt).toBe(p.createdAt)
  })

  it('ignores unknown id', () => {
    addProject({ title: 'T', description: 'd', path: '/t' })

    expect(() => updateProject('nonexistent-id', { title: 'X' })).not.toThrow()
    expect($projects.get()).toHaveLength(1)
  })
})

describe('removeProject', () => {
  it('deletes project by id', () => {
    const p1 = addProject({ title: 'A', description: '', path: '/a' })
    const p2 = addProject({ title: 'B', description: '', path: '/b' })

    removeProject(p1.id)

    const remaining = $projects.get()

    expect(remaining).toHaveLength(1)
    expect(remaining[0].id).toBe(p2.id)
  })

  it('does nothing for unknown id', () => {
    addProject({ title: 'A', description: '', path: '/a' })

    removeProject('nonexistent-id')

    expect($projects.get()).toHaveLength(1)
  })
})

describe('getProject', () => {
  it('returns the correct project by id', () => {
    const p = addProject({ title: 'Find Me', description: 'here', path: '/find' })
    const found = getProject(p.id)

    expect(found).toBeDefined()
    expect(found?.id).toBe(p.id)
    expect(found?.title).toBe('Find Me')
  })

  it('returns undefined for unknown id', () => {
    expect(getProject('nope')).toBeUndefined()
  })
})

describe('persistence key', () => {
  it('uses the correct storage key', () => {
    expect(PROJECTS_STORAGE_KEY).toBe('hermes.desktop.projects')
  })

  it('persists projects to localStorage on change', () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem')

    addProject({ title: 'Persisted', description: 'desc', path: '/p' })

    expect(setItemSpy).toHaveBeenCalledWith(PROJECTS_STORAGE_KEY, expect.any(String))

    setItemSpy.mockRestore()
  })
})
