import { atom } from 'nanostores'

export const PROJECTS_STORAGE_KEY = 'hermes.desktop.projects'

export interface Project {
  id: string
  title: string
  description: string
  path: string
  createdAt: number
}

function storedProjects(): Project[] {
  try {
    const value = window.localStorage.getItem(PROJECTS_STORAGE_KEY)

    if (!value) {
      return []
    }

    const parsed = JSON.parse(value)

    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.filter(
      (item): item is Project =>
        item &&
        typeof item === 'object' &&
        typeof item.id === 'string' &&
        typeof item.title === 'string' &&
        typeof item.description === 'string' &&
        typeof item.path === 'string' &&
        typeof item.createdAt === 'number'
    )
  } catch {
    return []
  }
}

function persistProjects(projects: Project[]) {
  try {
    if (projects.length === 0) {
      window.localStorage.removeItem(PROJECTS_STORAGE_KEY)
    } else {
      window.localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify(projects))
    }
  } catch {
    // Local preference; restricted storage should not break the app.
  }
}

export const $projects = atom<Project[]>(storedProjects())

export const setProjects = (projects: Project[]) => $projects.set(projects)

export function addProject(data: Omit<Project, 'id' | 'createdAt'>): Project {
  const project: Project = {
    ...data,
    id: crypto.randomUUID(),
    createdAt: Date.now()
  }

  $projects.set([...$projects.get(), project])

  return project
}

export function updateProject(id: string, updates: Partial<Omit<Project, 'id' | 'createdAt'>>): void {
  $projects.set($projects.get().map(p => (p.id === id ? { ...p, ...updates } : p)))
}

export function removeProject(id: string): void {
  $projects.set($projects.get().filter(p => p.id !== id))
}

export function getProject(id: string): Project | undefined {
  return $projects.get().find(p => p.id === id)
}

$projects.subscribe(projects => persistProjects([...projects]))
