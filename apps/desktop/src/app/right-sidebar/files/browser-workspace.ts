import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useEffect } from 'react'

const HISTORY_LIMIT = 50
const WORKSPACE_LIMIT = 24

export interface BrowserWorkspaceState {
  back: string[]
  connectionKey: string
  forward: string[]
  location: string
  sessionKey: string
  sessionRoot: string
}

const initialState: BrowserWorkspaceState = {
  back: [],
  connectionKey: '',
  forward: [],
  location: '',
  sessionKey: '',
  sessionRoot: ''
}

export const $browserWorkspace = atom<BrowserWorkspaceState>(initialState)
const workspaces = new Map<string, BrowserWorkspaceState>()

function registryKey(sessionKey: string, connectionKey: string): string {
  return `${sessionKey}\0${connectionKey}`
}

function setWorkspace(next: BrowserWorkspaceState): void {
  $browserWorkspace.set(next)

  if (next.sessionKey && next.connectionKey) {
    const key = registryKey(next.sessionKey, next.connectionKey)
    workspaces.delete(key)
    workspaces.set(key, next)

    while (workspaces.size > WORKSPACE_LIMIT) {
      const oldest = workspaces.keys().next().value

      if (oldest === undefined) {
        break
      }

      workspaces.delete(oldest)
    }
  }
}

function bounded(items: string[]): string[] {
  return items.slice(-HISTORY_LIMIT)
}

function clean(path: string): string {
  const value = String(path || '').trim()

  if (!value) {
    return ''
  }

  if (value === '/' || /^[A-Za-z]:[\\/]?$/.test(value)) {
    return value
  }

  return value.replace(/[\\/]+$/, '')
}

export function browserParentPath(path: string): string {
  const value = clean(path)

  if (!value || value === '/' || /^[A-Za-z]:[\\/]?$/.test(value)) {
    return value
  }

  const index = Math.max(value.lastIndexOf('/'), value.lastIndexOf('\\'))

  if (index < 0) {
    return value
  }

  if (index === 0) {
    return '/'
  }

  if (index === 2 && /^[A-Za-z]:/.test(value)) {
    return value.slice(0, 3)
  }

  return value.slice(0, index)
}

export function getBrowserWorkspace(): BrowserWorkspaceState {
  return $browserWorkspace.get()
}

export function resetBrowserWorkspace(): void {
  workspaces.clear()
  $browserWorkspace.set(initialState)
}

export function syncBrowserWorkspace(sessionCwd: string, connectionKey: string, sessionKey = 'detached'): void {
  const cwd = clean(sessionCwd)
  const current = $browserWorkspace.get()

  if (current.connectionKey !== connectionKey || current.sessionKey !== sessionKey) {
    const saved = workspaces.get(registryKey(sessionKey, connectionKey))
    setWorkspace(
      saved
        ? { ...saved, sessionRoot: cwd }
        : { back: [], connectionKey, forward: [], location: cwd, sessionKey, sessionRoot: cwd }
    )

    return
  }

  setWorkspace({ ...current, location: current.location || cwd, sessionRoot: cwd })
}

export function browserNavigate(nextPath: string): void {
  const next = clean(nextPath)
  const current = $browserWorkspace.get()

  if (!next || next === current.location) {
    return
  }

  setWorkspace({
    ...current,
    back: current.location ? bounded([...current.back, current.location]) : current.back,
    forward: [],
    location: next
  })
}

export function browserBack(): void {
  const current = $browserWorkspace.get()
  const next = current.back.at(-1)

  if (!next) {
    return
  }

  setWorkspace({
    ...current,
    back: current.back.slice(0, -1),
    forward: current.location ? [current.location, ...current.forward].slice(0, HISTORY_LIMIT) : current.forward,
    location: next
  })
}

export function browserForward(): void {
  const current = $browserWorkspace.get()
  const [next, ...rest] = current.forward

  if (!next) {
    return
  }

  setWorkspace({
    ...current,
    back: current.location ? bounded([...current.back, current.location]) : current.back,
    forward: rest,
    location: next
  })
}

export function browserUp(): void {
  const current = $browserWorkspace.get()
  browserNavigate(browserParentPath(current.location))
}

export function browserSessionRoot(): void {
  browserNavigate($browserWorkspace.get().sessionRoot)
}

export function useBrowserWorkspace(
  sessionCwd: string,
  connectionKey: string,
  sessionKey: string
): BrowserWorkspaceState {
  const state = useStore($browserWorkspace)
  useEffect(() => syncBrowserWorkspace(sessionCwd, connectionKey, sessionKey), [connectionKey, sessionCwd, sessionKey])

  return state
}
