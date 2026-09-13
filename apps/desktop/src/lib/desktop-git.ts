import { hermesApi } from '@/hermes'

import { desktopFsProfile, isDesktopFsRemoteMode } from './desktop-fs'
import { createGitRestBridge, type GitBridge } from './git-rest'

// Remote-aware git facade. Locally the desktop runs git through Electron
// (window.hermesDesktop.git); on a remote gateway that's the wrong filesystem,
// so we mirror the same surface over the dashboard REST API (/api/git/*) — the
// coding rail, worktree lanes, review pane, and branch ops then act on the
// BACKEND repo where sessions actually run. Mirrors desktop-fs.ts.

function desktopApi<T>(path: string, body?: Record<string, unknown>): Promise<T> {
  const desktop = window.hermesDesktop

  if (!desktop) {
    throw new Error('Hermes Desktop bridge is unavailable')
  }

  return hermesApi<T>(
    body ? { body, method: 'POST', path, profile: desktopFsProfile() } : { path, profile: desktopFsProfile() }
  )
}

function gitGet<T>(route: string, params: Record<string, boolean | null | string | undefined>): Promise<T> {
  const query = new URLSearchParams()

  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined) {
      query.set(key, String(value))
    }
  }

  return desktopApi<T>(`/api/git/${route}?${query.toString()}`)
}

function gitPost<T>(route: string, body: Record<string, unknown>): Promise<T> {
  return desktopApi<T>(`/api/git/${route}`, body)
}

const remoteGit = createGitRestBridge({ get: gitGet, post: gitPost })

export function desktopGit(): GitBridge | undefined {
  if (typeof window === 'undefined') {
    return undefined
  }

  return isDesktopFsRemoteMode() ? remoteGit : window.hermesDesktop?.git
}
