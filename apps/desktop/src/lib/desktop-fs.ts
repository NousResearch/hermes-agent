import type {
  HermesConnection,
  HermesReadDirResult,
  HermesReadFileTextResult,
  HermesSelectPathsOptions,
  HermesWorktreeInfo
} from '@/global'
import { dashboardApi } from '@/lib/browser-dashboard'
import { $connection } from '@/store/session'

export interface DesktopFsRemotePicker {
  selectPaths: (options?: HermesSelectPathsOptions) => Promise<string[]>
}

let remotePicker: DesktopFsRemotePicker | null = null

export function setDesktopFsRemotePicker(next: DesktopFsRemotePicker | null) {
  remotePicker = next
}

function connectionCacheKey(connection: HermesConnection | null) {
  if (!connection) {
    return 'local:'
  }
  return `${connection.mode || 'local'}:${connection.profile || ''}:${connection.baseUrl || ''}`
}

export function desktopFsCacheKey() {
  return connectionCacheKey($connection.get())
}

export function isDesktopFsRemoteMode() {
  return $connection.get()?.mode === 'remote'
}

function fsPath(endpoint: string, filePath: string) {
  return `/api/fs/${endpoint}?path=${encodeURIComponent(filePath)}`
}

function bridge() {
  const desktop = window.hermesDesktop
  if (!desktop) {
    throw new Error('Hermes Desktop bridge is unavailable')
  }
  return desktop
}

export async function readDesktopDir(path: string): Promise<HermesReadDirResult> {
  const desktop = window.hermesDesktop
  if (!isDesktopFsRemoteMode()) {
    return bridge().readDir(path)
  }
  return desktop?.api
    ? desktop.api<HermesReadDirResult>({ path: fsPath('list', path) })
    : dashboardApi<HermesReadDirResult>({ path: fsPath('list', path) })
}

export async function readDesktopFileText(path: string): Promise<HermesReadFileTextResult> {
  const desktop = window.hermesDesktop
  if (!isDesktopFsRemoteMode()) {
    return bridge().readFileText(path)
  }
  return desktop?.api
    ? desktop.api<HermesReadFileTextResult>({ path: fsPath('read-text', path) })
    : dashboardApi<HermesReadFileTextResult>({ path: fsPath('read-text', path) })
}

export async function readDesktopFileDataUrl(path: string): Promise<string> {
  const desktop = window.hermesDesktop
  if (!isDesktopFsRemoteMode()) {
    return bridge().readFileDataUrl(path)
  }

  const result = desktop?.api
    ? await desktop.api<string | { dataUrl?: string }>({ path: fsPath('read-data-url', path) })
    : await dashboardApi<string | { dataUrl?: string }>({ path: fsPath('read-data-url', path) })
  return typeof result === 'string' ? result : result.dataUrl || ''
}

export async function desktopGitRoot(path: string): Promise<string | null> {
  if (!isDesktopFsRemoteMode()) {
    const desktop = bridge()
    return desktop.gitRoot ? desktop.gitRoot(path) : null
  }

  const desktop = window.hermesDesktop
  const result = desktop?.api
    ? await desktop.api<{ root: string | null }>({ path: fsPath('git-root', path) })
    : await dashboardApi<{ root: string | null }>({ path: fsPath('git-root', path) })
  return result.root
}

// Worktree detection runs against the LOCAL filesystem (the electron main
// process). For a remote backend the session cwds live on another machine, so
// we can't resolve them here — callers fall back to the path-name heuristic.
export async function desktopWorktrees(cwds: string[]): Promise<Record<string, HermesWorktreeInfo | null>> {
  if (isDesktopFsRemoteMode()) {
    return {}
  }

  const desktop = bridge()

  return desktop.worktrees ? desktop.worktrees(cwds) : {}
}

export async function desktopDefaultCwd(): Promise<{ branch: string; cwd: string } | null> {
  if (!isDesktopFsRemoteMode()) {
    return null
  }

  return window.hermesDesktop?.api
    ? window.hermesDesktop.api<{ branch: string; cwd: string }>({ path: '/api/fs/default-cwd' })
    : dashboardApi<{ branch: string; cwd: string }>({ path: '/api/fs/default-cwd' })
}

export async function selectDesktopPaths(options?: HermesSelectPathsOptions): Promise<string[]> {
  if (!isDesktopFsRemoteMode()) {
    return bridge().selectPaths(options)
  }
  if (!options?.directories || options.multiple !== false) {
    return []
  }
  return remotePicker ? remotePicker.selectPaths(options) : []
}
