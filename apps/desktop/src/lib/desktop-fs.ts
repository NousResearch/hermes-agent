import { $connection } from '@/store/session'

import type { HermesConnection, HermesReadDirResult, HermesReadFileTextResult, HermesSelectPathsOptions } from '@/global'

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
  const desktop = bridge()
  if (!isDesktopFsRemoteMode()) {
    return desktop.readDir(path)
  }
  return desktop.api<HermesReadDirResult>({ path: fsPath('list', path) })
}

export async function readDesktopFileText(path: string): Promise<HermesReadFileTextResult> {
  const desktop = bridge()
  if (!isDesktopFsRemoteMode()) {
    return desktop.readFileText(path)
  }
  return desktop.api<HermesReadFileTextResult>({ path: fsPath('read-text', path) })
}

export async function readDesktopFileDataUrl(path: string): Promise<string> {
  const desktop = bridge()
  if (!isDesktopFsRemoteMode()) {
    return desktop.readFileDataUrl(path)
  }

  const result = await desktop.api<string | { dataUrl?: string }>({ path: fsPath('read-data-url', path) })
  return typeof result === 'string' ? result : result.dataUrl || ''
}

export async function desktopGitRoot(path: string): Promise<string | null> {
  const desktop = bridge()
  if (!isDesktopFsRemoteMode()) {
    return desktop.gitRoot ? desktop.gitRoot(path) : null
  }

  const result = await desktop.api<{ root: string | null }>({ path: fsPath('git-root', path) })
  return result.root
}

export async function selectDesktopPaths(options?: HermesSelectPathsOptions): Promise<string[]> {
  const desktop = bridge()
  if (!isDesktopFsRemoteMode()) {
    return desktop.selectPaths(options)
  }
  if (!options?.directories || options.multiple !== false) {
    return []
  }
  return remotePicker ? remotePicker.selectPaths(options) : []
}
