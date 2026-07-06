import { atom } from 'nanostores'

import {
  addSessionsToFolder,
  createSessionFolder,
  deleteSessionFolder,
  listSessionFolders,
  removeSessionsFromFolder
} from '@/hermes'
import type { SessionFolder } from '@/hermes'

export type { SessionFolder } from '@/hermes'

export const $folders = atom<SessionFolder[]>([])
export const $foldersLoading = atom(true)

export async function refreshFolders(profile?: string): Promise<void> {
  try {
    const result = await listSessionFolders(profile)
    $folders.set(result)
  } catch {
    // Silently fail — folders are non-critical UI
  } finally {
    $foldersLoading.set(false)
  }
}

export async function createFolder(name: string, profile?: string): Promise<SessionFolder | null> {
  try {
    const folder = await createSessionFolder(name, profile)
    $folders.set([...$folders.get(), folder])
    return folder
  } catch {
    return null
  }
}

export async function deleteAndRemoveFolder(id: string, profile?: string): Promise<boolean> {
  try {
    await deleteSessionFolder(id, profile)
    $folders.set($folders.get().filter(f => f.id !== id))
    return true
  } catch {
    return false
  }
}

export async function moveToFolder(
  sessionId: string,
  folderId: string,
  currentFolderId?: string | null,
  profile?: string
): Promise<boolean> {
  try {
    if (currentFolderId) {
      await removeSessionsFromFolder(currentFolderId, [sessionId], profile)
    }
    await addSessionsToFolder(folderId, [sessionId], profile)
    await refreshFolders(profile)
    return true
  } catch {
    return false
  }
}

export async function removeFromFolder(
  sessionId: string,
  folderId: string,
  profile?: string
): Promise<boolean> {
  try {
    await removeSessionsFromFolder(folderId, [sessionId], profile)
    await refreshFolders(profile)
    return true
  } catch {
    return false
  }
}
