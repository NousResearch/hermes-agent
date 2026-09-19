// The IDE editor's tab model: which files are open, which is active, and which
// have unsaved edits. The open set + active tab persist under an IDE-owned key;
// dirty state is session-only by design — a reload must never claim unsaved
// work it cannot restore.

import { atom } from 'nanostores'

import { type Codec, persistentAtom } from '@/lib/persisted'

export interface IdeEditorState {
  activePath: null | string
  openPaths: string[]
}

export const IDE_EDITOR_STORAGE_KEY = 'hermes.desktop.ideEditor.v1'

const EMPTY: IdeEditorState = { activePath: null, openPaths: [] }

const codec: Codec<IdeEditorState> = {
  decode: raw => {
    try {
      const parsed = JSON.parse(raw) as null | Partial<IdeEditorState> | undefined

      if (!parsed || typeof parsed !== 'object') {
        return EMPTY
      }

      const openPaths = Array.isArray(parsed.openPaths)
        ? parsed.openPaths.filter((path): path is string => typeof path === 'string' && path.length > 0)
        : []

      const activePath =
        typeof parsed.activePath === 'string' && openPaths.includes(parsed.activePath)
          ? parsed.activePath
          : (openPaths[openPaths.length - 1] ?? null)

      return { activePath, openPaths }
    } catch {
      return EMPTY
    }
  },
  encode: value => JSON.stringify(value)
}

export const $ideEditor = persistentAtom<IdeEditorState>(IDE_EDITOR_STORAGE_KEY, EMPTY, codec)

/** Absolute paths with unsaved edits. Session-only (never persisted). */
export const $ideDirtyPaths = atom<string[]>([])

export function activateIdeFile(path: string) {
  const state = $ideEditor.get()

  if (state.openPaths.includes(path) && state.activePath !== path) {
    $ideEditor.set({ ...state, activePath: path })
  }
}

export function clearIdeFileDirty(path: string) {
  const current = $ideDirtyPaths.get()

  if (current.includes(path)) {
    $ideDirtyPaths.set(current.filter(item => item !== path))
  }
}

export function closeIdeFile(path: string) {
  const state = $ideEditor.get()
  const openPaths = state.openPaths.filter(item => item !== path)
  const activePath = state.activePath === path ? (openPaths[openPaths.length - 1] ?? null) : state.activePath

  $ideEditor.set({ activePath, openPaths })
  clearIdeFileDirty(path)
}

export function openIdeFile(path: string) {
  const state = $ideEditor.get()
  const openPaths = state.openPaths.includes(path) ? state.openPaths : [...state.openPaths, path]

  $ideEditor.set({ activePath: path, openPaths })
}

export function setIdeFileDirty(path: string, dirty: boolean) {
  const current = $ideDirtyPaths.get()
  const has = current.includes(path)

  if (dirty === has) {
    return
  }

  $ideDirtyPaths.set(dirty ? [...current, path] : current.filter(item => item !== path))
}
