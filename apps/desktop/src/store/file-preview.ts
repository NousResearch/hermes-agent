import { atom } from 'nanostores'

export interface FilePreviewDirty {
  content: string
  original: string
  path: string
}

export const $filePreviewDirty = atom<FilePreviewDirty | null>(null)
export const $filePreviewEditing = atom(false)
export const $fileSaving = atom(false)

export function startEditing(path: string, originalContent: string) {
  $filePreviewEditing.set(true)
  $filePreviewDirty.set({ content: originalContent, original: originalContent, path })
}

export function updateEditingContent(content: string) {
  const current = $filePreviewDirty.get()
  if (current) {
    $filePreviewDirty.set({ ...current, content })
  }
}

export function cancelEditing() {
  $filePreviewEditing.set(false)
  $filePreviewDirty.set(null)
}

export function saveEditing() {
  $filePreviewEditing.set(false)
  $filePreviewDirty.set(null)
}

export function isDirty(): boolean {
  const d = $filePreviewDirty.get()
  return d !== null && d.content !== d.original
}
