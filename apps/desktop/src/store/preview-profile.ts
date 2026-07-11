import { atom } from 'nanostores'

export function normalizePreviewProfileKey(name: string | null | undefined): string {
  return (name ?? '').trim() || 'default'
}

export const $previewProfileKey = atom('default')

export function setPreviewProfileKey(name: string | null | undefined): void {
  $previewProfileKey.set(normalizePreviewProfileKey(name))
}
