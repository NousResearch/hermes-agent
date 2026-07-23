import { Codecs, persistentAtom } from '@/lib/persisted'

// Device-local presentation preference. The file browser is renderer-owned,
// so this stays with the feature instead of becoming backend config that could
// bleed across machines or profiles. Default true preserves existing behavior.
const SHOW_HIDDEN_FILES_STORAGE_KEY = 'hermes.desktop.fileBrowser.showHiddenFiles'

export const $showHiddenFiles = persistentAtom(SHOW_HIDDEN_FILES_STORAGE_KEY, true, Codecs.bool)

export function isHiddenFileName(name: string): boolean {
  return name.length > 1 && name.startsWith('.')
}

export function setShowHiddenFiles(show: boolean): void {
  $showHiddenFiles.set(show)
}

export function toggleShowHiddenFiles(): void {
  setShowHiddenFiles(!$showHiddenFiles.get())
}
