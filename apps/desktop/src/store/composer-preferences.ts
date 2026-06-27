import { atom } from 'nanostores'

export interface ComposerEditorPreferences {
  language: string
  spellcheck: boolean
}

export function systemComposerLanguage(): string {
  if (typeof navigator === 'undefined') {
    return 'en-US'
  }

  return navigator.language?.trim() || 'en-US'
}

export function normalizeComposerLanguage(value: unknown): string {
  return typeof value === 'string' && value.trim() ? value.trim() : systemComposerLanguage()
}

export function composerEditorPreferencesFromConfig(config: unknown): ComposerEditorPreferences {
  const desktop = config && typeof config === 'object' ? (config as { desktop?: unknown }).desktop : null
  const editor = desktop && typeof desktop === 'object' ? (desktop as { editor?: unknown }).editor : null
  const settings = editor && typeof editor === 'object' ? (editor as { language?: unknown; spellcheck?: unknown }) : null

  return {
    language: normalizeComposerLanguage(settings?.language),
    spellcheck: settings?.spellcheck !== false
  }
}

export const $composerEditorPreferences = atom<ComposerEditorPreferences>(composerEditorPreferencesFromConfig({}))

export function setComposerEditorPreferencesFromConfig(config: unknown): void {
  $composerEditorPreferences.set(composerEditorPreferencesFromConfig(config))
}
