import type { Translations } from './types'

// Language picker copy, composed by en.ts.
export const enLanguage = {
  language: {
    label: 'Language',
    description: 'Choose the language for the desktop interface.',
    saving: 'Saving language…',
    saveError: 'Language update failed',
    switchTo: 'Switch language',
    searchPlaceholder: 'Search languages…',
    noResults: 'No languages found'
  }
} satisfies Pick<Translations, 'language'>
