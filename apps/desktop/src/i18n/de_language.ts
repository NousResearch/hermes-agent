import type { LanguageTranslations } from './types_language'

export const deLanguage = {
  label: 'Sprache',
  description: 'Wählen Sie die Sprache der Desktop-Oberfläche.',
  saving: 'Sprache wird gespeichert…',
  saveError: 'Sprachupdate fehlgeschlagen',
  switchTo: 'Sprache wechseln',
  searchPlaceholder: 'Sprachen suchen…',
  noResults: 'Keine Sprachen gefunden'
} satisfies Partial<LanguageTranslations>
