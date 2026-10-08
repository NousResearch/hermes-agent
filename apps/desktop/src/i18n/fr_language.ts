import type { LanguageTranslations } from './types_language'

export const frLanguage = {
  label: 'Langue',
  description: "Choisissez la langue de l'interface du desktop.",
  saving: 'Enregistrement de la langue…',
  saveError: 'Échec de la mise à jour de la langue',
  switchTo: 'Changer de langue',
  searchPlaceholder: 'Rechercher des langues…',
  noResults: 'Aucune langue trouvée'
} satisfies Partial<LanguageTranslations>
