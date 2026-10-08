import type { LanguageTranslations } from './types_language'

export const esLanguage = {
  label: 'Idioma',
  description: 'Elige el idioma de la interfaz de escritorio.',
  saving: 'Guardando idioma…',
  saveError: 'No se pudo actualizar el idioma',
  switchTo: 'Cambiar idioma',
  searchPlaceholder: 'Buscar idiomas…',
  noResults: 'No se encontraron idiomas'
} satisfies Partial<LanguageTranslations>
