import type { Translations } from '@/i18n/types'

import type { CatalogEntry } from './catalog-data'

/** Translate known first-party descriptions for display/search only. Community
 * and user-authored skills can reuse a name and must keep their own description. */
export function localizeSkillEntry(entry: CatalogEntry, copy: Translations['skills']): CatalogEntry {
  const official = entry.source === 'built-in' || entry.source === 'optional'
  const description = official ? (copy.skillDescriptions?.[entry.name] ?? entry.description) : entry.description
  const categoryLabel = copy.skillCategoryNames?.[entry.category] ?? entry.categoryLabel

  if (description === entry.description && categoryLabel === entry.categoryLabel) {
    return entry
  }

  return {
    ...entry,
    description,
    categoryLabel,
    search: `${entry.search} ${description} ${categoryLabel}`.normalize('NFC').toLowerCase()
  }
}
