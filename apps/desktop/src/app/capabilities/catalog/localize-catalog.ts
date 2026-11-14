/**
 * Applies the display-only content dictionary (`@/i18n/catalog-content`) to catalog entries.
 * Ids, install identifiers and the original text survive on the entry; search covers both
 * languages. Entries without a translation are returned unchanged (same object).
 */
import {
  $contentDictionaryVersion,
  type ContentDictionary,
  contentKeys,
  contentLanguage,
  contentSearchText,
  hasContentDictionary,
  lookupCategoryLabel,
  lookupContent,
  overlayFrom
} from '@/i18n/catalog-content'
import { getRuntimeI18nLocale, runtimeTranslations } from '@/i18n/runtime'

import { type CatalogEntry, type CatalogKind, catalogLabel } from './catalog-data'

interface Cached {
  key: string
  value: CatalogEntry
}

// Catalog rows are stable objects from the query cache (100k+ skills), so one result per row and
// locale/overlay keeps re-renders at a WeakMap lookup instead of re-deriving every row.
const cache = new WeakMap<CatalogEntry, Cached>()
let overlayIds = new WeakMap<object, number>()
let nextOverlayId = 1

const overlayKey = (overlay: ContentDictionary | null | undefined) => {
  if (!overlay) {
    return 0
  }

  let id = overlayIds.get(overlay)

  if (!id) {
    id = nextOverlayId++
    overlayIds.set(overlay, id)
  }

  return id
}

export function localizeCatalogEntry(
  entry: CatalogEntry,
  kind: CatalogKind,
  locale: string,
  overlay?: ContentDictionary | null
): CatalogEntry {
  const key = `${contentLanguage(locale)}:${overlayKey(overlay)}:${$contentDictionaryVersion.get()}`
  const hit = cache.get(entry)

  if (hit?.key === key) {
    return hit.value
  }

  const contentKind = kind === 'plugins' ? 'plugin' : 'skill'

  // Exact ids first (no allocation per row); the normalized variants only on a miss.
  const text =
    lookupContent(contentKind, [entry.name], locale, overlay) ??
    (entry.identifier !== entry.name ? lookupContent(contentKind, [entry.identifier], locale, overlay) : null) ??
    (needsNormalizedKeys(entry) ? lookupContent(contentKind, contentKeys(entry.identifier, entry.name, entry.installIdentifier), locale, overlay) : null)

  // Categories are translated where they are drawn (`categoryText`): rewriting them here would
  // copy every one of the ~100k feed rows.
  const value: CatalogEntry = text
      ? {
          ...entry,
          name: text?.name ?? entry.name,
          description: text?.description ?? entry.description,
          originalName: text ? entry.name : undefined,
          originalDescription: text?.description ? entry.description : undefined,
          search: `${entry.search} ${contentSearchText(text)}`
        }
      : entry

  cache.set(entry, { key, value })

  return value
}

const needsNormalizedKeys = (entry: CatalogEntry) =>
  /[/:A-Z]/.test(entry.name) || /[/:A-Z]/.test(entry.identifier)

export function localizeCatalogEntries(
  entries: CatalogEntry[],
  kind: CatalogKind,
  locale: string,
  overlay?: ContentDictionary | null
): CatalogEntry[] {
  if (!hasContentDictionary(locale) && !overlay) {
    return entries
  }

  return entries.map(entry => localizeCatalogEntry(entry, kind, locale, overlay))
}

/** Category label for display: the translation for the active locale, else the slug in words. */
export const categoryText = (value: string) =>
  lookupCategoryLabel(value, getRuntimeI18nLocale(), overlayFrom(runtimeTranslations())) ?? catalogLabel(value)

/** "Ethos Explore · NVIDIA": the original title next to the author when the title is translated. */
export function catalogByline(entry: CatalogEntry): string {
  const author = entry.author || catalogLabel(entry.source)

  return entry.originalName ? `${catalogLabel(entry.originalName)} · ${author}` : author
}

/** Test seam: forget cached rows (e.g. between locale-switch assertions). */
export function resetCatalogLocalization() {
  overlayIds = new WeakMap()
}
