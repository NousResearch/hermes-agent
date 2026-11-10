import { localeAlias } from '@/i18n/languages'
import { normalizeLocaleId } from '@/i18n/registry'

export interface CatalogI18nFields {
  title?: string
  description?: string
}

export type CatalogI18nMap = Record<string, CatalogI18nFields>

export interface CatalogCopySource {
  name: string
  title?: string
  description: string
  i18n?: CatalogI18nMap
}

export interface CatalogCopy {
  title: string
  description: string
}

const text = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

/** ISO 3166-1 alpha-2 (2 letters) or UN M.49 (3 digits). Not a 4-letter script (`hant`). */
function isRegionSubtag(extra: string): boolean {
  return (extra.length === 2 && /^[a-z]+$/i.test(extra)) || (extra.length === 3 && /^\d+$/.test(extra))
}

/** Locale → {title?, description?}. Invalid / empty values dropped; unknown locales kept. */
export function parseCatalogI18n(raw: unknown): CatalogI18nMap {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {}
  }

  const out: CatalogI18nMap = {}

  for (const [loc, fields] of Object.entries(raw as Record<string, unknown>)) {
    const key = normalizeLocaleId(loc)

    if (!key || !fields || typeof fields !== 'object' || Array.isArray(fields)) {
      continue
    }

    const rec = fields as Record<string, unknown>
    const title = text(rec.title)
    const description = text(rec.description)

    if (!title && !description) {
      continue
    }

    out[key] = {
      ...(title ? { title } : {}),
      ...(description ? { description } : {})
    }
  }

  return out
}

/** Exact locale (after alias), then primary subtag when safe (`id-ID` → `id`; not `zh-hant` → `zh`). */
export function catalogLocaleCandidates(locale: string): string[] {
  const raw = normalizeLocaleId(locale)

  if (!raw) {
    return []
  }

  const exact = localeAlias(locale) ?? raw
  const keys = [exact]
  const parts = exact.split('-')
  const extra = parts[1]

  if (parts.length >= 2 && extra && isRegionSubtag(extra) && !keys.includes(parts[0]!)) {
    keys.push(parts[0]!)
  }

  return keys
}

/** Localized title/description for catalog cards. Never changes ``name``. */
export function resolveCatalogCopy(entry: CatalogCopySource, locale: string): CatalogCopy {
  const englishTitle = text(entry.title) || entry.name
  const englishDescription = text(entry.description) || entry.name
  const i18n = entry.i18n ?? {}
  let title = englishTitle
  let description = englishDescription
  let haveTitle = false
  let haveDescription = false

  for (const key of catalogLocaleCandidates(locale)) {
    const loc = i18n[key]

    if (!haveTitle && loc?.title) {
      title = loc.title
      haveTitle = true
    }

    if (!haveDescription && loc?.description) {
      description = loc.description
      haveDescription = true
    }

    if (haveTitle && haveDescription) {
      break
    }
  }

  return { title, description }
}
