/**
 * Display-only translations for Capabilities catalog content (skills, plugins, MCP servers,
 * tools, toolsets, connectors) keyed by the item's stable technical id.
 *
 * Nothing here renames an item: ids, install identifiers, tool names and API calls stay as the
 * backend sends them. A miss returns `null` so callers keep the original text (new catalog rows
 * render untranslated instead of breaking or pretending to be translated).
 *
 * Layers, highest first:
 *   1. `catalogContent` overrides from the merged locale registry — e.g. a user overlay
 *      `$HERMES_HOME/locales/ru.desktop.yaml` (`catalogContent.skill.<id>.name: ...`), no rebuild;
 *   2. the bundled dictionary for the locale (`ru.json`, generated, reviewed offline).
 * No model calls happen at display time.
 */
import { atom } from 'nanostores'

import { CONTENT_DICTIONARY_LOADERS } from './dictionaries'

export type ContentKind = 'skill' | 'plugin' | 'mcp' | 'tool' | 'toolset' | 'connector'
export type CategoryGroup = 'skills' | 'plugins' | 'toolsets' | 'labels'

export interface ContentText {
  name: string
  description?: string
  keywords?: string[]
}

export interface ContentDictionary {
  skill?: Record<string, ContentText>
  plugin?: Record<string, ContentText>
  mcp?: Record<string, ContentText>
  tool?: Record<string, ContentText>
  toolset?: Record<string, ContentText>
  connector?: Record<string, ContentText>
  categories?: Partial<Record<CategoryGroup, Record<string, string>>>
  /** Heading and one-line blurb for the shared plugin taxonomy (`desktop`, `memory`, …). */
  pluginCategories?: Record<string, { label: string; blurb: string }>
}

/** Dictionaries already loaded this session, by language. Filled by `loadContentDictionary`. */
const LOADED: Record<string, ContentDictionary> = {}
const PENDING: Record<string, Promise<void> | undefined> = {}

/** Bumps whenever a dictionary finishes loading, so memoized localization re-derives. */
export const $contentDictionaryVersion = atom(0)

/** `ru-RU`, `ru_RU` → `ru`. */
export const contentLanguage = (locale: string | null | undefined) =>
  String(locale || '').toLowerCase().split(/[-_]/)[0]

/** True when the app ships (or has already loaded) a content dictionary for this language. */
export const hasContentDictionary = (locale: string | null | undefined) =>
  contentLanguage(locale) in CONTENT_DICTIONARY_LOADERS

/**
 * Load a language's dictionary once; resolves immediately for English and for languages without
 * one. A failed load leaves the original text on screen (never blocks or throws into the UI).
 */
export function loadContentDictionary(locale: string | null | undefined): Promise<void> {
  const lang = contentLanguage(locale)
  const loader = CONTENT_DICTIONARY_LOADERS[lang]

  if (!loader || LOADED[lang]) {
    return Promise.resolve()
  }

  PENDING[lang] ??= loader()
    .then(dictionary => {
      LOADED[lang] = dictionary
      $contentDictionaryVersion.set($contentDictionaryVersion.get() + 1)
    })
    .catch(() => {
      PENDING[lang] = undefined
    })

  return PENDING[lang]!
}

/** Test seam: install a dictionary synchronously. */
export function setContentDictionaryForTest(locale: string, dictionary: ContentDictionary | null) {
  const lang = contentLanguage(locale)

  if (dictionary) {
    LOADED[lang] = dictionary
  } else {
    delete LOADED[lang]
  }

  $contentDictionaryVersion.set($contentDictionaryVersion.get() + 1)
}

const isText = (value: unknown): value is ContentText =>
  !!value && typeof value === 'object' && typeof (value as ContentText).name === 'string' && !!(value as ContentText).name.trim()

/** Candidate keys for one item: exact id, lower-case, and the last path segment of `owner/name`. */
export function contentKeys(...ids: (string | null | undefined)[]): string[] {
  const keys: string[] = []

  for (const raw of ids) {
    const id = String(raw ?? '').trim()

    if (!id) {
      continue
    }

    for (const key of [id, id.toLowerCase(), id.split(/[/:]/).pop() ?? '']) {
      if (key && !keys.includes(key)) {
        keys.push(key)
      }
    }
  }

  return keys
}

/** Translation for an item, or null when none exists for this locale. */
export function lookupContent(
  kind: ContentKind,
  ids: string[],
  locale: string | null | undefined,
  overlay?: ContentDictionary | null
): ContentText | null {
  // Connectors and MCP servers share one vocabulary (a connector slug is often an MCP server name).
  const kinds: ContentKind[] = kind === 'connector' ? ['connector', 'mcp'] : kind === 'mcp' ? ['mcp', 'connector'] : [kind]
  const lang = contentLanguage(locale)

  for (const each of kinds) {
    const bundled = LOADED[lang]?.[each]
    const fromOverlay = overlay?.[each]

    for (const key of ids) {
      const hit = fromOverlay?.[key] ?? bundled?.[key]

      if (isText(hit)) {
        return hit
      }
    }
  }

  return null
}

/** Localized label for a category / toolset slug, or null. */
export function lookupCategory(
  slug: string,
  locale: string | null | undefined,
  overlay?: ContentDictionary | null,
  groups: CategoryGroup[] = ['skills', 'plugins', 'toolsets']
): string | null {
  const lang = contentLanguage(locale)
  const key = String(slug ?? '').trim()

  if (!key) {
    return null
  }

  for (const group of groups) {
    const label = overlay?.categories?.[group]?.[key] ?? LOADED[lang]?.categories?.[group]?.[key]

    if (typeof label === 'string' && label.trim()) {
      return label
    }
  }

  return null
}

/** Localized category for a slug or an English display label ("Software Development"), or null. */
export function lookupCategoryLabel(value: string, locale: string | null | undefined, overlay?: ContentDictionary | null): string | null {
  const key = String(value ?? '').trim()

  if (!key || !hasContentDictionary(locale)) {
    return overlay ? lookupCategory(key, locale, overlay) : null
  }

  return lookupCategory(key, locale, overlay) ?? lookupCategory(key.toLowerCase(), locale, overlay, ['labels'])
}

/** Extra lower-cased text for search: translated name, description and keywords. */
export function contentSearchText(text: ContentText | null): string {
  if (!text) {
    return ''
  }

  return [text.name, text.description ?? '', ...(text.keywords ?? [])].join(' ').toLowerCase()
}

/** The overlay subtree a locale registry carries (`t.catalogContent`), if any. */
export function overlayFrom(translations: unknown): ContentDictionary | null {
  const value = (translations as { catalogContent?: unknown } | null)?.catalogContent

  return value && typeof value === 'object' ? (value as ContentDictionary) : null
}

/** Coverage numbers for diagnostics and tests. */
export function dictionaryStats(locale: string) {
  const dict = LOADED[contentLanguage(locale)] ?? {}
  const count = (kind: ContentKind) => Object.keys(dict[kind] ?? {}).length

  return {
    skill: count('skill'),
    plugin: count('plugin'),
    mcp: count('mcp'),
    tool: count('tool'),
    toolset: count('toolset'),
    connector: count('connector')
  }
}

/** Localized plugin-category heading/blurb (shared taxonomy keys such as `memory`), or null. */
export const pluginCategoryText = (key: string, locale: string | null | undefined, overlay?: ContentDictionary | null) =>
  overlay?.pluginCategories?.[key] ?? LOADED[contentLanguage(locale)]?.pluginCategories?.[key] ?? null
