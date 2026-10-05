import { type Locale, LOCALE_METADATA, localeDirection, LOCALES } from './locale-registry.js'

// Locale scaffolding shared by the desktop and web i18n layers. Generic over the
// translation catalog type: each app supplies its own `Translations`/`en` and
// wraps `mergeTranslations` in a one-line `defineLocale`.

/** Partial-locale shape: every key optional, but functions/arrays are atomic and
 *  unknown keys still fail the type-check. */
export type TranslationOverride<T> = T extends (...args: never[]) => string
  ? T
  : T extends readonly unknown[]
    ? T
    : T extends string
      ? string
      : T extends object
        ? { [K in keyof T]?: TranslationOverride<T[K]> }
        : T

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Deep-merge a partial locale over the English base: nested records recurse,
 *  everything else (strings, functions, arrays) is replaced wholesale. */
export function mergeTranslations<T>(base: T, overrides: TranslationOverride<T> | undefined): T {
  if (!isRecord(base) || !isRecord(overrides)) {
    return (overrides ?? base) as T
  }

  const result: Record<string, unknown> = { ...base }

  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) {
      continue
    }

    const baseValue = result[key]
    result[key] = isRecord(baseValue) && isRecord(value) ? mergeTranslations(baseValue, value) : value
  }

  return result as T
}

/** Render a language-pack template with POSITIONAL placeholders (`{0}`, `{1}`)
 *  against the arguments the English function entry would have received. A
 *  pack author writes `"{0} of {1} tools on"` for `(on, total) => …`; an
 *  index the template names but the call didn't supply renders empty. */
export function formatPositional(template: string, args: readonly unknown[]): string {
  return template.replace(/\{(\d+)\}/g, (_match, index: string) => {
    const value = args[Number(index)]

    return value === undefined || value === null ? '' : String(value)
  })
}

/** Where the base catalog holds a function (`(n) => string`) and a pack
 *  supplies a plain string, wrap the string into a positional formatter so
 *  the merged catalog keeps the base's call shape. Records recurse; every
 *  other shape passes through untouched (a string over a string, a function
 *  over a function, arrays wholesale). Packs come from YAML, which cannot
 *  express functions — this is the only bridge between the two. */
export function adaptStringOverrides(base: unknown, overrides: unknown): unknown {
  if (typeof base === 'function' && typeof overrides === 'string') {
    return (...args: unknown[]) => formatPositional(overrides, args)
  }

  if (!isRecord(base) || !isRecord(overrides)) {
    return overrides
  }

  const result: Record<string, unknown> = {}

  for (const [key, value] of Object.entries(overrides)) {
    result[key] = adaptStringOverrides(base[key], value)
  }

  return result
}

/** Message leaf a language pack can carry after parsing (YAML is text-only). */
export type FlatMessages = Record<string, string>

/** `{ "a.b.c": "x" }` → `{ a: { b: { c: "x" } } }`. Keys are split on `.`,
 *  except where `base` (the catalog the pack overlays) has a key that itself
 *  contains dots (`keybinds.actions["session.slot.1"]`): at each level the
 *  longest segment run naming an existing base key wins, so dotted leaf keys
 *  round-trip. A flat key that collides with an earlier branch is dropped
 *  rather than clobbering it, so one bad line can't erase a subtree. */
export function unflattenMessages(flat: FlatMessages, base?: unknown): Record<string, unknown> {
  const root: Record<string, unknown> = {}

  for (const [path, value] of Object.entries(flat)) {
    const parts = path.split('.')
    let cursor = root
    let guide: unknown = base
    let index = 0

    while (index < parts.length) {
      // Prefer the longest run of segments that names a key in the base guide;
      // without a guide (or a match) fall back to one segment at a time.
      let take = 1

      if (isRecord(guide)) {
        for (let end = parts.length; end > index; end -= 1) {
          if (Object.hasOwn(guide, parts.slice(index, end).join('.'))) {
            take = end - index

            break
          }
        }
      }

      const key = parts.slice(index, index + take).join('.')
      index += take
      const last = index >= parts.length

      if (last) {
        if (!isRecord(cursor[key])) {
          cursor[key] = value
        }

        break
      }

      const next = cursor[key]

      if (next === undefined) {
        const branch: Record<string, unknown> = {}
        cursor[key] = branch
        cursor = branch
      } else if (isRecord(next)) {
        cursor = next
      } else {
        break
      }

      guide = isRecord(guide) ? guide[key] : undefined
    }
  }

  return root
}

/** Every leaf's dotted path, sorted. Function-valued leaves are keys too
 *  (a pack overrides them with a positional-placeholder string); arrays are
 *  one leaf. This is the key set `locales/_keys.<surface>.json` publishes for
 *  pack validation. */
export function flattenMessageKeys(tree: unknown, prefix = ''): string[] {
  if (!isRecord(tree)) {
    return prefix ? [prefix] : []
  }

  const keys: string[] = []

  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key

    if (isRecord(value)) {
      keys.push(...flattenMessageKeys(value, path))
    } else {
      keys.push(path)
    }
  }

  return keys.sort()
}

// Endonyms (native names) for the language pickers so users recognize their
// language regardless of the current UI language. No country flags: languages
// are not countries (English ≠ GB, Portuguese ≠ PT, Chinese variants ≠ any
// single jurisdiction). Desktop supports a subset of these ids; web all of them.
export const LOCALE_ENDONYMS = Object.fromEntries(
  LOCALES.map(locale => [locale, LOCALE_METADATA[locale].name])
) as Record<Locale, string>
export type EndonymLocale = Locale
export const RTL_LOCALES: ReadonlySet<string> = new Set(LOCALES.filter(locale => localeDirection(locale) === 'rtl'))
