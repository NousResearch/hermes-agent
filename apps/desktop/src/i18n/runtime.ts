import { isRecord } from '@hermes/shared/i18n'
import { atom } from 'nanostores'

import { DEFAULT_LOCALE } from './languages'
import { resolveTranslations } from './registry'
import type { Locale, Translations } from './types'

const $runtimeLocale = atom<Locale>(DEFAULT_LOCALE)

/** The language `display.language` asked for, normalized but NOT yet checked
 *  against what the app can render. A pack-only language (`pl`) lands here
 *  before its pack is fetched; the backend-pack sync reads it to know which
 *  `i18n.catalog` to pull, and the provider promotes it to the active locale
 *  once the registry knows it. `null` = nothing saved (OS inference). */
export const $requestedLocale = atom<null | string>(null)

/** Walk a dot-path (`a.b.c`) into a nested message tree. */
function resolvePath(source: unknown, key: string): unknown {
  return key.split('.').reduce<unknown>((current, part) => (isRecord(current) ? current[part] : undefined), source)
}

/** A string is returned as-is, a function is called with `args`, else `null`. */
function render(value: unknown, args: unknown[]): null | string {
  if (typeof value === 'string') {
    return value
  }

  if (typeof value === 'function') {
    return (value as (...args: unknown[]) => string)(...args)
  }

  return null
}

/** The active → DEFAULT → key resolution every translator shares. `source`
 *  yields a message tree per locale — the app catalog, or a plugin's bundles. */
export function translateFrom(
  source: (locale: Locale) => unknown,
  locale: Locale,
  key: string,
  args: unknown[]
): string {
  const active = render(resolvePath(source(locale), key), args)

  if (active !== null) {
    return active
  }

  if (locale !== DEFAULT_LOCALE) {
    const fallback = render(resolvePath(source(DEFAULT_LOCALE), key), args)

    if (fallback !== null) {
      return fallback
    }
  }

  return key
}

export function setRuntimeI18nLocale(locale: Locale) {
  $runtimeLocale.set(locale)
}

/** Observe changes to the locale used by non-React plugin contributions. */
export const subscribeRuntimeI18nLocale = $runtimeLocale.listen

/** The locale module-level translators resolve against (the app's active
 *  `display.language`). Plugin `ctx.i18n.t` reads this too. */
export function getRuntimeI18nLocale(): Locale {
  return $runtimeLocale.get()
}

/** The merged catalog for the active runtime locale (bundled + registered
 *  packs) — for module-level code that reads whole copy blocks rather than
 *  one key. React should keep using `useI18n().t`. */
export function runtimeTranslations(): Translations {
  return resolveTranslations($runtimeLocale.get())
}

export function translateNow(key: string, ...args: unknown[]): string {
  return translateFrom(resolveTranslations, $runtimeLocale.get(), key, args)
}
