/**
 * The active UI locale, published for consumers outside the React tree.
 *
 * The provider holds the locale in `useState` — the right place for rendering,
 * but invisible to a plugin, which is why the appearance hub observes
 * `<html lang>` with a MutationObserver today. This module is the smallest
 * seam that gives those consumers a subscription without moving the value out
 * of the provider.
 *
 * Two neighbouring signals are **not** this one:
 *
 * - `$appLocaleVersion` bumps when a language pack is registered or dropped,
 *   not when the user picks another language.
 * - `$requestedLocale` is what the profile asked for (`null` = OS inference),
 *   not what the app resolved and actually painted.
 */

import { DEFAULT_LOCALE } from './languages'
import type { Locale } from './types'

let currentAppLocale: Locale = DEFAULT_LOCALE

const listeners = new Set<(locale: Locale) => void>()

/** The locale the app is currently rendering in. */
export function getAppLocale(): Locale {
  return currentAppLocale
}

/**
 * Follow locale changes. Emits the current value immediately, then after every
 * switch — the same contract `host.settings.subscribe` uses.
 */
export function subscribeAppLocale(listener: (locale: Locale) => void): () => void {
  listeners.add(listener)
  listener(currentAppLocale)

  return () => {
    listeners.delete(listener)
  }
}

/**
 * Called by the provider whenever the rendered locale changes. A no-op when
 * the value is unchanged, so a registry-version re-render does not fire
 * listeners that only care about the language.
 */
export function publishAppLocale(locale: Locale): void {
  if (currentAppLocale === locale) {
    return
  }

  currentAppLocale = locale

  for (const listener of listeners) {
    listener(locale)
  }
}
