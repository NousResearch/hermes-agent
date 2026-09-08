/**
 * Resolve Bot Mode message keys against the plugin's own locale bundles.
 *
 * Tests that render Bot Mode components stub `usePluginI18n` with this instead
 * of registering the bundle for real: registration normally happens in
 * `ctx.register`, and the registry lives behind an app-internal module a plugin
 * (or its tests) may not import.
 */

import { BOTS_LOCALES } from './i18n'

export function botsTranslator(locale: 'en' | 'ja' | 'zh' | 'zh-hant') {
  return (key: string, ...args: unknown[]): string => {
    const value = key
      .split('.')
      .reduce<unknown>(
        (node, part) => (node && typeof node === 'object' ? (node as Record<string, unknown>)[part] : undefined),
        BOTS_LOCALES[locale]
      )

    if (typeof value === 'function') {
      return String((value as (...params: unknown[]) => string)(...args))
    }

    return typeof value === 'string' ? value : key
  }
}

export const translateBots = botsTranslator('en')
