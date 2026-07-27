/**
 * Plugin-scoped i18n — the `ctx.storage` analog for locale bundles. A plugin
 * ships its own strings and registers them under its id; it never edits core
 * `en.ts`. Resolution mirrors the app translator: active locale → the plugin's
 * own `en` bundle → the key itself. The active locale is always the app's
 * (`display.language`) — plugins follow the user's choice, they don't own it.
 *
 * Two consumers, same shape as core (`useI18n` / `translateNow`):
 *  - `usePluginI18n(id)` — reactive translator for React UI (re-renders on a
 *    locale switch or a late bundle registration);
 *  - `ctx.i18n.t` — module-level translator for handlers/stores (non-reactive).
 */

import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback } from 'react'

import { useI18n } from './context'
import { getRuntimeI18nLocale, translateFrom } from './runtime'
import type { Locale } from './types'

/** A leaf message: a literal or an interpolator (`n => `${n} left``). */
export type PluginMessageValue = string | ((...args: never[]) => string)

/** A plugin's messages for one locale — nested trees allowed, addressed by
 *  dot-path (`panel.title`). */
export interface PluginMessages {
  [key: string]: PluginMessages | PluginMessageValue
}

/** Locale → messages. Keyed by the app's locales so autocomplete guides you;
 *  a bundle for a locale the app can't select is simply never resolved. */
export type PluginLocaleBundles = Partial<Record<Locale, PluginMessages>>

/** Resolve `key` for this plugin against `args`; falls back to English, then
 *  the raw key. */
export type PluginTranslate = (key: string, ...args: unknown[]) => string

export interface PluginI18n {
  /** Merge locale bundles for this plugin (call once at `register`). Returns a
   *  disposer that drops the plugin's bundles on unload/reload. */
  register: (bundles: PluginLocaleBundles) => () => void
  /** Module-level translator against the app's active locale (mirrors
   *  `translateNow`). Non-reactive — in React prefer `usePluginI18n`. */
  t: PluginTranslate
}

const registry = new Map<string, Map<Locale, PluginMessages>>()

/** Bumps whenever a plugin's bundles change, so React translators re-render on
 *  a registration that lands after first paint. */
const $version = atom(0)

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function mergeMessages(base: PluginMessages, overrides: PluginMessages): PluginMessages {
  const result: PluginMessages = { ...base }

  for (const [key, value] of Object.entries(overrides)) {
    const prev = result[key]
    result[key] = isRecord(prev) && isRecord(value) ? mergeMessages(prev, value) : value
  }

  return result
}

export function registerPluginLocales(pluginId: string, bundles: PluginLocaleBundles): () => void {
  const byLocale = registry.get(pluginId) ?? new Map<Locale, PluginMessages>()
  registry.set(pluginId, byLocale)

  for (const [locale, messages] of Object.entries(bundles) as [Locale, PluginMessages | undefined][]) {
    if (!messages) {
      continue
    }

    const prev = byLocale.get(locale)
    byLocale.set(locale, prev ? mergeMessages(prev, messages) : messages)
  }

  $version.set($version.get() + 1)

  return () => {
    registry.delete(pluginId)
    $version.set($version.get() + 1)
  }
}

export function translatePlugin(pluginId: string, locale: Locale, key: string, args: unknown[]): string {
  return translateFrom(l => registry.get(pluginId)?.get(l), locale, key, args)
}

/** Build the `ctx.i18n` door for a plugin. `track` records the disposer so the
 *  loader tears bundles down on unload (same lifecycle as `register`/`socket`). */
export function createPluginI18n(pluginId: string, track: (dispose: () => void) => () => void): PluginI18n {
  return {
    register: bundles => track(registerPluginLocales(pluginId, bundles)),
    t: (key, ...args) => translatePlugin(pluginId, getRuntimeI18nLocale(), key, args)
  }
}

/** Reactive scoped translator for React UI. Re-renders on a locale switch or a
 *  late bundle registration. Pass your plugin id (your default export's `id`). */
export function usePluginI18n(pluginId: string): PluginTranslate {
  const { locale } = useI18n()

  // Subscribe so a bundle registered after mount repaints; `translatePlugin`
  // reads the live registry, so the memoized closure needs only id + locale.
  useStore($version)

  return useCallback(
    (key: string, ...args: unknown[]) => translatePlugin(pluginId, locale, key, args),
    [pluginId, locale]
  )
}
