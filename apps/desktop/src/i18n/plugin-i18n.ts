/**
 * Plugin-scoped i18n — the `ctx.storage` analog for locale bundles. A plugin
 * ships its own strings and registers them under its id; it never edits core
 * `en.ts`. Resolution mirrors the app translator: active locale → the plugin's
 * own `en` bundle → the key itself. The active locale is always the app's
 * (`display.language`) — plugins follow the user's choice, they don't own it.
 *
 * Shell wording mirrors the app catalog too (`registry.ts` `resolveTranslations`):
 * a plugin may register a `redesign` layer holding ONLY the strings whose words
 * differ in the redesigned shell. While this window renders that shell the
 * layer is merged over the plugin's bundles, per locale, before the English
 * fallback; classic reads the bundles verbatim. Without a layer a plugin reads
 * the same in both shells.
 *
 * Two consumers, same shape as core (`useI18n` / `translateNow`):
 *  - `usePluginI18n(id)` — reactive translator for React UI (re-renders on a
 *    locale switch or a late bundle registration);
 *  - `ctx.i18n.t` — module-level translator for handlers/stores (non-reactive).
 */

import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useCallback } from 'react'

import { usePortalPresentation } from '@/store/shell-presentation'

import { useI18n } from './context'
import {
  type AppLocaleRegistration,
  currentShellWording,
  registerAppLocale,
  type ShellWording,
  shellWordingFor
} from './registry'
import { getRuntimeI18nLocale, subscribeRuntimeI18nLocale, translateFrom } from './runtime'
import type { BundledLocale, Locale } from './types'

/** A leaf message: a literal or an interpolator (`n => `${n} left``). */
export type PluginMessageValue = string | ((...args: never[]) => string)

/** A plugin's messages for one locale — nested trees allowed, addressed by
 *  dot-path (`panel.title`). */
export interface PluginMessages {
  [key: string]: PluginMessages | PluginMessageValue
}

/** Locale → messages. The bundled ids autocomplete; any other id (a language
 *  a pack plugin registered with `registerAppLocale`) is accepted too. A
 *  bundle for a locale the app can't select is simply never resolved. */
export type PluginLocaleBundles = Partial<Record<BundledLocale, PluginMessages>> & {
  [locale: Locale]: PluginMessages | undefined
}

/** Resolve `key` for this plugin against `args`; falls back to English, then
 *  the raw key. */
export type PluginTranslate = (key: string, ...args: unknown[]) => string

/** Per-shell wording layers over a plugin's bundles: each holds only the keys
 *  whose words differ in that shell, per locale. `redesign` is the redesigned
 *  shell's vocabulary (Agents, Machines, Schedules, Chats, Library). */
export interface PluginShellWording {
  redesign?: PluginLocaleBundles
}

export interface PluginI18n {
  /** Merge locale bundles for this plugin (call once at `register`), plus an
   *  optional wording layer per shell. Returns a disposer that drops the
   *  plugin's bundles on unload/reload. */
  register: (bundles: PluginLocaleBundles, wording?: PluginShellWording) => () => void
  /** Module-level translator against the app's active locale (mirrors
   *  `translateNow`). Non-reactive — in React prefer `usePluginI18n`. */
  t: PluginTranslate
  /** Observe locale changes (not initial registration) to rebuild static labels.
   *  Returns an unsubscribe function; also disposed on plugin unload. */
  onLocaleChange: (listener: () => void) => () => void
  /** Add (or extend) a language for the WHOLE app — a language pack. The
   *  strings are a partial of the app catalog (nested, or flat dotted keys)
   *  merged over the bundled catalog for that id, or English for a new
   *  language; a string where English has a function becomes a positional
   *  `{0}`/`{1}` formatter. Attributed to this plugin and dropped on unload;
   *  the returned disposer drops it early. Registering never changes the
   *  active language. */
  registerAppLocale: (id: string, registration: AppLocaleRegistration) => () => void
}

interface PluginBundles {
  base: Map<Locale, PluginMessages>
  redesign: Map<Locale, PluginMessages>
}

const registry = new Map<string, PluginBundles>()

/** Bumps whenever a plugin's bundles change, so React translators re-render on
 *  a registration that lands after first paint. */
const $version = atom(0)

/** Redesign layers merged over their bundles, per plugin and locale; cleared on every change. */
const mergedLayers = new Map<string, PluginMessages>()

const bump = () => {
  mergedLayers.clear()
  $version.set($version.get() + 1)
}

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

function mergeBundles(byLocale: Map<Locale, PluginMessages>, bundles: PluginLocaleBundles | undefined) {
  for (const [locale, messages] of Object.entries(bundles ?? {}) as [Locale, PluginMessages | undefined][]) {
    if (messages) {
      const prev = byLocale.get(locale)
      byLocale.set(locale, prev ? mergeMessages(prev, messages) : messages)
    }
  }
}

export function registerPluginLocales(
  pluginId: string,
  bundles: PluginLocaleBundles,
  wording: PluginShellWording = {}
): () => void {
  const entry = registry.get(pluginId) ?? { base: new Map(), redesign: new Map() }
  registry.set(pluginId, entry)
  mergeBundles(entry.base, bundles)
  mergeBundles(entry.redesign, wording.redesign)
  bump()

  return () => {
    registry.delete(pluginId)
    bump()
  }
}

/** One locale's messages in `shell`'s wording: the bundle, with the shell's layer over it. */
function shellMessages(pluginId: string, locale: Locale, shell: ShellWording): PluginMessages | undefined {
  const entry = registry.get(pluginId)
  const base = entry?.base.get(locale)
  const layer = shell === 'redesign' ? entry?.redesign.get(locale) : undefined

  if (!layer) {
    return base
  }

  const cacheKey = `${pluginId}\u0000${locale}`
  const cached = mergedLayers.get(cacheKey)

  if (cached) {
    return cached
  }

  const merged = base ? mergeMessages(base, layer) : layer
  mergedLayers.set(cacheKey, merged)

  return merged
}

export function translatePlugin(
  pluginId: string,
  locale: Locale,
  key: string,
  args: unknown[],
  shell: ShellWording = currentShellWording()
): string {
  return translateFrom(l => shellMessages(pluginId, l, shell), locale, key, args)
}

/** Build the `ctx.i18n` door for a plugin. `track` records the disposer so the
 *  loader tears bundles down on unload (same lifecycle as `register`/`socket`). */
export function createPluginI18n(pluginId: string, track: (dispose: () => void) => () => void): PluginI18n {
  return {
    register: (bundles, wording) => track(registerPluginLocales(pluginId, bundles, wording)),
    onLocaleChange: listener => track(subscribeRuntimeI18nLocale(listener)),
    registerAppLocale: (id, registration) => track(registerAppLocale(id, registration, `plugin:${pluginId}`)),
    t: (key, ...args) => translatePlugin(pluginId, getRuntimeI18nLocale(), key, args)
  }
}

/** Reactive scoped translator for React UI. Re-renders on a locale switch, a
 *  shell flip or a late bundle registration. Pass your plugin id (your default
 *  export's `id`). */
export function usePluginI18n(pluginId: string): PluginTranslate {
  const { locale } = useI18n()
  const shell = shellWordingFor(usePortalPresentation())
  const version = useStore($version)

  // `version` is the registry's change token and must key the translator's
  // identity: memoized consumers (React.memo, React Compiler output) cache
  // render slices on `t` itself, so a stable `t` over a mutated registry
  // serves stale strings after a late bundle registration.
  return useCallback(
    (key: string, ...args: unknown[]) => translatePlugin(pluginId, locale, key, args, shell),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pluginId, locale, shell, version]
  )
}
