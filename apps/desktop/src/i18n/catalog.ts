import { en } from './en'
import type { BundledLocale, Translations } from './types'

/** Every locale id compiled into the app; catalogs are fetched on demand. */
export const BUNDLED_LOCALES = ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru', 'fr', 'de', 'es'] as const

/**
 * The catalogs compiled into the app.
 *
 * Only English belongs to the boot graph. The other eight are ~1.8 MB of
 * dictionaries that a given session never renders, and the bundler used to
 * sweep them into a shared chunk that `index.html` preloaded on every cold
 * start — paying for nine languages to show one. Each one is now a dynamic
 * import, fetched the first time `resolveTranslations()` asks for it (see
 * `./registry`); the type stays total while the runtime map fills in lazily.
 *
 * Runtime-registered languages (plugin packs, backend `.desktop.yaml` packs)
 * are NOT here — resolve through `resolveTranslations()` in `./registry`, which
 * layers them over these.
 */
export const TRANSLATIONS = { en } as Record<BundledLocale, Translations>

export function isBundledLocale(value: unknown): value is BundledLocale {
  return typeof value === 'string' && (BUNDLED_LOCALES as readonly string[]).includes(value)
}

const loaders: Record<Exclude<BundledLocale, 'en'>, () => Promise<Translations>> = {
  ar: () => import('./ar').then(m => m.ar),
  de: () => import('./de').then(m => m.de),
  es: () => import('./es').then(m => m.es),
  fr: () => import('./fr').then(m => m.fr),
  ja: () => import('./ja').then(m => m.ja),
  ru: () => import('./ru').then(m => m.ru),
  zh: () => import('./zh').then(m => m.zh),
  'zh-hant': () => import('./zh-hant').then(m => m.zhHant)
}

const pending = new Map<BundledLocale, Promise<Translations | undefined>>()

/**
 * Fetch and register one bundled catalog. Idempotent and shared: concurrent
 * callers await the same import, and a failed fetch resolves `undefined` so a
 * missing chunk degrades to English instead of throwing on the boot path.
 */
export function loadBundledLocale(id: BundledLocale): Promise<Translations | undefined> {
  const already = TRANSLATIONS[id]

  if (already) {
    return Promise.resolve(already)
  }

  if (id === 'en') {
    return Promise.resolve(en)
  }

  const inFlight = pending.get(id)

  if (inFlight) {
    return inFlight
  }

  const request = loaders[id]()
    .then(translations => {
      TRANSLATIONS[id] = translations

      return translations
    })
    .catch(() => undefined)
    .finally(() => pending.delete(id))

  pending.set(id, request)

  return request
}
