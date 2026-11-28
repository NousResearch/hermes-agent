import { type AppLocaleRegistration, type LanguageOption, languageOptions, registerAppLocale } from '@/i18n'
import { getAppLocale, subscribeAppLocale } from '@/i18n/app-locale'
import type { Locale } from '@/i18n/types'

/** THE language-pack surface: add a whole UI language (or extend one) at
 *  runtime. A registration is a PARTIAL catalog — nested like `en.ts` or flat
 *  dotted keys (`{ "common.save": "Zapisz" }`) — merged over the bundled
 *  catalog for that id, or English for a new language; a plain string where
 *  English has a function becomes a positional `{0}`/`{1}` formatter. The
 *  switcher lists it by its endonym at once; `display.language` stays
 *  whatever the user chose (registering is not selecting).
 *
 *  Prefer `ctx.i18n.registerAppLocale` inside `register(ctx)`: same call,
 *  attributed to the plugin and disposed on unload. This host form is for
 *  code with no ctx in reach; hand its disposer to `ctx.onDispose`. */
export const i18nHost = {
  /** Register (or replace) a language. Returns the disposer that drops it. */
  registerAppLocale: (id: string, registration: AppLocaleRegistration): (() => void) =>
    registerAppLocale(id, registration, 'app'),

  /** Every selectable language right now — bundled ∪ plugin-registered ∪
   *  the backend's `i18n.languages` — endonym, direction and source. */
  languageOptions: (): LanguageOption[] => languageOptions(),

  /** The locale the app is rendering in right now. */
  locale: (): Locale => getAppLocale(),

  /** Follow locale switches — the user picking a language, a profile's saved
   *  language taking effect, a pack arriving late. Emits the current value
   *  immediately, then after every change; the same contract as
   *  `host.settings.subscribe`. Retire it with `ctx.onDispose`. */
  subscribeLocale: (listener: (locale: Locale) => void): (() => void) => subscribeAppLocale(listener)
}
