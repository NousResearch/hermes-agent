import type * as Runtime from '../src/i18n/runtime'

import { browserMessages } from './messages-locales'
import type { BrowserMessages } from './messages-locales'

let runtime: typeof Runtime | undefined

// The locale catalog imports renderer modules; load it only after installing the host.
export async function initializeBrowserMessages() {
  runtime = await import('../src/i18n/runtime')
}

export function browserMessage(key: keyof BrowserMessages): string {
  if (!runtime) {return browserMessages.en[key]}

  return runtime.translateFrom(locale => browserMessages[locale as keyof typeof browserMessages], runtime.getRuntimeI18nLocale(), key, [])
}

export function sharedMessage(key: string): string {
  return runtime?.translateNow(key) || key
}
