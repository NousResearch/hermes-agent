import { ipcMain } from 'electron'

import { type ApplicationMenuLocale, isApplicationMenuLocale } from './application-menu'

export const APPLICATION_MENU_LOCALE_CHANNEL = 'hermes:application-menu:set-locale'

export function registerApplicationMenuIpc(setLocale: (locale: ApplicationMenuLocale) => void): void {
  ipcMain.handle(APPLICATION_MENU_LOCALE_CHANNEL, (_event, requested: unknown) => {
    if (typeof requested !== 'string') {
      throw new TypeError('Invalid application menu locale')
    }

    // Plugins and the backend can register display languages at runtime that
    // have no menu translation; show English then instead of keeping the
    // previous language's menu.
    const locale: ApplicationMenuLocale = isApplicationMenuLocale(requested) ? requested : 'en'

    setLocale(locale)

    return { locale, ok: true }
  })
}
