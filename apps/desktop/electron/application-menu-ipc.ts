import { ipcMain } from 'electron'

import { type ApplicationMenuLocale, isApplicationMenuLocale } from './application-menu'

export const APPLICATION_MENU_LOCALE_CHANNEL = 'hermes:application-menu:set-locale'

export function registerApplicationMenuIpc(setLocale: (locale: ApplicationMenuLocale) => void): void {
  ipcMain.handle(APPLICATION_MENU_LOCALE_CHANNEL, (_event, locale: unknown) => {
    if (!isApplicationMenuLocale(locale)) {
      throw new TypeError('Invalid application menu locale')
    }

    setLocale(locale)

    return { locale, ok: true }
  })
}
