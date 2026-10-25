import { afterEach, expect, it } from 'vitest'

import { setRuntimeI18nLocale } from '../src/i18n/runtime'

import { browserMessage, initializeBrowserMessages, sharedMessage } from './messages'
import { browserMessages } from './messages-locales'

afterEach(() => setRuntimeI18nLocale('en'))

it('uses the current renderer language for browser failures after language changes', async () => {
  await initializeBrowserMessages()
  setRuntimeI18nLocale('de')
  expect(browserMessage('attachmentExpired')).toBe(browserMessages.de.attachmentExpired)
  expect(sharedMessage('preview.unavailable')).not.toBe('preview.unavailable')
  setRuntimeI18nLocale('ja')
  expect(browserMessage('terminalClosed')).toBe(browserMessages.ja.terminalClosed)
})

it('falls back to English for backend language packs without browser copy', async () => {
  await initializeBrowserMessages()
  setRuntimeI18nLocale('pl')
  expect(browserMessage('desktopOnly')).toBe(browserMessages.en.desktopOnly)
})

it('resolves shared failure messages in every bundled language', async () => {
  await initializeBrowserMessages()

  for (const locale of Object.keys(browserMessages) as Array<keyof typeof browserMessages>) {
    setRuntimeI18nLocale(locale)

    for (const key of ['settings.gateway.signOutFailed', 'fileMenu.downloadFailed', 'desktop.noClipboardImage', 'desktop.imageDownloadFailed', 'boot.ready', 'preview.unavailable']) {
      expect(sharedMessage(key)).not.toBe(key)
    }
  }
})
