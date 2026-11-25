// @vitest-environment jsdom
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { BUNDLED_LOCALES, TRANSLATIONS } from '@/i18n'
import { resetAppLocaleRegistry } from '@/i18n/registry'
import { setRuntimeI18nLocale } from '@/i18n/runtime'
import { providePortalPresentation } from '@/store/shell-presentation'

import { bootReadyCopy, bootStepCopy } from './boot-copy'

interface HostGlobals {
  __HERMES_AUTH_REQUIRED__?: boolean
  __HERMES_UI_SURFACE__?: string
  hermesDesktop?: unknown
}

const win = window as unknown as HostGlobals

function reset() {
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  delete win.__HERMES_AUTH_REQUIRED__
  delete win.__HERMES_UI_SURFACE__
  delete win.hermesDesktop
  providePortalPresentation(atom(false))
  resetAppLocaleRegistry()
  setRuntimeI18nLocale('en')
}

afterEach(reset)

describe('boot copy by host', () => {
  it('drops the Desktop wording in a browser tab (the Webapp)', () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'

    expect(bootStepCopy('startingHermesDesktop')).toBe('Starting Hermes…')
    expect(bootStepCopy('startingDesktopConnection')).toBe('Starting connection')
    expect(bootReadyCopy()).toBe('Hermes is ready')
  })

  it('drops it before the bridge writes the marker, from the serving globals', () => {
    win.__HERMES_UI_SURFACE__ = 'webapp'

    expect(bootStepCopy('startingHermesDesktop')).toBe('Starting Hermes…')
    expect(bootReadyCopy()).toBe('Hermes is ready')
  })

  it('keeps the shipped Desktop wording in native Electron', () => {
    expect(bootStepCopy('startingHermesDesktop')).toBe('Starting Hermes Desktop…')
    expect(bootStepCopy('startingDesktopConnection')).toBe('Starting desktop connection')
    expect(bootReadyCopy()).toBe('Hermes Desktop is ready')
  })

  it('keeps the Desktop wording under the Electron preload despite a stale Webapp global', () => {
    win.hermesDesktop = {}
    win.__HERMES_UI_SURFACE__ = 'webapp'

    expect(bootStepCopy('startingHermesDesktop')).toBe('Starting Hermes Desktop…')
    expect(bootReadyCopy()).toBe('Hermes Desktop is ready')
  })

  it('follows the host marker written after module import', () => {
    expect(bootStepCopy('startingHermesDesktop')).toBe('Starting Hermes Desktop…')
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    expect(bootStepCopy('startingHermesDesktop')).toBe('Starting Hermes…')
  })
})

describe('boot.webapp locale coverage', () => {
  // Every bundled catalog carries a Webapp boot copy that actually drops the Desktop wording, so
  // no locale silently falls back to its classic string on the browser host.
  it.each(BUNDLED_LOCALES)('%s localizes the Webapp boot copy', locale => {
    const boot = TRANSLATIONS[locale].boot

    expect(boot.webapp.ready).not.toBe(boot.ready)
    expect(boot.webapp.steps.startingHermes).not.toBe(boot.steps.startingHermesDesktop)
    expect(boot.webapp.steps.startingConnection).not.toBe(boot.steps.startingDesktopConnection)
  })
})
