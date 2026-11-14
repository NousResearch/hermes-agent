import { act, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it } from 'vitest'

import { providePortalPresentation } from '@/store/shell-presentation'

import { TRANSLATIONS } from '../catalog'
import { I18nProvider, useI18n } from '../context'
import { registerAppLocale, resetAppLocaleRegistry, resolveTranslations } from '../registry'
import { setRuntimeI18nLocale, translateNow } from '../runtime'

const portal = atom(false)

afterEach(() => {
  portal.set(false)
  providePortalPresentation(atom(false))
  resetAppLocaleRegistry()
  setRuntimeI18nLocale('en')
})

describe('per-shell wording', () => {
  it('keeps classic on the base catalog and swaps each redesign term', () => {
    const classic = resolveTranslations('en', 'classic')
    const redesign = resolveTranslations('en', 'redesign')

    expect(classic).toBe(TRANSLATIONS.en)
    expect([classic.sidebar.sessions, redesign.sidebar.sessions]).toEqual(['Sessions', 'Chats'])
    expect([classic.sidebar.nav.artifacts, redesign.sidebar.nav.artifacts]).toEqual(['Artifacts', 'Library'])
    expect([classic.profiles.title, redesign.profiles.title]).toEqual(['Profiles', 'Agents'])
    expect([classic.cron.title, redesign.cron.title]).toEqual(['Scheduled jobs', 'Routines'])
    expect([classic.settings.nav.gateway, redesign.settings.nav.gateway]).toEqual(['Gateways', 'Machines'])
  })

  // Classic is main's copy verbatim; the redesign's shared-account warning lives in its overlay.
  it("keeps main's connector and blueprint copy in classic", () => {
    const classic = resolveTranslations('en', 'classic').connectorsPage
    const redesign = resolveTranslations('en', 'redesign').connectorsPage

    expect(classic.dialog.disconnectBody).toBe('Hermes stops acting as this account. You can connect again at any time.')
    expect(classic.accounts.removeBody).toBe('Hermes stops acting as this account. Your other accounts stay connected.')
    expect(redesign.dialog.disconnectBody).toMatch(/for all agents using the same Nous account/)
    expect(redesign.accounts.removeBody).toMatch(/for all agents using the same Nous account/)
    expect(TRANSLATIONS.en.cron.blueprints.scheduleIt).toBe('Schedule it')
    expect(TRANSLATIONS.en.cron.blueprints.scheduling).toBe('Scheduling...')
  })

  it('overrides function-valued messages too', () => {
    expect(resolveTranslations('en', 'classic').profiles.count(2)).toBe('2 profiles')
    expect(resolveTranslations('en', 'redesign').profiles.count(2)).toBe('2 agents')
  })

  it('gives a translated locale its own wording and an untranslated key the English redesign', () => {
    expect(TRANSLATIONS.ja.sidebar.sessions).toBe('セッション')
    expect(resolveTranslations('ja', 'redesign').sidebar.sessions).toBe('チャット')

    // A pack-only language starts from English, so it reads the English redesign.
    registerAppLocale('pl', { translations: { 'common.save': 'Zapisz' } })
    expect(resolveTranslations('pl', 'redesign').sidebar.sessions).toBe('Chats')
    expect(resolveTranslations('pl', 'classic').sidebar.sessions).toBe('Sessions')
  })

  it('lets a registered pack win over both shells', () => {
    registerAppLocale('en', { translations: { 'sidebar.sessions': 'Pack wording' } }, 'backend')

    expect(resolveTranslations('en', 'classic').sidebar.sessions).toBe('Pack wording')
    expect(resolveTranslations('en', 'redesign').sidebar.sessions).toBe('Pack wording')
    expect(resolveTranslations('en', 'redesign').sidebar.nav.artifacts).toBe('Library')
  })

  it('re-resolves the runtime translator when the shell flips', () => {
    providePortalPresentation(portal)

    expect(translateNow('sidebar.sessions')).toBe('Sessions')
    portal.set(true)
    expect(translateNow('sidebar.sessions')).toBe('Chats')
    expect(translateNow('profiles.count', 3)).toBe('3 agents')
    portal.set(false)
    expect(translateNow('sidebar.sessions')).toBe('Sessions')
  })

  it('re-renders hook consumers with the other wording without a reload', () => {
    providePortalPresentation(portal)

    function Label() {
      return <span>{useI18n().t.sidebar.nav.artifacts}</span>
    }

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <Label />
      </I18nProvider>
    )

    expect(screen.getByText('Artifacts')).toBeTruthy()
    act(() => portal.set(true))
    expect(screen.getByText('Library')).toBeTruthy()
    act(() => portal.set(false))
    expect(screen.getByText('Artifacts')).toBeTruthy()
  })
})
