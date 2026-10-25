import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord } from '@/hermes'
import type { I18nConfigClient} from '@/i18n';
import { I18nProvider, useI18n } from '@/i18n'
import { TRANSLATIONS } from '@/i18n/catalog'
import { registerAppLocale, resetAppLocaleRegistry } from '@/i18n/registry'
import { setRuntimeI18nLocale } from '@/i18n/runtime'
import type { Locale } from '@/i18n/types'

import { useComposerPlaceholder } from './use-composer-placeholder'

interface ProbeProps {
  disabled?: boolean
  reconnecting?: boolean
  sessionId?: null | string
}

const api: {
  placeholder?: string
  setLocale?: (next: Locale) => Promise<void>
} = {}

function Probe({ disabled = false, reconnecting = false, sessionId = null }: ProbeProps) {
  api.setLocale = useI18n().setLocale
  api.placeholder = useComposerPlaceholder({ disabled, reconnecting, sessionId })

  return null
}

function renderProbe(props: ProbeProps = {}) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <Probe {...props} />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  api.placeholder = undefined
  api.setLocale = undefined
  resetAppLocaleRegistry()
  setRuntimeI18nLocale('en')
  vi.restoreAllMocks()
})

const koreanComposer = {
  newSessionPlaceholders: ['시작 안내 0', '시작 안내 1'],
  followUpPlaceholders: ['후속 안내 0', '후속 안내 1']
}

/** A config that saved `ko` before any pack could render it — the parked pick. */
const koConfigClient: I18nConfigClient = {
  getConfig: () => Promise.resolve({ display: { language: 'ko' } } as HermesConfigRecord),
  saveConfig: () => Promise.resolve({ ok: true })
}

describe('useComposerPlaceholder', () => {
  it('re-reads the resting starter from the active catalogue once the locale lands', async () => {
    renderProbe()

    const enPool = TRANSLATIONS.en.composer.newSessionPlaceholders
    expect(enPool).toContain(api.placeholder)

    const pickedSlot = enPool.indexOf(api.placeholder!)

    await act(async () => {
      await api.setLocale!('es')
    })

    const esPool = TRANSLATIONS.es.composer.newSessionPlaceholders
    const localized = api.placeholder!
    expect(esPool).toContain(localized)
    // The slot survives the catalogue swap: the same starter position renders
    // in the new language rather than re-rolling under the user.
    expect(esPool.indexOf(localized)).toBe(pickedSlot)
  })

  it('keeps the picked starter across the null→id persist and re-rolls it only on a different conversation', () => {
    const view = renderProbe({ sessionId: null })

    const starter = api.placeholder!
    expect(TRANSLATIONS.en.composer.newSessionPlaceholders).toContain(starter)

    view.rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <Probe sessionId="session-1" />
      </I18nProvider>
    )
    expect(api.placeholder).toBe(starter)

    view.rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <Probe sessionId="session-2" />
      </I18nProvider>
    )
    expect(TRANSLATIONS.en.composer.followUpPlaceholders).toContain(api.placeholder)
  })

  it('follows a saved locale once its plugin pack registers, holding the starter slot', async () => {
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.75)
    render(
      <I18nProvider configClient={koConfigClient} initialLocale="en">
        <Probe />
      </I18nProvider>
    )

    // The saved `ko` is parked, not renderable: English shows on first paint.
    const enPool = TRANSLATIONS.en.composer.newSessionPlaceholders
    expect(enPool).toContain(api.placeholder)
    const pickedSlot = enPool.indexOf(api.placeholder!)

    await act(async () => {
      registerAppLocale('ko', { translations: { composer: koreanComposer } }, 'plugin:hermes-lang-ko')
    })

    // The parked pick promotes, and the same slot renders in Korean — the
    // starter is translated, not re-rolled under the user.
    const koPool = koreanComposer.newSessionPlaceholders
    expect(api.placeholder).toBe(koPool[pickedSlot % koPool.length])
    expect(random).toHaveBeenCalledTimes(1)

    await act(async () => {
      resetAppLocaleRegistry()
    })
    // Unloading the pack falls back to English, still under the held slot.
    expect(api.placeholder).toBe(enPool[pickedSlot])
  })

  it('wraps the held follow-up slot when a same-language pack ships a shorter pool', async () => {
    const random = vi.spyOn(Math, 'random').mockReturnValue(0.75)
    render(
      <I18nProvider configClient={koConfigClient} initialLocale="en">
        <Probe sessionId="existing" />
      </I18nProvider>
    )

    const enPool = TRANSLATIONS.en.composer.followUpPlaceholders
    expect(enPool).toContain(api.placeholder)
    const pickedSlot = enPool.indexOf(api.placeholder!)

    await act(async () => {
      registerAppLocale('ko', { translations: { composer: koreanComposer } }, 'plugin:hermes-lang-ko')
    })

    // The pack's pool is shorter than the held slot: resolve wraps instead of
    // dropping to undefined.
    const koPool = koreanComposer.followUpPlaceholders
    expect(koPool.length).toBeLessThan(enPool.length)
    expect(api.placeholder).toBe(koPool[pickedSlot % koPool.length])
    expect(random).toHaveBeenCalledTimes(1)
  })
})
