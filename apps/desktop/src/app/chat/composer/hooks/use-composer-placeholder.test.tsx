import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'
import { TRANSLATIONS } from '@/i18n/catalog'
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
  setRuntimeI18nLocale('en')
})

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
})
