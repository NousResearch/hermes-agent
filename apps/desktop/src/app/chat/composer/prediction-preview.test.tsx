import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { BUNDLED_LOCALES, TRANSLATIONS } from '@/i18n/catalog'
import type { BundledLocale } from '@/i18n/types'

import { ComposerPredictionPreview } from './prediction-preview'

const active = vi.hoisted(() => ({ locale: 'en' as BundledLocale }))
vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: TRANSLATIONS[active.locale] })
}))

afterEach(cleanup)

describe('composer prediction preview', () => {
  it.each(BUNDLED_LOCALES)('uses the host %s catalog for hints and announcements', locale => {
    active.locale = locale
    const suggestions = ['First candidate', '完整的下一条消息\nWith a second line']

    const { container, rerender } = render(
      <ComposerPredictionPreview prediction={{ suggestions, index: 0 }} stacked={false} />
    )

    const copy = TRANSLATIONS[locale].composer
    expect(copy.predictionHint).toBeTruthy()
    expect(container.querySelector('[data-slot="composer-prediction-hint"]')?.textContent).toContain(
      copy.predictionHint
    )
    expect(screen.getByRole('status').textContent).toContain(copy.predictionAnnouncement(1, 2, suggestions[0]))

    rerender(<ComposerPredictionPreview prediction={{ suggestions, index: 1 }} stacked />)
    expect(screen.getByRole('status').textContent).toContain(copy.predictionAnnouncement(2, 2, suggestions[1]))
    expect(screen.getByRole('status').getAttribute('aria-atomic')).toBe('true')
    expect(container.querySelector('[data-slot="composer-prediction-text"]')?.textContent).toBe(suggestions[1])
    expect(container.querySelector('[contenteditable], input, textarea, button, [tabindex]')).toBeNull()
  })
})
