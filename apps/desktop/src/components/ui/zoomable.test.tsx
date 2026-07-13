import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ScrollGate } from '@/components/assistant-ui/embeds/scroll-gate'
import { I18nProvider } from '@/i18n'

import { Zoomable } from './zoomable'

describe('Zoomable Arabic copy', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('localizes the full-view dialog and every toolbar action', () => {
    render(
      <I18nProvider configClient={null} initialLocale="ar">
        <Zoomable onCopy={vi.fn()}>
          <span>مخطط تجريبي</span>
        </Zoomable>
      </I18nProvider>
    )

    fireEvent.click(screen.getByTitle('فتح العرض الكامل'))

    expect(screen.getByRole('dialog', { name: 'فتح العرض الكامل' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'تصغير' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'إعادة الضبط' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'تكبير' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'نسخ' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'إغلاق' })).toBeTruthy()
  })

  it('localizes the map zoom gate hint', () => {
    render(
      <I18nProvider configClient={null} initialLocale="ar">
        <ScrollGate />
      </I18nProvider>
    )

    expect(screen.getByText('اضغط على مفتاح الأوامر للتكبير')).toBeTruthy()
  })
})
