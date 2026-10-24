import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { NotificationDetail } from '@/components/notifications'
import { I18nProvider } from '@/i18n'

const LONG_DETAIL =
  'STT provider not found. Checked: faster-whisper (not installed), GROQ_API_KEY (unset), OPENAI_API_KEY (unset). ' +
  'Configure one of these providers and restart voice mode. See https://example.com/docs/stt-providers. '.repeat(6)

let clipboard: { writeText: ReturnType<typeof vi.fn> }

function renderDetail(detail = LONG_DETAIL) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <NotificationDetail detail={detail} />
    </I18nProvider>
  )
}

describe('NotificationDetail', () => {
  beforeEach(() => {
    clipboard = { writeText: vi.fn().mockResolvedValue(undefined) }
    vi.stubGlobal('navigator', { ...navigator, clipboard })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    cleanup()
  })

  it('scrolls long detail inside its bounded block instead of spilling over the copy button', () => {
    const { container } = renderDetail()
    const pre = container.querySelector('pre')!

    // #69478: a capped height with visible overflow let long text paint over
    // the button below. jsdom doesn't compute Tailwind, so assert the utilities.
    expect(pre.className).toMatch(/\bmax-h-\S+/)
    expect(pre.className).toContain('overflow-y-auto')
    expect(pre.textContent).toBe(LONG_DETAIL)
  })

  it('keeps the copy button outside the scroll area and copies the full detail', async () => {
    const { container } = renderDetail()
    const pre = container.querySelector('pre')!
    const button = screen.getByRole('button', { name: /Copy detail/i })

    expect(pre.contains(button)).toBe(false)
    expect(pre.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    fireEvent.click(button)
    await vi.waitFor(() => expect(clipboard.writeText).toHaveBeenCalledWith(LONG_DETAIL))
  })
})
