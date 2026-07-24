import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'

import { useComposerPlaceholder } from './use-composer-placeholder'

interface PlaceholderProbeProps {
  disabled?: boolean
  reconnecting?: boolean
  sessionId?: string | null
}

function PlaceholderProbe({ disabled = false, reconnecting = false, sessionId }: PlaceholderProbeProps) {
  const { setLocale } = useI18n()
  const placeholder = useComposerPlaceholder({ disabled, reconnecting, sessionId })

  return (
    <div>
      <p data-testid="placeholder">{placeholder}</p>
      <button onClick={() => void setLocale('zh')} type="button">
        switch
      </button>
    </div>
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useComposerPlaceholder', () => {
  it('re-rolls the resting placeholder when I18nProvider changes locale', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <PlaceholderProbe />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('What are we building?')

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe('我们要构建什么？'))
  })

  it('keeps the starter placeholder when a new session receives its first id', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)

    const { rerender } = render(
      <I18nProvider configClient={null} initialLocale="en">
        <PlaceholderProbe sessionId={null} />
      </I18nProvider>
    )

    rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <PlaceholderProbe sessionId="session-1" />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('What are we building?')
  })

  it('keeps transport-state output localized while reconnecting', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <PlaceholderProbe disabled reconnecting />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('Reconnecting to Hermes…')

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe('正在重新连接 Hermes…'))
  })
})
