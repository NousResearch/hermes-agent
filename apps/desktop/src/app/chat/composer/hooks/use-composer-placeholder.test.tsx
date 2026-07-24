import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type I18nConfigClient, I18nProvider, useI18n } from '@/i18n'

import { useComposerPlaceholder } from './use-composer-placeholder'

type SessionId = null | string | undefined

interface RenderRecord {
  locale: string
  placeholder: string
}

const mocks = vi.hoisted(() => ({ resetBrowseState: vi.fn() }))

vi.mock('@/store/composer-input-history', () => ({ resetBrowseState: mocks.resetBrowseState }))

function PlaceholderProbe({
  disabled = false,
  onRender,
  reconnecting = false,
  sessionId
}: {
  disabled?: boolean
  onRender?: (record: RenderRecord) => void
  reconnecting?: boolean
  sessionId: SessionId
}) {
  const { isLoadingConfig, locale, setLocale } = useI18n()
  const placeholder = useComposerPlaceholder({ disabled, reconnecting, sessionId })

  useEffect(() => {
    onRender?.({ locale, placeholder })
  }, [locale, onRender, placeholder])

  return (
    <>
      <p data-testid="locale">{locale}</p>
      <p data-testid="loading">{String(isLoadingConfig)}</p>
      <p data-testid="placeholder">{placeholder}</p>
      <button onClick={() => void setLocale('es').catch(() => undefined)} type="button">
        switch to Spanish
      </button>
    </>
  )
}

function SessionAndLocaleProbe() {
  const [sessionId, setSessionId] = useState<SessionId>(null)
  const { setLocale } = useI18n()
  const placeholder = useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId })

  return (
    <>
      <p data-testid="placeholder">{placeholder}</p>
      <button
        onClick={() => {
          setSessionId('session-a')
          void setLocale('es')
        }}
        type="button"
      >
        persist and switch
      </button>
    </>
  )
}

describe('useComposerPlaceholder', () => {
  beforeEach(() => {
    mocks.resetBrowseState.mockClear()
    vi.spyOn(Math, 'random').mockReturnValue(0.35)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('uses the selected starter and follow-up positions', () => {
    const starter = render(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId={null} />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe("What's on your mind?")
    starter.unmount()

    render(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId="session-a" />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('Refine the request')
  })

  it('switches an existing conversation to the corresponding Spanish text without resetting history', async () => {
    const renders: RenderRecord[] = []
    vi.spyOn(Math, 'random').mockReset().mockReturnValueOnce(0.35).mockReturnValue(0.5)

    render(
      <I18nProvider configClient={null}>
        <PlaceholderProbe onRender={record => renders.push(record)} sessionId="session-a" />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('Refine the request')

    fireEvent.click(screen.getByRole('button', { name: 'switch to Spanish' }))

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe('Refinar la solicitud'))

    expect(renders).not.toContainEqual({ locale: 'es', placeholder: 'Refine the request' })
    expect(mocks.resetBrowseState).not.toHaveBeenCalled()
  })

  it('keeps a starter when persistence and a locale switch happen together', async () => {
    vi.spyOn(Math, 'random').mockReset().mockReturnValueOnce(0.35).mockReturnValue(0.5)
    render(
      <I18nProvider configClient={null}>
        <SessionAndLocaleProbe />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe("What's on your mind?")
    fireEvent.click(screen.getByRole('button', { name: 'persist and switch' }))

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe('¿Qué tienes en mente?'))
    expect(mocks.resetBrowseState).not.toHaveBeenCalled()
  })

  it('treats null and undefined as the same new-session state', () => {
    vi.spyOn(Math, 'random').mockReset().mockReturnValueOnce(0.35).mockReturnValue(0.5)

    const view = render(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId={undefined} />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe("What's on your mind?")
    view.rerender(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId={null} />
      </I18nProvider>
    )
    view.rerender(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId={undefined} />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe("What's on your mind?")
    expect(mocks.resetBrowseState).not.toHaveBeenCalled()
  })

  it('resets only the previous conversation and selects a new follow-up on a real session change', async () => {
    vi.restoreAllMocks()
    const random = vi.spyOn(Math, 'random').mockReturnValueOnce(0.35).mockReturnValueOnce(0.5)

    const view = render(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId="session-a" />
      </I18nProvider>
    )

    view.rerender(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId="session-b" />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe("What's next?"))
    expect(mocks.resetBrowseState).toHaveBeenCalledTimes(1)
    expect(mocks.resetBrowseState).toHaveBeenCalledWith('session-a')
    expect(random).toHaveBeenCalled()
  })

  it('resets the previous conversation and selects a starter when returning to a new session', async () => {
    vi.restoreAllMocks()
    const random = vi.spyOn(Math, 'random').mockReturnValueOnce(0.35).mockReturnValueOnce(0.5)

    const view = render(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId="session-a" />
      </I18nProvider>
    )

    view.rerender(
      <I18nProvider configClient={null}>
        <PlaceholderProbe sessionId={null} />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe('Describe what you need'))
    expect(mocks.resetBrowseState).toHaveBeenCalledTimes(1)
    expect(mocks.resetBrowseState).toHaveBeenCalledWith('session-a')
    expect(random).toHaveBeenCalled()
  })

  it('keeps the same selection through a failed locale save and rollback', async () => {
    vi.spyOn(Math, 'random').mockReset().mockReturnValueOnce(0.35).mockReturnValue(0.5)

    let rejectSave: (reason?: unknown) => void = () => {}

    const pendingSave = new Promise<{ ok: boolean }>((_resolve, reject) => {
      rejectSave = reject
    })

    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'en' } }),
      saveConfig: vi.fn().mockReturnValue(pendingSave)
    }

    const renders: RenderRecord[] = []

    render(
      <I18nProvider configClient={configClient}>
        <PlaceholderProbe onRender={record => renders.push(record)} sessionId="session-a" />
      </I18nProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('false')
      expect(configClient.getConfig).toHaveBeenCalledTimes(1)
    })
    fireEvent.click(screen.getByRole('button', { name: 'switch to Spanish' }))

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe('Refinar la solicitud'))
    await waitFor(() => expect(configClient.saveConfig).toHaveBeenCalledTimes(1))
    await act(async () => {
      rejectSave(new Error('save failed'))
    })

    await waitFor(() => expect(screen.getByTestId('placeholder').textContent).toBe('Refine the request'))

    expect(renders).not.toContainEqual({ locale: 'es', placeholder: 'Refine the request' })
    expect(mocks.resetBrowseState).not.toHaveBeenCalled()
  })

  it('uses localized connection-state copy', () => {
    const starting = render(
      <I18nProvider configClient={null} initialLocale="es">
        <PlaceholderProbe disabled reconnecting={false} sessionId="session-a" />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('Iniciando Hermes...')
    starting.unmount()

    render(
      <I18nProvider configClient={null} initialLocale="es">
        <PlaceholderProbe disabled reconnecting sessionId="session-a" />
      </I18nProvider>
    )

    expect(screen.getByTestId('placeholder').textContent).toBe('Reconectando con Hermes…')
  })
})
