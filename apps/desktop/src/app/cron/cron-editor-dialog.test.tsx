import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { CronEditorDialog } from './index'

vi.mock('@/lib/model-options', () => ({
  requestModelOptions: vi.fn().mockResolvedValue({ providers: [] })
}))

afterEach(cleanup)

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.setPointerCapture ??= () => undefined
  Element.prototype.releasePointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

function renderCronEditor(onClose = vi.fn()) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <QueryClientProvider client={queryClient}>
        <CronEditorDialog editor={{ mode: 'create' }} onClose={onClose} onSave={async () => {}} />
      </QueryClientProvider>
    </I18nProvider>
  )

  return onClose
}

describe('CronEditorDialog', () => {
  it('does not discard the editor when an outside pointer event dismisses an open nested select', async () => {
    const onClose = renderCronEditor()

    await act(async () => {
      fireEvent.pointerDown(screen.getByRole('combobox', { name: 'Frequency' }), {
        button: 0,
        ctrlKey: false,
        pointerType: 'mouse'
      })
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    })

    // This confirms the real SelectContent lifecycle, including its Radix portal.
    expect(await screen.findByRole('listbox')).toBeTruthy()

    // Radix installs its document-level outside-interaction listener on the
    // next task after the Select portal opens.
    await act(async () => {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    })

    // The first click dismisses the portal-backed Select. Once its layer has
    // unmounted, the following click must not dismiss the parent editor.
    await act(async () => {
      fireEvent.pointerDown(window.document.body, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      fireEvent.pointerUp(window.document.body, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      fireEvent.click(window.document.body)
    })
    await waitFor(() => expect(screen.queryByRole('listbox')).toBeNull())

    await act(async () => {
      fireEvent.pointerDown(window.document.body, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      fireEvent.pointerUp(window.document.body, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      fireEvent.click(window.document.body)
    })

    expect(onClose).not.toHaveBeenCalled()
    expect(screen.getByRole('dialog', { name: 'New cron job' })).toBeTruthy()
  })

  it('still closes through Cancel', () => {
    const onClose = renderCronEditor()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(onClose).toHaveBeenCalledOnce()
  })
})
