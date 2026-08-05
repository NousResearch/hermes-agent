// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ConfirmDialog } from './confirm-dialog'

afterEach(cleanup)

describe('desktop ConfirmDialog typed confirmation', () => {
  it('does not run until the exact phrase is entered', () => {
    const onConfirm = vi.fn()
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ConfirmDialog
          onClose={() => undefined}
          onConfirm={onConfirm}
          open
          title="Update Hermes"
          typedConfirmation="UPDATE"
        />
      </I18nProvider>
    )

    const confirm = screen.getByRole('button', { name: 'Confirm' })
    const input = screen.getByLabelText(/Type UPDATE to confirm/i)
    expect((confirm as HTMLButtonElement).disabled).toBe(true)

    fireEvent.change(input, { target: { value: 'UPDATE' } })
    expect((confirm as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(confirm)
    expect(onConfirm).toHaveBeenCalledOnce()
  })

  it('blocks repeat submission while a dismissing action is pending', async () => {
    let finish: (() => void) | undefined

    const onConfirm = vi.fn(
      () =>
        new Promise<void>(resolve => {
          finish = resolve
        })
    )

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <ConfirmDialog
          dismissOnConfirm
          onClose={() => undefined}
          onConfirm={onConfirm}
          open
          title="Restart Hermes"
        />
      </I18nProvider>
    )

    const confirm = screen.getByRole('button', { name: 'Confirm' }) as HTMLButtonElement
    fireEvent.click(confirm)
    fireEvent.click(confirm)

    expect(confirm.disabled).toBe(true)
    expect(onConfirm).toHaveBeenCalledOnce()
    await act(async () => {
      finish?.()
    })
  })
})
