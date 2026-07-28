// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
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
})
