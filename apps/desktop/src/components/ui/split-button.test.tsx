import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { SplitButton } from './split-button'

afterEach(cleanup)

it('localizes the shared alternate-action trigger without changing the primary action', () => {
  const onTrigger = vi.fn()
  render(
    <I18nProvider configClient={null} initialLocale="ar">
      <SplitButton
        actions={[{ id: 'save', label: 'حفظ' }]}
        onTrigger={onTrigger}
        onValueChange={vi.fn()}
        value="save"
      />
    </I18nProvider>
  )

  expect(screen.getByRole('button', { name: 'إجراءات إضافية' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'حفظ' }))
  expect(onTrigger).toHaveBeenCalledWith('save')
})
