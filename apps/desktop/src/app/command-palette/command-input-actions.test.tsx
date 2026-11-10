import { fireEvent, render, screen } from '@testing-library/react'
import { Dialog as DialogPrimitive } from 'radix-ui'
import { describe, expect, it, vi } from 'vitest'

import { CommandInputActions } from './command-input-actions'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: { common: { close: 'Close' } } }) }))

describe('command input actions', () => {
  it('preserves inline actions and dismisses the owning dialog', () => {
    render(
      <DialogPrimitive.Root defaultOpen>
        <DialogPrimitive.Content aria-describedby={undefined}>
          <DialogPrimitive.Title>Command palette</DialogPrimitive.Title>
          <CommandInputActions extra={<button>Existing inline action</button>} />
        </DialogPrimitive.Content>
      </DialogPrimitive.Root>
    )

    expect(screen.getByRole('button', { name: 'Existing inline action' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
