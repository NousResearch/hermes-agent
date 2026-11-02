import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Dialog, DialogContent, DialogDescription, DialogTitle } from './dialog'

afterEach(cleanup)

describe('Dialog prose direction', () => {
  it('keeps an icon in its original flex position while marking only title text', () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle icon={() => <svg data-testid="title-icon" />}>نام project</DialogTitle>
        </DialogContent>
      </Dialog>
    )

    const title = screen.getByRole('heading', { name: /project/ })
    expect(title.firstElementChild).toBe(screen.getByTestId('title-icon'))
    expect(title.lastElementChild?.hasAttribute('data-bidi-prose')).toBe(true)
    expect(title.lastElementChild?.classList.contains('flex-1')).toBe(true)
    expect(title.hasAttribute('dir')).toBe(false)
  })

  it('marks title and description for per-block bidi without changing dialog chrome', () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>آیا project را حذف کنیم؟</DialogTitle>
          <DialogDescription>نام project در فهرست باقی نمی‌ماند.</DialogDescription>
        </DialogContent>
      </Dialog>
    )

    expect(screen.getByRole('heading', { name: /project/ }).hasAttribute('data-bidi-prose')).toBe(true)
    expect(screen.getByText(/نام project/).hasAttribute('data-bidi-prose')).toBe(true)
    expect(screen.getByRole('dialog').hasAttribute('data-bidi-prose')).toBe(false)
  })
})

describe('DialogContent close button', () => {
  it('closes the dialog when clicked', () => {
    const onOpenChange = vi.fn()
    render(
      <Dialog onOpenChange={onOpenChange} open>
        <DialogContent>
          <DialogTitle>Test dialog</DialogTitle>
        </DialogContent>
      </Dialog>
    )

    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })
})
