import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $notifications, clearNotifications } from '@/store/notifications'

import { CopyButton, copyTextWithFeedback } from './copy-button'

describe('CopyButton i18n', () => {
  afterEach(() => {
    cleanup()
    clearNotifications()
    vi.restoreAllMocks()
  })

  it('uses localized default labels and copied feedback', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText }
    })

    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <CopyButton text="hello" />
      </I18nProvider>
    )

    const button = screen.getByRole('button', { name: '复制' })

    expect(button.textContent).toContain('复制')
    fireEvent.click(button)

    await waitFor(() => expect(writeText).toHaveBeenCalledWith('hello'))
    await waitFor(() => expect(screen.getByRole('button', { name: '已复制' })).toBeTruthy())
    expect(screen.getByRole('button', { name: '已复制' }).textContent).toContain('已复制')
  })

  it('posts a success notification only when opted in', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText }
    })

    await copyTextWithFeedback('hello', {
      haptic: false,
      notifySuccess: true,
      successMessage: 'Copied session ID',
      successTitle: 'Copy ID'
    })

    expect(writeText).toHaveBeenCalledWith('hello')
    expect($notifications.get()[0]).toMatchObject({
      kind: 'success',
      message: 'Copied session ID',
      title: 'Copy ID'
    })
  })

  it('toasts the failure and rethrows when the clipboard refuses the write', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: vi.fn().mockRejectedValue(new Error('Write permission denied')) }
    })

    await expect(copyTextWithFeedback('hello', { haptic: false })).rejects.toThrow('Write permission denied')
    expect($notifications.get()).toHaveLength(1)
    expect($notifications.get()[0]).toMatchObject({ kind: 'error' })
  })
})
