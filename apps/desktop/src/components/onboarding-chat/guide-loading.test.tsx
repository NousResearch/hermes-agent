import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { GuideLoading } from './guide-loading'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it.each([false, true])('keeps startup visible without chat controls (reduced motion: %s)', reducedMotion => {
  vi.stubGlobal('matchMedia', () => ({
    matches: reducedMotion,
    addEventListener: () => {},
    removeEventListener: () => {}
  }))
  let next: FrameRequestCallback | undefined
  const cancel = vi.fn()
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    next = callback

    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', cancel)

  const { container, unmount } = render(
    <I18nProvider>
      <GuideLoading />
    </I18nProvider>
  )

  const status = screen.getByRole('status')
  expect(status.getAttribute('aria-busy')).toBe('true')
  expect(status.textContent).toMatch(/Starting Hermes Desktop/)
  expect(container.querySelector('input, textarea, [contenteditable], button')).toBeNull()

  if (reducedMotion) {
    expect(next).toBeUndefined()
    expect(status.querySelector('img')).not.toBeNull()
  } else {
    const path = status.querySelector('path')
    const before = path?.getAttribute('d')
    act(() => next?.(performance.now() + 60_000))
    expect(path?.getAttribute('d')).not.toBe(before)
    expect(screen.getByRole('status')).toBe(status)
  }

  unmount()

  if (!reducedMotion) {
    expect(cancel).toHaveBeenCalled()
  }
})
