import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $composerPopoutGesturesEnabled } from '@/store/composer-popout'

import { ComposerLockSetting } from './composer-lock-setting'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        appearance: {
          composerPopoutTitle: 'Lock composer to the bottom',
          composerPopoutDesc: 'Stops the composer from peeling out of the dock when you drag.'
        }
      }
    }
  })
}))

describe('ComposerLockSetting', () => {
  beforeEach(() => {
    $composerPopoutGesturesEnabled.set(false)
  })

  afterEach(() => {
    cleanup()
    $composerPopoutGesturesEnabled.set(false)
  })

  it('is on (locked) when pop-out gestures are disabled', () => {
    render(<ComposerLockSetting />)

    expect(screen.getByRole('switch', { name: 'Lock composer to the bottom' }).getAttribute('aria-checked')).toBe(
      'true'
    )
  })

  it('turning the lock off re-enables composer drag (#70422)', () => {
    render(<ComposerLockSetting />)

    fireEvent.click(screen.getByRole('switch', { name: 'Lock composer to the bottom' }))

    expect($composerPopoutGesturesEnabled.get()).toBe(true)
  })
})
