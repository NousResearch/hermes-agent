import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $turnOrigin } from '@/store/session'

import { ResponseLoadingIndicator } from './status'

describe('turn-origin loading attribution', () => {
  afterEach(() => {
    cleanup()
    $turnOrigin.set(null)
  })

  it('labels notification work as background-result processing', () => {
    $turnOrigin.set('notification')
    render(<ResponseLoadingIndicator />)

    expect(screen.getByRole('status', { name: 'Processing background result' })).toBeTruthy()
  })
})
