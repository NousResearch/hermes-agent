import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { SessionTokensStatus } from './session-tokens-status'

afterEach(cleanup)

describe('SessionTokensStatus', () => {
  it('renders cumulative input and output as separate compact counters', () => {
    render(<SessionTokensStatus input="114.6k" output="12.3k" />)

    expect(screen.getByLabelText('Session tokens: 114.6k in, 12.3k out')).toBeTruthy()
    expect(screen.getByText('In')).toBeTruthy()
    expect(screen.getByText('114.6k')).toBeTruthy()
    expect(screen.getByText('Out')).toBeTruthy()
    expect(screen.getByText('12.3k')).toBeTruthy()
  })
})
