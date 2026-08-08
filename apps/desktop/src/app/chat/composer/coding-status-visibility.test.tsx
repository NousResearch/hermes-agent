import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { CodingStatusVisibility, shouldShowCodingStatus } from './coding-status-visibility'

afterEach(cleanup)

describe('shouldShowCodingStatus', () => {
  it('keeps the row visible when config is unavailable or unset', () => {
    expect(shouldShowCodingStatus(undefined)).toBe(true)
    expect(shouldShowCodingStatus({})).toBe(true)
    expect(shouldShowCodingStatus({ display: {} })).toBe(true)
  })

  it('hides the row only when the setting is explicitly false', () => {
    expect(shouldShowCodingStatus({ display: { show_coding_status: false } })).toBe(false)
    expect(shouldShowCodingStatus({ display: { show_coding_status: true } })).toBe(true)
  })
})

describe('CodingStatusVisibility', () => {
  it('does not mount its row when the setting is explicitly off', () => {
    render(
      <CodingStatusVisibility config={{ display: { show_coding_status: false } }}>
        <div data-testid="coding-status-row" />
      </CodingStatusVisibility>
    )

    expect(screen.queryByTestId('coding-status-row')).toBeNull()
  })

  it('mounts its row by default and when explicitly on', () => {
    const { rerender } = render(
      <CodingStatusVisibility config={undefined}>
        <div data-testid="coding-status-row" />
      </CodingStatusVisibility>
    )

    expect(screen.getByTestId('coding-status-row')).not.toBeNull()

    rerender(
      <CodingStatusVisibility config={{ display: { show_coding_status: true } }}>
        <div data-testid="coding-status-row" />
      </CodingStatusVisibility>
    )

    expect(screen.getByTestId('coding-status-row')).not.toBeNull()
  })
})
