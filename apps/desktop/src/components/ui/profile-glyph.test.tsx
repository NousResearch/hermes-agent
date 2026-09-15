import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ProfileGlyph } from './profile-glyph'

afterEach(cleanup)

describe('ProfileGlyph', () => {
  it('shows the first Unicode grapheme for a non-default profile', () => {
    render(<ProfileGlyph color={null} data-testid="glyph" isDefault={false} name="研究助手" />)

    expect(screen.getByTestId('glyph').textContent).toBe('研')
  })
})
