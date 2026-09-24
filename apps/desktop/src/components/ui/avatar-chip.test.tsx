import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { AvatarChip, monogramFor } from './avatar-chip'

afterEach(cleanup)

const Glyph = () => <svg data-testid="glyph" />

describe('the identity ladder', () => {
  it('prefers a brand-supplied monogram over the first letter', () => {
    render(<AvatarChip brand={{ color: '#000', monogram: 'n8' }} name="n8n" />)

    expect(screen.getByText('n8')).toBeTruthy()
  })

  it('lets a caller replace the mark entirely, which is how a resolved favicon gets in', () => {
    render(
      <AvatarChip brand={{ Icon: Glyph, color: '#000' }} name="Acme">
        <img alt="" data-testid="favicon" src="data:image/png;base64,iVBOR" />
      </AvatarChip>
    )

    expect(screen.getByTestId('favicon')).toBeTruthy()
    expect(screen.queryByTestId('glyph')).toBeNull()
  })
})

describe('the monogram', () => {
  it.each([
    ['linear', 'L'],
    ['  spaced', 'S'],
    ['', '']
  ])('%s → %s', (name, expected) => {
    expect(monogramFor(name)).toBe(expected)
  })
})
