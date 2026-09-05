import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { AvatarChip, monogramFor } from './avatar-chip'

afterEach(cleanup)

const Glyph = () => <svg data-testid="glyph" />

describe('the identity ladder', () => {
  it('paints a brand glyph on a tint of its own color', () => {
    render(<AvatarChip brand={{ Icon: Glyph, color: '#5E6AD2' }} data-testid="chip" name="Linear" />)

    const background = screen.getByTestId('chip').style.backgroundColor

    expect(screen.getByTestId('glyph')).toBeTruthy()
    // jsdom normalises the hex, so assert the tint rather than the literal.
    expect(background).toContain('rgb(94, 106, 210)')
    expect(background).toContain('16%')
  })

  it('falls back to a letter when we have no mark for the name', () => {
    render(<AvatarChip name="Zulip" />)

    expect(screen.getByText('Z')).toBeTruthy()
  })

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

  it('leaves a monochrome mark to follow the surrounding text color rather than vanish into the theme', () => {
    render(<AvatarChip brand={{ Icon: Glyph, color: '#0E1128', monochrome: true }} data-testid="chip" name="Unreal" />)

    expect(screen.getByTestId('chip').style.color).toBe('')
  })

  it('carries an overlay for a status dot', () => {
    render(<AvatarChip name="Acme" overlay={<span data-testid="dot" />} />)

    expect(screen.getByTestId('dot')).toBeTruthy()
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
