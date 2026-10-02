import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { ProfileGlyph } from './profile-glyph'

afterEach(cleanup)

describe('ProfileGlyph', () => {
  it('preserves the first readable grapheme across scripts and emoji', () => {
    const names = [
      '研究助手',
      'Работа',
      'Ελληνικά',
      'Álvaro',
      'A\u0301lvaro',
      '👨‍💻',
      '🇨🇳',
      '1️⃣ helper',
      '𠮷野',
      '--dev',
      ''
    ]

    const expected = ['研', 'Р', 'Ε', 'Á', 'A\u0301', '👨‍💻', '🇨🇳', '1️⃣', '𠮷', 'd', '?']

    const { container } = render(
      <>
        {names.map(name => (
          <ProfileGlyph color={null} isDefault={false} key={name} name={name} />
        ))}
      </>
    )

    expect(Array.from(container.children, child => child.textContent)).toEqual(expected)
  })

  it('keeps the default profile home mark instead of using an initial', () => {
    const { container } = render(<ProfileGlyph color={null} isDefault name="研究助手" />)

    expect(container.textContent).toBe('')
    expect(container.querySelector('.codicon-home')).not.toBeNull()
  })
})
