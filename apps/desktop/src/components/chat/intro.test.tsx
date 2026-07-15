// @vitest-environment jsdom
import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { Intro } from './intro'

describe('Intro', () => {
  it('renders a configured landing background image behind the intro', () => {
    const { container } = render(
      <Intro backgroundImage="https://example.com/profile-background.png" personality="default" seed={1} />
    )

    const image = container.querySelector<HTMLImageElement>('[data-slot="aui_intro_background"]')
    expect(image?.getAttribute('src')).toBe('https://example.com/profile-background.png')
    expect(image?.getAttribute('alt')).toBe('')
  })

  it('preserves the existing landing page when no background is configured', () => {
    const { container } = render(<Intro personality="default" seed={1} />)

    expect(container.querySelector('[data-slot="aui_intro_background"]')).toBeNull()
    expect(container.querySelector('[data-slot="aui_intro"]')).toBeTruthy()
  })
})
