import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { composerInputBacking } from '@/components/chat/composer-dock'

import { ChatBarFallback } from './index'

describe('ChatBarFallback', () => {
  it('paints the shared composerInputBacking layer', () => {
    const { container } = render(<ChatBarFallback />)
    const paint = container.querySelector<HTMLElement>('[aria-hidden="true"]')

    expect(paint?.className).toBe(composerInputBacking)
  })
})
