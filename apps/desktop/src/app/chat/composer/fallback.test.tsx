import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { composerInputSurface } from '@/components/chat/composer-dock'

import { ChatBarFallback } from './index'

const hasBackdropFilter = (classes: string) =>
  classes.includes('backdrop-blur') || classes.includes('backdrop-saturate') || classes.includes('backdrop-filter')

describe('ChatBarFallback', () => {
  it('uses the same filter-free surface treatment as the hydrated composer', () => {
    const { container } = render(<ChatBarFallback />)
    const paint = container.querySelector<HTMLElement>('[aria-hidden="true"]')

    expect(paint).not.toBeNull()
    expect(paint?.className).toContain(composerInputSurface)
    expect(hasBackdropFilter(paint?.className ?? '')).toBe(false)
  })
})
