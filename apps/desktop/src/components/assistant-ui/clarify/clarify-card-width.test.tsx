import { render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $clarifyCardWidth, setClarifyCardWidth } from '@/store/clarify-card-width'

import { ClarifyShell } from './core/shell'

afterEach(() => {
  setClarifyCardWidth('full')
})

describe('clarify card width', () => {
  it('spans the full message column by default', () => {
    expect($clarifyCardWidth.get()).toBe('full')
    const view = render(<ClarifyShell>question</ClarifyShell>)
    const card = view.container.querySelector('[data-slot="clarify-inline"]')

    expect(card?.className).toContain('w-full')
    expect(card?.className).toContain('max-w-full')
  })

  it('keeps the narrow reading width when set to compact', () => {
    setClarifyCardWidth('compact')
    const view = render(<ClarifyShell>question</ClarifyShell>)
    const card = view.container.querySelector('[data-slot="clarify-inline"]')

    expect(card?.className).toContain('max-w-md')
    expect(card?.className).not.toContain('max-w-full')
  })
})
