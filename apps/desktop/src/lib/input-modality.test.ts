import { describe, expect, it } from 'vitest'

import { lastInputModality } from './input-modality'

describe('lastInputModality', () => {
  it('defaults to keyboard before any input', () => {
    expect(lastInputModality()).toBe('keyboard')
  })

  it('tracks the device behind the last interaction', () => {
    document.dispatchEvent(new Event('pointerdown'))

    expect(lastInputModality()).toBe('pointer')

    document.dispatchEvent(new Event('keydown'))

    expect(lastInputModality()).toBe('keyboard')
  })

  it('sees events a handler stops from bubbling (capture phase)', () => {
    const target = document.createElement('button')

    document.body.append(target)
    target.addEventListener('pointerdown', event => event.stopPropagation())
    target.dispatchEvent(new Event('pointerdown', { bubbles: true }))

    expect(lastInputModality()).toBe('pointer')

    target.remove()
  })
})
