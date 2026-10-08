import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { closeQuestionnaire, goToStep, openQuestionnaire } from '../store'

import { LayoutStep } from './looks'

afterEach(() => {
  cleanup()
  closeQuestionnaire('skipped')
})

// jsdom has no layout, so the box size is read from its sizing classes.
const sizeClasses = (element: Element | null) =>
  (element?.className ?? '').split(/\s+/).filter(name => /^(h|w|aspect|max-w)-/.test(name))

describe('LayoutStep', () => {
  it('renders Basic and Elite previews in one fixed box size, whatever their panes or copy', () => {
    openQuestionnaire()
    goToStep('layout')
    render(<LayoutStep />)

    const basic = screen.getByRole('button', { name: /Basic/ })
    const elite = screen.getByRole('button', { name: /Elite/ })
    const basicBox = sizeClasses(basic.firstElementChild)

    expect(basicBox).toEqual(sizeClasses(elite.firstElementChild))
    // A box sized by its own width and height, not by the button's content width.
    expect(basicBox.some(name => name.startsWith('h-'))).toBe(true)
    expect(basicBox.some(name => /^w-(\d|\[)/.test(name))).toBe(true)
    expect(basicBox).not.toContain('w-full')
    // Equal cells: each button fills its card, so its label row starts under a box of the same height.
    expect(sizeClasses(basic)).toContain('w-full')
    expect(sizeClasses(elite)).toContain('w-full')
  })
})
