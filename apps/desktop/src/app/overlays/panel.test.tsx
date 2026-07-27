import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { PanelRowMenu } from './panel'

beforeAll(() => {
  Element.prototype.hasPointerCapture ??= () => false
  Element.prototype.releasePointerCapture ??= () => undefined
  Element.prototype.setPointerCapture ??= () => undefined
  HTMLElement.prototype.scrollIntoView ??= () => undefined
})

describe('PanelRowMenu', () => {
  afterEach(() => {
    cleanup()
  })

  it('opens its actions menu when the trigger has a tooltip', async () => {
    const onSelect = vi.fn()

    render(<PanelRowMenu items={[{ label: 'Rename', onSelect }]} />)

    const trigger = screen.getByRole('button', { name: 'Actions' })

    fireEvent.pointerMove(trigger, { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toContain('Actions')

    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename' }))

    expect(onSelect).toHaveBeenCalledOnce()
  })
})
