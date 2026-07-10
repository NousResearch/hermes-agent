import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { Tip, TipHintLabel } from './tooltip'

describe('TooltipContent', () => {
  afterEach(() => {
    cleanup()
  })

  it('forces a block-level label child back to inline-flex so it cannot break the decoration geometry', async () => {
    render(
      <Tip label={<span className="flex items-center gap-2">broken label</span>}>
        <button type="button">trigger</button>
      </Tip>
    )

    fireEvent.pointerMove(screen.getByRole('button', { name: 'trigger' }), { pointerType: 'mouse' })
    await screen.findByRole('tooltip')

    const content = document.querySelector<HTMLElement>('[data-slot="tooltip-content"]')
    const decoration = content?.firstElementChild
    const label = decoration?.firstElementChild as HTMLElement | null | undefined

    expect(label).not.toBeNull()
    expect(label && getComputedStyle(label).display).toBe('inline')
  })

  it('keeps TipHintLabel inline by default when a hint is present', async () => {
    render(
      <Tip label={<TipHintLabel hint="Ctrl+`" text="PowerShell" />}>
        <button type="button">trigger</button>
      </Tip>
    )

    fireEvent.pointerMove(screen.getByRole('button', { name: 'trigger' }), { pointerType: 'mouse' })
    await screen.findByRole('tooltip')

    const content = document.querySelector<HTMLElement>('[data-slot="tooltip-content"]')
    const decoration = content?.firstElementChild
    const label = decoration?.firstElementChild

    expect(label?.classList.contains('inline-flex')).toBe(true)
    expect(label?.classList.contains('flex')).toBe(false)
  })
})
