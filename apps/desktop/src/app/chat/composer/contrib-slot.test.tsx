import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { COMPOSER_AREAS, type ComposerRenderContext } from './contrib'
import { ComposerRenderSlot } from './contrib-slot'

const disposers: Array<() => void> = []

/** Register a composer render-area contribution whose render receives the
 *  slot's context argument (the seam under test). */
function contribute(
  area: string,
  render: (ctx: ComposerRenderContext | undefined) => ReactNode,
  order?: number
) {
  disposers.push(
    registry.register({
      area,
      id: `t-${disposers.length}`,
      order,
      render: render as (ctx?: unknown) => ReactNode
    })
  )
}

afterEach(() => {
  disposers.splice(0).forEach(d => d())
  cleanup()
})

describe('ComposerRenderSlot', () => {
  const ctx: ComposerRenderContext = {
    insertText: vi.fn()
  }

  it('renders nothing when the area is empty', () => {
    const { container } = render(<ComposerRenderSlot area={COMPOSER_AREAS.actions} ctx={ctx} />)

    expect(container.textContent).toBe('')
  })

  it('hands the composer edit bridge to render-area contributions', () => {
    let seen: ComposerRenderContext | undefined

    contribute(COMPOSER_AREAS.actions, c => {
      seen = c

      return <button type="button">bold</button>
    })

    render(<ComposerRenderSlot area={COMPOSER_AREAS.actions} ctx={ctx} />)

    expect(screen.getByRole('button', { name: 'bold' })).toBeTruthy()
    expect(seen?.insertText).toBe(ctx.insertText)
  })

  it('passes insertText through to the bridge implementation', () => {
    let seen: ComposerRenderContext | undefined

    contribute(COMPOSER_AREAS.actions, c => {
      seen = c

      return null
    })

    render(<ComposerRenderSlot area={COMPOSER_AREAS.actions} ctx={ctx} />)

    seen?.insertText('**bold**')

    expect(ctx.insertText).toHaveBeenCalledWith('**bold**')
  })

  it('renders contributions in registry order', () => {
    contribute(COMPOSER_AREAS.actions, () => <span>first</span>, 20)
    contribute(COMPOSER_AREAS.actions, () => <span>second</span>, 10)

    const { container } = render(<ComposerRenderSlot area={COMPOSER_AREAS.actions} ctx={ctx} />)

    expect(container.textContent).toBe('secondfirst')
  })

  it('keeps a crashing contribution isolated from the rest', () => {
    contribute(COMPOSER_AREAS.actions, () => {
      throw new Error('broken render')
    })
    contribute(COMPOSER_AREAS.actions, () => <span>fine</span>)

    const { container } = render(<ComposerRenderSlot area={COMPOSER_AREAS.actions} ctx={ctx} />)

    expect(container.textContent).toContain('fine')
  })
})
