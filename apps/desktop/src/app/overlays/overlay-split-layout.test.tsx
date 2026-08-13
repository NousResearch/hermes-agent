import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { OverlayMain, OverlaySidebar, OverlaySplitLayout } from './overlay-split-layout'

describe('OverlaySplitLayout', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders a two-column grid with a fixed sidebar and flexible main', () => {
    const { container } = render(
      <OverlaySplitLayout>
        <div>sidebar</div>
        <div>main</div>
      </OverlaySplitLayout>
    )

    const grid = container.firstElementChild as HTMLElement

    expect(grid.className).toContain('grid')
    expect(grid.className).toContain('grid-cols-[13rem_minmax(0,1fr)]')
  })

  it('collapses to a single column below 47.5rem', () => {
    const { container } = render(
      <OverlaySplitLayout>
        <div>sidebar</div>
        <div>main</div>
      </OverlaySplitLayout>
    )

    const grid = container.firstElementChild as HTMLElement

    expect(grid.className).toContain('max-[47.5rem]:grid-cols-1')
  })
})

describe('OverlayMain', () => {
  afterEach(() => {
    cleanup()
  })

  it('left-aligns content (no mx-auto) so it sits flush against the sidebar', () => {
    render(
      <OverlayMain>
        <p>content</p>
      </OverlayMain>
    )

    const main = screen.getByRole('main')

    expect(main.className).not.toContain('mx-auto')
  })

  it('caps content width on ultrawide displays via PAGE_MAX_W', () => {
    render(
      <OverlayMain>
        <p>content</p>
      </OverlayMain>
    )

    const main = screen.getByRole('main')

    expect(main.className).toContain('max-w-[75rem]')
  })

  it('applies the responsive horizontal clamp gutter', () => {
    render(
      <OverlayMain>
        <p>content</p>
      </OverlayMain>
    )

    const main = screen.getByRole('main')

    expect(main.className).toContain('px-[clamp(0.8333rem,2.6667vw,2.6667rem)]')
  })

  it('merges consumer className overrides', () => {
    render(
      <OverlayMain className="px-0 pb-0">
        <p>content</p>
      </OverlayMain>
    )

    const main = screen.getByRole('main')

    expect(main.className).toContain('px-0')
    expect(main.className).toContain('pb-0')
  })
})

describe('OverlaySidebar', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders an aside with the sidebar surface background', () => {
    render(
      <OverlaySidebar>
        <nav>links</nav>
      </OverlaySidebar>
    )

    const aside = screen.getByRole('complementary')

    expect(aside.className).toContain('bg-(--ui-sidebar-surface-background)')
  })

  it('includes the shared overlay top clearance', () => {
    render(
      <OverlaySidebar>
        <nav>links</nav>
      </OverlaySidebar>
    )

    const aside = screen.getByRole('complementary')

    expect(aside.className).toContain('pt-[calc(var(--titlebar-height)/2-0.4375rem)]')
  })
})
