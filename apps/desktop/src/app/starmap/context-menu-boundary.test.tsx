import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { StarmapContextMenuBoundary } from './context-menu-boundary'

describe('StarmapContextMenuBoundary', () => {
  afterEach(() => {
    cleanup()
  })

  it('marks the Star Map as owning plain right-clicks', () => {
    render(
      <StarmapContextMenuBoundary>
        <canvas data-testid="starmap-canvas" />
      </StarmapContextMenuBoundary>
    )

    const canvas = screen.getByTestId('starmap-canvas')
    expect(canvas.closest('[data-context-menu-skip]')).toBeTruthy()
  })
})
