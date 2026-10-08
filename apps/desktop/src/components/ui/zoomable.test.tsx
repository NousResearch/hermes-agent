import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { Zoomable } from './zoomable'

afterEach(cleanup)

describe('Zoomable', () => {
  it('opens the full-view overlay when the trigger is clicked', () => {
    render(
      <Zoomable label="Open diagram" overlay={<div data-testid="overlay">Expanded diagram</div>}>
        <div>Inline diagram</div>
      </Zoomable>
    )

    expect(screen.queryByTestId('overlay')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Open diagram' }))
    expect(screen.getByTestId('overlay')).toBeTruthy()
  })
})
