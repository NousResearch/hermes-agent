import { act, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'

import { FloatingPet } from '@/components/pet/floating-pet'
import { type PetInfo, setPetInfo } from '@/store/pet'

// Shared by the FloatingPet tests; each file still mocks the gateway hook,
// theme and the canvas sprite (labelled like the real one) itself.
export const PET = {
  displayName: 'Boba',
  enabled: true,
  frameH: 208,
  frameW: 192,
  loopMs: 1000,
  scale: 0.5,
  spritesheetBase64: 'stub',
  stateRows: ['idle', 'giggle', 'dangle', 'tumble', 'thud']
}

/** Render the in-window pet (installing `info` first if given); returns its container. */
export function mountPet(info?: PetInfo): HTMLElement {
  if (info) {
    setPetInfo(info)
  }

  render(
    <MemoryRouter>
      <FloatingPet />
    </MemoryRouter>
  )

  return screen.queryByLabelText('Boba pet')?.parentElement?.parentElement as HTMLElement
}

type Point = [number, number]

/** Press at `from`, drag to `to` (if given), run `whileHeld`, release. */
export function press(el: HTMLElement, from: Point, to?: Point, whileHeld?: () => void): void {
  const [x, y] = to ?? from

  act(() => {
    fireEvent.pointerDown(el, { clientX: from[0], clientY: from[1], pointerId: 1 })

    if (to) {
      fireEvent.pointerMove(el, { clientX: x, clientY: y, pointerId: 1 })
    }
  })
  whileHeld?.()
  act(() => void fireEvent.pointerUp(el, { clientX: x, clientY: y, pointerId: 1 }))
}
