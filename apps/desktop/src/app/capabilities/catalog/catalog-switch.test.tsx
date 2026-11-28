import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, expect, it } from 'vitest'

import { CatalogSwitch } from './catalog-switch'

afterEach(cleanup)

function Harness() {
  const [checked, setChecked] = useState(true)

  return <CatalogSwitch aria-label="Memory" checked={checked} onCheckedChange={setChecked} />
}

// Both captions stay mounted (one is `invisible`) so the switch never shifts;
// the shown one must name the switch's current state, never the click's action.
const shownCaption = () =>
  Array.from(screen.getByText('On').parentElement!.children).find(el => !el.classList.contains('invisible'))
    ?.textContent

it('captions the switch with its current state and follows a flip', () => {
  render(<Harness />)
  const toggle = screen.getByRole('switch', { name: 'Memory' })

  expect(toggle.getAttribute('aria-checked')).toBe('true')
  expect(shownCaption()).toBe('On')

  fireEvent.click(toggle)

  expect(toggle.getAttribute('aria-checked')).toBe('false')
  expect(shownCaption()).toBe('Off')
})
