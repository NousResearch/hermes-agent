// Declares the statusbar toggle's availability, as the app does at boot.
import '@/store/keybinds'

import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { $shellChoice } from '@/store/shell-variant'

import { PALETTE_AREA, paletteToggle, usePaletteContributions } from './contrib'

let rows: string[] = []

function Probe() {
  rows = usePaletteContributions().map(row => row.label)

  return null
}

const toggle = (id: string, label: string, action?: string) =>
  paletteToggle({ id, label, action, get: () => false, set: () => undefined })

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  $shellChoice.set(null)
})

it("keeps the statusbar toggle at its registered position in classic and drops it where the shell has no strip", () => {
  $shellChoice.set(false)
  disposers.push(
    registry.registerMany([
      toggle('test.reset', 'Reset layout'),
      toggle('view.toggleStatusbar', 'Toggle status bar', 'view.toggleStatusbar'),
      toggle('test.rail', 'Toggle profile rail')
    ])
  )
  render(<Probe />)

  expect(rows).toEqual(['Reset layout', 'Toggle status bar', 'Toggle profile rail'])

  act(() => $shellChoice.set(true))
  expect(rows).toEqual(['Reset layout', 'Toggle profile rail'])

  act(() => $shellChoice.set(false))
  expect(rows).toEqual(['Reset layout', 'Toggle status bar', 'Toggle profile rail'])
})

it('keeps rows whose action is outside the keybind catalog', () => {
  disposers.push(registry.register({ ...toggle('test.plugin', 'Plugin row', 'plugin.unknown'), area: PALETTE_AREA }))
  render(<Probe />)

  expect(rows).toContain('Plugin row')
})
