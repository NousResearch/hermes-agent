import { describe, expect, it } from 'vitest'

import { applyWindowsBackdrop, type TranslucencyState, type WindowsBackgroundMaterial } from './translucency'

const glass = (intensity: number): TranslucencyState => ({
  mode: 'glass',
  intensity,
  fade: 0,
  material: 'under-window',
  scope: 'sidebar'
})

function electronWindow() {
  let background = '#initial'
  let materialWrites = 0
  let colourWrites = 0

  const win = {
    setBackgroundColor(colour: string) {
      background = colour
      colourWrites++
    },
    setBackgroundMaterial(material: WindowsBackgroundMaterial) {
      materialWrites++
      // Electron 40.10.2 BrowserWindow::SetBackgroundMaterial also sets the
      // backing: white for 'none', transparent for an active material.
      win.setBackgroundColor(material === 'none' ? '#FFFFFFFF' : '#00000000')
    }
  }

  return { win, background: () => background, writes: () => ({ materialWrites, colourWrites }) }
}

describe('Windows backdrop updates', () => {
  it.each([
    { state: glass(0), backing: true, themedColor: '#111111' },
    { state: glass(0), backing: false, themedColor: '#f7f7f7' },
    { state: { ...glass(60), mode: 'clear' as const }, backing: false, themedColor: '#242424' }
  ])('keeps $themedColor after clearing the material (backing=$backing)', ({ state, backing, themedColor }) => {
    const window = electronWindow()

    applyWindowsBackdrop(window.win, state, { backing, material: true }, themedColor)

    expect(window.background()).toBe(themedColor)
    expect(window.writes().materialWrites).toBe(1)
  })

  it('keeps active glass transparent when changing its material', () => {
    const window = electronWindow()

    applyWindowsBackdrop(
      window.win,
      { ...glass(60), material: 'popover' },
      { backing: false, material: true },
      '#111111'
    )

    expect(window.background()).toBe('#00000000')
  })

  it('can update only the backing without touching the material', () => {
    const window = electronWindow()

    applyWindowsBackdrop(window.win, glass(0), { backing: true, material: false }, '#222222')

    expect(window.background()).toBe('#222222')
    expect(window.writes().materialWrites).toBe(0)
  })

  it('does not write native state for tint-only or opacity-only changes', () => {
    const window = electronWindow()

    applyWindowsBackdrop(window.win, glass(60), { backing: false, material: false }, '#111111')

    expect(window.background()).toBe('#initial')
    expect(window.writes()).toEqual({ materialWrites: 0, colourWrites: 0 })
  })
})
