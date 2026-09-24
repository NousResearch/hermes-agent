import { describe, expect, it } from 'vitest'

import { createHoldCommandDictation } from './hold-command-dictation'

const command = (overrides: Partial<KeyboardEvent> = {}) =>
  ({ altKey: false, ctrlKey: false, key: 'Meta', metaKey: true, repeat: false, shiftKey: false, ...overrides }) as KeyboardEvent

describe('hold-Command dictation', () => {
  it('starts after a lone Command hold and stops when Command is released', () => {
    const hold = createHoldCommandDictation()

    expect(hold.keyDown(command())).toBe('arm')
    expect(hold.begin()).toBe('start')
    expect(hold.keyUp(command())).toBe('stop')
  })

  it('does not start dictation for Command chords', () => {
    const hold = createHoldCommandDictation()

    expect(hold.keyDown(command())).toBe('arm')
    expect(hold.keyDown(command({ key: 'c' }))).toBe('cancel')
    expect(hold.begin()).toBe('ignore')
    expect(hold.keyUp(command())).toBe('ignore')
  })

  it('ignores repeated or mixed-modifier Command presses', () => {
    const hold = createHoldCommandDictation()

    expect(hold.keyDown(command({ repeat: true }))).toBe('ignore')
    expect(hold.keyDown(command({ shiftKey: true }))).toBe('ignore')
    expect(hold.keyDown(command({ ctrlKey: true }))).toBe('ignore')
    expect(hold.keyDown(command({ altKey: true }))).toBe('ignore')
  })
})
