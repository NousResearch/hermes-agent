import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $planMode, applyPlanMode, setPlanMode, togglePlanMode } from './plan-mode'

beforeEach(() => {
  setPlanMode(false)
})

afterEach(() => {
  setPlanMode(false)
})

describe('plan mode toggle', () => {
  it('toggles on and off', () => {
    expect($planMode.get()).toBe(false)
    togglePlanMode()
    expect($planMode.get()).toBe(true)
    togglePlanMode()
    expect($planMode.get()).toBe(false)
  })
})

describe('applyPlanMode', () => {
  it('is a no-op when plan mode is off', () => {
    setPlanMode(false)
    expect(applyPlanMode('build the thing')).toBe('build the thing')
  })

  it('prepends /plan when on', () => {
    setPlanMode(true)
    expect(applyPlanMode('build the thing')).toBe('/plan build the thing')
  })

  it('preserves leading whitespace in the original text', () => {
    setPlanMode(true)
    // The raw text is kept after the directive (only trimStart is inspected).
    expect(applyPlanMode('  hello')).toBe('/plan   hello')
  })

  it('never hijacks an explicit slash command', () => {
    setPlanMode(true)
    expect(applyPlanMode('/help')).toBe('/help')
    expect(applyPlanMode('  /resume abc')).toBe('  /resume abc')
  })

  it('leaves an empty / whitespace-only submission untouched', () => {
    setPlanMode(true)
    expect(applyPlanMode('')).toBe('')
    expect(applyPlanMode('   ')).toBe('   ')
  })

  it('does not double-prefix a message that already starts with /plan', () => {
    setPlanMode(true)
    // /plan starts with '/', so the slash-guard prevents a second prefix.
    expect(applyPlanMode('/plan do x')).toBe('/plan do x')
  })
})
