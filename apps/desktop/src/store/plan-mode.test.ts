import { afterEach, describe, expect, it, vi } from 'vitest'

import { applyPlanMode, setPlanMode, togglePlanMode, withPlanMode } from './plan-mode'

afterEach(() => setPlanMode(false))

describe('applyPlanMode', () => {
  it('routes a typed, text-only message through /plan only while plan mode is on', () => {
    expect(applyPlanMode('build the thing')).toBe('build the thing')

    togglePlanMode()

    expect(applyPlanMode('  build the thing\n')).toBe('/plan build the thing')
    expect(applyPlanMode('   ')).toBe('   ')
    // An explicit command wins, and an explicit /plan is never doubled.
    expect(applyPlanMode('/help')).toBe('/help')
    expect(applyPlanMode('/plan do x')).toBe('/plan do x')
  })

  it('leaves machine text and attachment sends untouched', () => {
    setPlanMode(true)
    const image = { id: 'a', kind: 'image', label: 'shot.png' } as never

    expect(applyPlanMode('look at this', { attachments: [image] })).toBe('look at this')
    expect(applyPlanMode('setup note', { displayKind: 'hidden' })).toBe('setup note')
    expect(applyPlanMode('<expanded skill body>', { displayText: '/work fix it' })).toBe('<expanded skill body>')
    expect(applyPlanMode('spoken words', { surface: 'voice-live' })).toBe('spoken words')
  })

  it('withPlanMode sends the routed text and passes the options through', async () => {
    setPlanMode(true)
    const submit = vi.fn(async () => true)

    await withPlanMode(submit)('draft', { composerScope: 's1' })

    expect(submit).toHaveBeenCalledWith('/plan draft', { composerScope: 's1' })
  })
})
