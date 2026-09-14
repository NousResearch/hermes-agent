import { describe, expect, it } from 'vitest'

import {
  orbParamsForState,
  type OrbState,
  orbStates,
  type OrbStateSignals,
  orbStateVisuals,
  orbTailPhase,
  resolveOrbState
} from './orb-state'
import { defaultOrbConfig } from './orb-url'

const idleSignals: OrbStateSignals = {
  awaitingInput: false,
  busy: false,
  compacting: false,
  draftingTool: false,
  justCompleted: false,
  messageError: false,
  providerWait: false,
  statusReason: undefined,
  statusType: undefined,
  tailPhase: 'none'
}

function signals(overrides: Partial<OrbStateSignals>): OrbStateSignals {
  return { ...idleSignals, ...overrides }
}

describe('resolveOrbState', () => {
  it('is idle when nothing is happening', () => {
    expect(resolveOrbState(signals({}))).toBe('idle')
  })

  it('ranks error above every other signal', () => {
    expect(
      resolveOrbState(
        signals({
          awaitingInput: true,
          compacting: true,
          messageError: true,
          providerWait: true,
          statusType: 'running',
          tailPhase: 'streaming'
        })
      )
    ).toBe('error')
  })

  it('maps an incomplete error reason to error', () => {
    expect(resolveOrbState(signals({ statusReason: 'error', statusType: 'incomplete' }))).toBe('error')
  })

  it('maps incomplete stops without an error payload to cancelled', () => {
    for (const reason of ['cancelled', 'length', 'content-filter', 'other', 'tool-calls']) {
      expect(resolveOrbState(signals({ statusReason: reason, statusType: 'incomplete' }))).toBe('cancelled')
    }
  })

  it('ranks waiting-input above background phases', () => {
    expect(
      resolveOrbState(signals({ awaitingInput: true, compacting: true, providerWait: true, statusType: 'running' }))
    ).toBe('waiting-input')
    expect(resolveOrbState(signals({ statusReason: 'interrupt', statusType: 'requires-action' }))).toBe(
      'waiting-input'
    )
  })

  it('ranks compacting above provider wait and running phases', () => {
    expect(
      resolveOrbState(signals({ compacting: true, providerWait: true, statusType: 'running', tailPhase: 'streaming' }))
    ).toBe('compacting')
  })

  it('maps a provider wait to model-loading', () => {
    expect(resolveOrbState(signals({ providerWait: true, statusType: 'running' }))).toBe('model-loading')
  })

  it('derives the running phase from the tail part', () => {
    expect(resolveOrbState(signals({ statusType: 'running', tailPhase: 'tool-running' }))).toBe('tool-running')
    expect(resolveOrbState(signals({ statusType: 'running', tailPhase: 'tool-result' }))).toBe('tool-result')
    expect(resolveOrbState(signals({ statusType: 'running', tailPhase: 'streaming' }))).toBe('streaming')
    expect(resolveOrbState(signals({ statusType: 'running', tailPhase: 'reasoning' }))).toBe('thinking')
    expect(resolveOrbState(signals({ statusType: 'running', tailPhase: 'none' }))).toBe('thinking')
  })

  it('treats a drafted tool call as tool-running', () => {
    expect(resolveOrbState(signals({ draftingTool: true, statusType: 'running', tailPhase: 'none' }))).toBe(
      'tool-running'
    )
  })

  it('treats session busy without a running message as thinking', () => {
    expect(resolveOrbState(signals({ busy: true, tailPhase: 'none' }))).toBe('thinking')
  })

  it('shows the complete flash before falling back to idle', () => {
    expect(resolveOrbState(signals({ justCompleted: true, statusType: 'complete' }))).toBe('complete')
    expect(resolveOrbState(signals({ justCompleted: false, statusType: 'complete' }))).toBe('idle')
  })
})

describe('orbTailPhase', () => {
  it('returns none for an empty tail', () => {
    expect(orbTailPhase([])).toBe('none')
  })

  it('distinguishes a tool call in flight from a landed result', () => {
    expect(orbTailPhase([{ type: 'tool-call', toolName: 'read_file' }])).toBe('tool-running')
    expect(orbTailPhase([{ type: 'tool-call', toolName: 'read_file', result: 'ok' }])).toBe('tool-result')
  })

  it('ignores silent tools', () => {
    expect(orbTailPhase([{ type: 'tool-call', toolName: 'todo' }])).toBe('other')
  })

  it('maps reasoning and text tails', () => {
    expect(orbTailPhase([{ type: 'reasoning' }])).toBe('reasoning')
    expect(orbTailPhase([{ type: 'text', text: 'hello' }])).toBe('streaming')
    expect(orbTailPhase([{ type: 'text', text: '' }])).toBe('other')
  })

  it('reads only the last part', () => {
    expect(
      orbTailPhase([
        { type: 'tool-call', toolName: 'read_file', result: 'ok' },
        { type: 'text', text: 'done' }
      ])
    ).toBe('streaming')
  })
})

describe('orbStateVisuals', () => {
  it('documents every orb state exactly once', () => {
    expect(Object.keys(orbStateVisuals).sort()).toEqual([...orbStates].sort())
  })

  it('gives every state a distinct speed/exposure/glow signature', () => {
    const signatures = (Object.keys(orbStateVisuals) as OrbState[]).map(
      state => `${orbStateVisuals[state].speed}:${orbStateVisuals[state].exposure}:${orbStateVisuals[state].glow}`
    )

    expect(new Set(signatures).size).toBe(signatures.length)
  })
})

describe('orbParamsForState', () => {
  it('leaves the thinking orb on the user base params', () => {
    const base = defaultOrbConfig()
    const params = orbParamsForState('thinking', base)

    expect(params.speed).toBe(base.speed)
    expect(params.exposure).toBe(base.exposure)
    expect(params.radius).toBe(base.radius)
    expect(params.glowColor).toBe(base.glowColor)
    expect(params.style).toBe(base.style)
  })

  it('applies the state deltas without mutating the base', () => {
    const base = defaultOrbConfig()
    const params = orbParamsForState('error', base)
    const visual = orbStateVisuals.error

    expect(params.speed).toBeCloseTo(base.speed * visual.speed, 10)
    expect(params.exposure).toBeCloseTo(base.exposure * visual.exposure, 10)
    expect(params.glowColor).toBe(visual.glow)
    expect(params.shellEdge).toBe(visual.glow)
    // The orb's body colors stay the user's own.
    expect(params.colorA).toBe(base.colorA)
    expect(params.style).toBe(base.style)
    // Base untouched.
    expect(base.glowColor).not.toBe(visual.glow)
  })

  it('keeps the user glow colors for states without a tint', () => {
    const base = defaultOrbConfig()

    expect(orbParamsForState('streaming', base).glowColor).toBe(base.glowColor)
    expect(orbParamsForState('idle', base).shellEdge).toBe(base.shellEdge)
  })
})
