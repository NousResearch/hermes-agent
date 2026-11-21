import { afterEach, describe, expect, it } from 'vitest'

import { $comboIndex, bindingsFor, conflictsFor } from '@/store/keybinds'
import { $newAgentRequest, hostNewAgentRequests, requestNewAgent } from '@/store/new-agent-request'

import { KEYBIND_ACTIONS } from './actions'
import { canonicalizeCombo } from './combo'

const unmounts: (() => void)[] = []

afterEach(() => {
  unmounts.splice(0).forEach(unmount => unmount())
})

describe('agents.new', () => {
  it('ships the new-chat chord, layered before session.new so it can hand the chord on', () => {
    const ids = KEYBIND_ACTIONS.map(action => action.id)

    expect(bindingsFor('agents.new')).toEqual(bindingsFor('session.new'))
    expect(bindingsFor('agents.new').length).toBeGreaterThan(0)
    expect(ids.indexOf('agents.new')).toBeLessThan(ids.indexOf('session.new'))
  })

  it('shares its chord with nothing but session.new and reports no conflict', () => {
    for (const combo of bindingsFor('agents.new')) {
      expect($comboIndex.get().get(canonicalizeCombo(combo))).toEqual(['agents.new', 'session.new'])
      expect(conflictsFor('agents.new', combo)).toEqual([])
      expect(conflictsFor('session.new', combo)).toEqual([])
    }
  })

  it('declines when no New agent button is showing, so the chord starts a new chat', () => {
    expect(requestNewAgent()).toBe(false)
  })

  it('asks the newest mounted New agent button, and stops once it unmounts', () => {
    const panel = Symbol('panel')
    const header = Symbol('header')
    unmounts.push(hostNewAgentRequests(panel))
    const unmountHeader = hostNewAgentRequests(header)

    expect(requestNewAgent()).toBe(true)
    expect($newAgentRequest.get()?.host).toBe(header)

    unmountHeader()
    const before = $newAgentRequest.get()?.seq ?? 0
    expect(requestNewAgent()).toBe(true)
    expect($newAgentRequest.get()).toEqual({ host: panel, seq: before + 1 })

    unmounts.splice(0).forEach(unmount => unmount())
    expect(requestNewAgent()).toBe(false)
  })
})
