import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { host } from '@/sdk'
import {
  $composerSelectModelAction,
  $selectedStoredSessionId,
  type ComposerModelSelection,
  setActiveSessionId
} from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

// host.composer.selectModel is a thin, fail-closed door onto the model menu's own selectModel
// (registered by the main window's wiring). These tests pin its address resolution and contract;
// the switch semantics themselves (paint + config.set + rollback + confirm) are the menu's.
describe('host.composer.selectModel', () => {
  afterEach(() => {
    $composerSelectModelAction.set(null)
    setActiveSessionId(null)
    $selectedStoredSessionId.set(null)
    clearAllSessionStates()
  })

  const register = (result = true) => {
    const calls: ComposerModelSelection[] = []

    $composerSelectModelAction.set(async selection => {
      calls.push(selection)

      return result
    })

    return calls
  }

  it('routes the primary chat (null, its runtime id, or its stored id) to the primary surface', async () => {
    const calls = register()
    setActiveSessionId('rt-main')
    $selectedStoredSessionId.set('stored-main')

    await expect(host.composer.selectModel(null, { model: ' m/1 ', provider: 'tensorfold' })).resolves.toBe(true)
    await expect(host.composer.selectModel('rt-main', { model: 'm/2', provider: 'tensorfold' })).resolves.toBe(true)
    await expect(host.composer.selectModel('stored-main', { model: 'm/3', provider: 'tensorfold' })).resolves.toBe(true)

    expect(calls).toEqual([
      { model: 'm/1', provider: 'tensorfold', sessionId: null },
      { model: 'm/2', provider: 'tensorfold', sessionId: null },
      { model: 'm/3', provider: 'tensorfold', sessionId: null }
    ])
  })

  it('routes a tile session by runtime or stored id to that runtime id, never to the primary', async () => {
    const calls = register()
    setActiveSessionId('rt-main')
    $selectedStoredSessionId.set('stored-main')
    publishSessionState('rt-tile', { ...createClientSessionState(), storedSessionId: 'stored-tile' })

    await host.composer.selectModel('rt-tile', { model: 'a', provider: 'p' })
    await host.composer.selectModel('stored-tile', { model: 'b', provider: 'p' })

    expect(calls.map(call => call.sessionId)).toEqual(['rt-tile', 'rt-tile'])
  })

  it("addresses 'new' only while the primary has no session; unknown ids fail closed", async () => {
    const calls = register()

    await expect(host.composer.selectModel('new', { model: 'a', provider: 'p' })).resolves.toBe(true)
    setActiveSessionId('rt-main')
    await expect(host.composer.selectModel('new', { model: 'b', provider: 'p' })).resolves.toBe(false)
    await expect(host.composer.selectModel('nobody', { model: 'c', provider: 'p' })).resolves.toBe(false)

    expect(calls).toEqual([{ model: 'a', provider: 'p', sessionId: null }])
  })

  it('returns false without calling the menu for empty input or before the wiring registers', async () => {
    await expect(host.composer.selectModel(null, { model: 'a', provider: 'p' })).resolves.toBe(false)

    const calls = register()
    await expect(host.composer.selectModel(null, { model: '  ', provider: 'p' })).resolves.toBe(false)
    await expect(host.composer.selectModel(null, { model: 'a', provider: '' })).resolves.toBe(false)
    expect(calls).toEqual([])
  })

  it("passes through the menu's answer and turns a throw into false", async () => {
    register(false)
    await expect(host.composer.selectModel(null, { model: 'a', provider: 'p' })).resolves.toBe(false)

    $composerSelectModelAction.set(vi.fn(async () => Promise.reject(new Error('gateway down'))))
    await expect(host.composer.selectModel(null, { model: 'a', provider: 'p' })).resolves.toBe(false)
  })
})
