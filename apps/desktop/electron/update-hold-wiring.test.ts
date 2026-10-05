import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerUpdateHoldIpc, type UpdateHoldWire } from './update-hold-wiring'

const HOLD: UpdateHoldWire = {
  holdId: 'hold-1',
  verdict: 'held',
  ownerPid: 4242,
  since: 1,
  checkedAt: 2,
  logPath: '/logs/desktop.log'
}

function register(state: { phase: string; updateHold: UpdateHoldWire | null }) {
  const handlers = new Map<string, (event: any, ...args: any[]) => Promise<any>>()
  const primary = { id: 'main' }
  let quits = 0

  registerUpdateHoldIpc({ handle: (channel: string, fn: any) => handlers.set(channel, fn) } as any, {
    isPrimaryBootSender: (event: any) => event.sender === primary,
    bootProgress: () => state,
    currentHold: () => null,
    log: () => {},
    flushLog: () => {},
    quit: () => {
      quits += 1
    }
  })

  return { handlers, primary, quits: () => quits }
}

// Review G3: boot-progress pushes (the hold clearing included) reach only the
// main window and Quit/Check again refuse every other sender, so a HUD or
// session window that mounted the blocked screen from its snapshot pull could
// never take it down. Only the primary window's snapshot carries the hold.
test('the boot snapshot carries the update hold to the primary window only', async () => {
  const state = { phase: 'backend.update-hold', updateHold: HOLD }
  const { handlers, primary } = register(state)
  const get = handlers.get('hermes:boot-progress:get')

  assert.ok(get, 'registerUpdateHoldIpc owns the boot snapshot channel')
  assert.deepEqual(await get({ sender: primary }), state)
  assert.deepEqual(await get({ sender: { id: 'hud' } }), { phase: 'backend.update-hold', updateHold: null })
  assert.equal(state.updateHold, HOLD, 'main.ts state is never mutated')
})

test('a non-primary window cannot quit from the update-hold screen', async () => {
  const { handlers, primary, quits } = register({ phase: 'x', updateHold: HOLD })
  const quit = handlers.get('hermes:update-hold:quit')!

  assert.deepEqual(await quit({ sender: { id: 'session' } }), { ok: false })
  assert.equal(quits(), 0)
  assert.deepEqual(await quit({ sender: primary }), { ok: true })
  assert.equal(quits(), 1)
})
