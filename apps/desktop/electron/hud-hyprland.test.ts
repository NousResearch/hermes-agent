/**
 * Unit tests for the Hyprland HUD overlay promote. The socket itself needs a
 * live compositor; what is covered here is the grammar (classic vs Lua), the
 * client lookup, and which dispatchers fire for a tiled vs already-floating bar.
 */

import assert from 'node:assert/strict'

import { afterEach, test } from 'vitest'

import {
  classifyHyprlandDispatchReply,
  hudOverlayCommands,
  parseHudHyprlandClient,
  promoteHudOnHyprland,
  resetHyprlandDispatchSyntax
} from './hud-hyprland'

const TITLE = 'Hermes HUD'
const ADDRESS = '0x55d1'
const SOCKET_ENV = { HYPRLAND_INSTANCE_SIGNATURE: 'abc123', XDG_RUNTIME_DIR: '/run/user/1000' }

const LUA_REJECT =
  "error: [string \"return hl.dispatch(setfloating address:0x55d1)\"]:1: ')' expected near 'address'\n\n → Note: dispatch in lua is a shorthand for hl.dispatch(...), your syntax might need to be updated."

afterEach(() => {
  resetHyprlandDispatchSyntax()
})

const hudClient = (over: Record<string, unknown> = {}) => ({
  address: ADDRESS,
  floating: false,
  pinned: false,
  title: TITLE,
  ...over
})

test('classifies Hyprland dispatch replies the way 0.54 and 0.56 actually write them', () => {
  assert.equal(classifyHyprlandDispatchReply('ok'), 'ok')
  assert.equal(classifyHyprlandDispatchReply('ok\n'), 'ok')
  assert.equal(classifyHyprlandDispatchReply('Invalid dispatcher'), 'wrong-syntax')
  assert.equal(classifyHyprlandDispatchReply(LUA_REJECT), 'wrong-syntax')
  assert.equal(classifyHyprlandDispatchReply('No such window found'), 'failed')
  assert.equal(classifyHyprlandDispatchReply('warning: =[C]:-1: hl.focus: window not found'), 'failed')
})

test('finds the HUD by exact title among other windows of the same pid', () => {
  const found = parseHudHyprlandClient(
    JSON.stringify([
      { address: '0x111', pid: 42, title: 'Hermes', floating: true, pinned: false },
      hudClient({ pid: 42 })
    ]),
    TITLE
  )

  assert.deepEqual(found, { address: ADDRESS, floating: false, pinned: false })
})

test('prefixes a bare hex address so the selector is well-formed', () => {
  const found = parseHudHyprlandClient(JSON.stringify([hudClient({ address: '55d1' })]), TITLE)

  assert.equal(found?.address, ADDRESS)
})

test('ignores a payload that is not the client list or has the wrong title', () => {
  assert.equal(parseHudHyprlandClient('not json', TITLE), null)
  assert.equal(parseHudHyprlandClient('{}', TITLE), null)
  assert.equal(parseHudHyprlandClient(JSON.stringify([hudClient({ title: 'other' })]), TITLE), null)
})

test('legacy float is setfloating (idempotent), not togglefloating', () => {
  const commands = hudOverlayCommands(ADDRESS, 'float')

  assert.equal(commands.legacy, 'dispatch setfloating address:0x55d1')
  assert.equal(commands.lua, 'dispatch hl.dsp.window.float({ action = "enable", window = "address:0x55d1" })')
})

test('lua pin uses enable so a second promote does not unpin', () => {
  const commands = hudOverlayCommands(ADDRESS, 'pin')

  assert.equal(commands.legacy, 'dispatch pin address:0x55d1')
  assert.equal(commands.lua, 'dispatch hl.dsp.window.pin({ action = "enable", window = "address:0x55d1" })')
})

test('skips compositors that are not Hyprland', async () => {
  const calls: string[] = []

  const promoted = await promoteHudOnHyprland({
    title: TITLE,
    env: {},
    uid: 1000,
    request: async (_socket, command) => {
      calls.push(command)

      return 'ok'
    }
  })

  assert.equal(promoted, false)
  assert.deepEqual(calls, [])
})

test('floats then pins a tiled HUD on a classic hyprlang session', async () => {
  const calls: string[] = []

  const promoted = await promoteHudOnHyprland({
    title: TITLE,
    env: SOCKET_ENV,
    uid: 1000,
    attempts: 1,
    delayMs: 0,
    request: async (_socket, command) => {
      calls.push(command)

      if (command === 'j/clients') {
        return JSON.stringify([hudClient()])
      }

      return 'ok'
    }
  })

  assert.equal(promoted, true)
  assert.deepEqual(calls, ['j/clients', 'dispatch setfloating address:0x55d1', 'dispatch pin address:0x55d1'])
})

test('falls back to Lua dispatch when the classic session grammar is rejected', async () => {
  const calls: string[] = []

  const promoted = await promoteHudOnHyprland({
    title: TITLE,
    env: SOCKET_ENV,
    uid: 1000,
    attempts: 1,
    delayMs: 0,
    request: async (_socket, command) => {
      calls.push(command)

      if (command === 'j/clients') {
        return JSON.stringify([hudClient()])
      }

      if (command.startsWith('dispatch setfloating') || command.startsWith('dispatch pin ')) {
        return LUA_REJECT
      }

      return 'ok'
    }
  })

  assert.equal(promoted, true)
  assert.deepEqual(calls, [
    'j/clients',
    'dispatch setfloating address:0x55d1',
    'dispatch hl.dsp.window.float({ action = "enable", window = "address:0x55d1" })',
    'dispatch hl.dsp.window.pin({ action = "enable", window = "address:0x55d1" })'
  ])
})

test('caches Lua syntax so the next promote does not probe classic first', async () => {
  const request = async (_socket: string, command: string) => {
    if (command === 'j/clients') {
      return JSON.stringify([hudClient()])
    }

    if (command.startsWith('dispatch setfloating') || command.startsWith('dispatch pin ')) {
      return LUA_REJECT
    }

    return 'ok'
  }

  await promoteHudOnHyprland({
    title: TITLE,
    env: SOCKET_ENV,
    uid: 1000,
    attempts: 1,
    delayMs: 0,
    request
  })

  const calls: string[] = []

  await promoteHudOnHyprland({
    title: TITLE,
    env: SOCKET_ENV,
    uid: 1000,
    attempts: 1,
    delayMs: 0,
    request: async (_socket, command) => {
      calls.push(command)

      return request(_socket, command)
    }
  })

  assert.deepEqual(calls, [
    'j/clients',
    'dispatch hl.dsp.window.float({ action = "enable", window = "address:0x55d1" })',
    'dispatch hl.dsp.window.pin({ action = "enable", window = "address:0x55d1" })'
  ])
})

test('does not pin a window that is already pinned, and does not float one that already is', async () => {
  const calls: string[] = []

  const promoted = await promoteHudOnHyprland({
    title: TITLE,
    env: SOCKET_ENV,
    uid: 1000,
    attempts: 1,
    delayMs: 0,
    request: async (_socket, command) => {
      calls.push(command)

      if (command === 'j/clients') {
        return JSON.stringify([hudClient({ floating: true, pinned: true })])
      }

      return 'ok'
    }
  })

  assert.equal(promoted, true)
  assert.deepEqual(calls, ['j/clients'])
})

test('retries until Hyprland lists the HUD', async () => {
  let lookups = 0

  const promoted = await promoteHudOnHyprland({
    title: TITLE,
    env: SOCKET_ENV,
    uid: 1000,
    attempts: 3,
    delayMs: 0,
    sleep: async () => undefined,
    request: async (_socket, command) => {
      if (command === 'j/clients') {
        lookups += 1

        if (lookups < 3) {
          return JSON.stringify([{ address: '0x111', title: 'Hermes', floating: true }])
        }

        return JSON.stringify([hudClient({ floating: true, pinned: true })])
      }

      return 'ok'
    }
  })

  assert.equal(promoted, true)
  assert.equal(lookups, 3)
})
