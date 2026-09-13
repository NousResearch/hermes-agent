import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

const native = vi.hoisted(() => ({ home: '', handlers: new Map<string, (...args: any[]) => any>() }))
vi.mock('electron', () => ({ app: { getPath: () => native.home }, ipcMain: {
  handle: (name: string, handler: (...args: any[]) => any) => native.handlers.set(name, handler)
} }))
import { preparedJournal, registerPreparedSubmissions } from './prepared-submissions'

test('real shared store IPC separates window owners and conditionally claims/retires exact entries', () => {
  native.home = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-ipc-'))
  registerPreparedSubmissions()
  const first = { sender: {}, senderFrame: { url: 'http://same-origin/chat' } }
  const second = { sender: {}, senderFrame: { url: 'http://same-origin/chat' } }
  const call = (name: string, ...args: unknown[]) => native.handlers.get(`hermes:prepared-submissions:${name}`)!(...args)

  try {
    const ownerA = call('owner', first)
    expect(call('owner', { ...first, senderFrame: { url: 'http://same-origin/reloaded' } })).toBe(ownerA)
    const ownerB = call('owner', second)
    expect(ownerB).not.toBe(ownerA)
    const a = JSON.stringify({ id: 'a', owner: ownerA })
    const b = JSON.stringify({ id: 'b', owner: ownerB })
    expect(call('compare-and-set', first, 'intent-a', null, a)).toBe(true)
    expect(call('compare-and-set', second, 'intent-b', null, b)).toBe(true)
    expect(call('compare-and-set', second, 'intent-a', null, b)).toBe(false)
    const transferred = JSON.stringify({ id: 'a', owner: ownerB })
    expect(call('compare-and-set', second, 'intent-a', a, transferred)).toBe(true)
    expect(call('compare-and-set', first, 'intent-a', a, null)).toBe(false)
    expect(call('compare-and-set', second, 'intent-a', transferred, null)).toBe(true)
    expect(preparedJournal(native.home, 'http://same-origin').read()).toEqual({ 'intent-b': JSON.parse(b) })
    expect(preparedJournal(native.home, 'http://other-origin').read()).toEqual({})
  } finally {fs.rmSync(native.home, { recursive: true, force: true })}
})

test('Group CAS keeps its caps while ordinary CAS preserves large legacy records and exact comparisons', () => {
  native.home = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-purpose-'))
  registerPreparedSubmissions()
  const event = { sender: {}, senderFrame: { url: 'http://same-origin/chat' } }
  const call = (name: string, ...args: unknown[]) => native.handlers.get(`hermes:prepared-submissions:${name}`)!(event, ...args)
  const large = JSON.stringify({ id: 'legacy-id', text: 'x'.repeat(1024 * 1024), params: { session_id: 'original' } })
  const longKey = JSON.stringify(['original-owner', 'original-session', 'legacy text '.repeat(512)])

  try {
    expect(() => call('compare-and-set', 'group', null, large)).toThrow('Invalid prepared submission comparison')
    expect(() => call('compare-and-set', longKey, null, '{}')).toThrow('Invalid prepared submission comparison')
    expect(call('compare-and-set', 'group', null, '{}')).toBe(true)
    expect(call('compare-send', longKey, null, large)).toBe(true)
    expect(call('compare-send', longKey, '{}', null)).toBe(false)
    expect(JSON.stringify(preparedJournal(native.home, 'http://same-origin').read()[longKey])).toBe(large)
    expect(call('compare-send', longKey, large, null)).toBe(true)
    expect(preparedJournal(native.home, 'http://same-origin').read()).toEqual({ group: {} })
  } finally {fs.rmSync(native.home, { recursive: true, force: true })}
})
