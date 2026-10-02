import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

const handlers = vi.hoisted(() => new Map<string, (...args: any[]) => any>())
const location = vi.hoisted(() => ({ directory: '' }))
vi.mock('electron', () => ({ app: { getPath: () => location.directory }, ipcMain: { handle: (name: string, callback: (...args: any[]) => any) => handlers.set(name, callback) } }))
import { preparedJournal, registerPreparedSubmissions } from './prepared-submissions'

afterEach(() => {if (location.directory) {fs.rmSync(location.directory, { recursive: true, force: true })}; handlers.clear()})

function fixture() {
  location.directory = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-native-ipc-'))
  registerPreparedSubmissions()
  const event = (sender: object) => ({ sender, senderFrame: { url: 'http://localhost:5174/index.html' } })

  return { event, store: preparedJournal(location.directory, 'http://localhost:5174') }
}

it('gives one WebContents a reload-stable owner and another window an independent owner', () => {
  const { event } = fixture()
  const owner = handlers.get('hermes:prepared-submissions:owner')!
  const first = {}, second = {}

  expect(owner(event(first))).toBe(owner(event(first)))
  expect(owner(event(first))).not.toBe(owner(event(second)))
})

it('native compare-send preserves a newer exact record and accepts large frozen payloads', () => {
  const { event, store } = fixture()
  const compare = handlers.get('hermes:prepared-submissions:compare-send')!
  const sender = event({})
  const first = { text: 'Ω'.repeat(600000), id: 'same-id' }
  expect(compare(sender, 'slot', null, JSON.stringify(first))).toBe(true)
  const newer = { ...first, id: 'newer-id' }
  expect(compare(sender, 'slot', JSON.stringify(first), JSON.stringify(newer))).toBe(true)
  expect(compare(sender, 'slot', JSON.stringify(first), null)).toBe(false)
  expect(store.read().slot).toEqual(newer)
  expect(compare(sender, 'slot', JSON.stringify(newer), null)).toBe(true)
})

it('does not reinterpret unreadable native storage as an empty journal', () => {
  const { event, store } = fixture()
  store.update('slot', { id: 'original' })
  const file = fs.readdirSync(location.directory).find(name => name.endsWith('.json'))!
  fs.writeFileSync(path.join(location.directory, file), '[]')
  const before = fs.readFileSync(path.join(location.directory, file))
  expect(() => handlers.get('hermes:prepared-submissions:compare-send')!(event({}), 'slot', null, '{}')).toThrow('Invalid prepared submission journal')
  expect(fs.readFileSync(path.join(location.directory, file))).toEqual(before)
})
