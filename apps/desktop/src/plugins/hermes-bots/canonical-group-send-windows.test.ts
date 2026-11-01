// @vitest-environment node
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

// The real Electron-main journal owner, reached through its real IPC handlers; only ipcMain/app are faked.
const main = vi.hoisted(() => ({ userData: '', handlers: new Map<string, (...args: unknown[]) => unknown>() }))
vi.mock('electron', () => ({
  app: { getPath: () => main.userData },
  ipcMain: { handle: (name: string, handler: (...args: unknown[]) => unknown) => main.handlers.set(name, handler) }
}))
const nativeModule = '../../../electron/prepared-submissions'
const { preparedJournal, registerPreparedSubmissions } = await import(/* @vite-ignore */ nativeModule)
import { prepareCanonicalGroupSend, readCanonicalGroupSend, retireCanonicalGroupSend } from './canonical-group-send'

const binding = { connectionId: 'local', profile: 'default', roomId: 'room-a' }
const origin = 'http://localhost:5174'
afterEach(() => vi.unstubAllGlobals())

/** Every renderer window of the origin reaches the one main-process journal over IPC. Each call
 *  takes an IPC hop each way; `hold` parks one window's next mutation until released. */
function windows() {
  main.userData = fs.mkdtempSync(path.join(os.tmpdir(), 'group-send-windows-'))
  main.handlers.clear()
  registerPreparedSubmissions()
  const event = { sender: {}, senderFrame: { url: `${origin}/index.html` } }
  const hop = () => new Promise(resolve => setImmediate(resolve))
  let held: Promise<void> | undefined

  const ipc = (channel: string) => async (...args: unknown[]) => {
    await hop()
    const handler = main.handlers.get(`hermes:prepared-submissions:${channel}`)!

    if (channel !== 'read' && held) { const gate = held; held = undefined; await gate }
    const result = handler(event, ...args)
    await hop()

    return result
  }

  const preparedSubmissions = { read: ipc('read'), update: ipc('update'), compareAndSet: ipc('compare-and-set') }
  vi.stubGlobal('window', { hermesDesktop: { preparedSubmissions } })

  return {
    durable: () => preparedJournal(main.userData, origin).read() as Record<string, { params: { event_id: string } }>,
    holdNextMutation: () => { let release!: () => void; held = new Promise<void>(r => { release = r });

 return release },
    cleanup: () => fs.rmSync(main.userData, { recursive: true, force: true })
  }
}

test('two windows preparing one room slot at once acknowledge the single durable winner', async () => {
  const rig = windows()

  try {
    const [a, b] = await Promise.all([
      prepareCanonicalGroupSend(binding, { text: 'from window A' }),
      prepareCanonicalGroupSend(binding, { text: 'from window B' })
    ])

    const stored = Object.values(rig.durable())
    expect(stored).toHaveLength(1)
    // Each acknowledged preparation names the durable record, so a reload recovers what was sent.
    expect(a.params.event_id).toBe(stored[0].params.event_id)
    expect(b.params.event_id).toBe(stored[0].params.event_id)
    expect(await readCanonicalGroupSend(binding)).toEqual(a)
  } finally { rig.cleanup() }
})

test('a delayed ACK for a retired intent cannot delete the newer intent another window prepared', async () => {
  const rig = windows()

  try {
    const a = await prepareCanonicalGroupSend(binding, { text: 'first' })
    await retireCanonicalGroupSend(binding, 'wrong-id')
    expect(Object.values(rig.durable()).map(entry => entry.params.event_id)).toEqual([a.params.event_id])
    const release = rig.holdNextMutation()
    const delayed = retireCanonicalGroupSend(binding, a.params.event_id) // window A's ACK, parked at main
    await new Promise(resolve => setTimeout(resolve, 20))
    await retireCanonicalGroupSend(binding, a.params.event_id) // window B's ACK for the same send
    const b = await prepareCanonicalGroupSend(binding, { text: 'second' })
    release()
    await delayed
    expect(Object.values(rig.durable()).map(entry => entry.params.event_id)).toEqual([b.params.event_id])
    expect(await readCanonicalGroupSend(binding)).toEqual(b)
  } finally { rig.cleanup() }
})
