import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { IpcMainInvokeEvent } from 'electron'
import { afterEach, expect, test } from 'vitest'

import { registerDurableSendIpc } from './durable-send-ipc'
import type { DurableSendIntent } from './durable-send-types'

const homes: string[] = []

function fixture() {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'durable-ipc-'))
  homes.push(home)
  const handlers = new Map<string, (...args: unknown[]) => unknown>()
  const frame = { url: 'file:///fixture/index.html?peer=1#/chat-a' }
  const sender = { mainFrame: frame, getLastWebPreferences: () => ({ preload: '/fixture/preload.js' }) }
  const owner = { isDestroyed: () => false, webContents: sender }
  registerDurableSendIpc({
    ipc: {
      handle: (name, handler) => {
        handlers.set(name, handler as (...args: unknown[]) => unknown)
      }
    },
    userData: () => home,
    rendererUrl: () => 'file:///fixture/index.html',
    getOwner: candidate => (candidate === (sender as unknown) ? (owner as never) : null)
  })
  const event = { sender, senderFrame: frame } as unknown as IpcMainInvokeEvent
  const invoke = (name: string, ...args: unknown[]) => handlers.get(`hermes:durable-send:${name}`)!(event, ...args)

  return { home, frame, sender, event, invoke, handlers }
}

afterEach(() => homes.splice(0).forEach(home => fs.rmSync(home, { recursive: true, force: true })))

test('registered IPC carries actual binary bytes into a queued record, with no path API', () => {
  const f = fixture()
  const bytes = new Uint8Array([0, 1, 255, 10, 42])

  const intent: DurableSendIntent = {
    id: randomUUID(),
    owner: { connectionId: 'fixture', gatewayId: 'fixture', profile: 'alpha', principalId: 'user' },
    target: { sessionId: 'a' },
    text: 'fixture',
    files: [
      {
        id: randomUUID(),
        name: 'binary.dat',
        mediaType: 'application/octet-stream',
        size: bytes.length,
        sha256: createHash('sha256').update(bytes).digest('hex')
      }
    ]
  }

  expect(f.invoke('begin', intent)).toMatchObject({ state: 'staging' })
  expect(f.invoke('append', intent.id, intent.files[0].id, 0, bytes)).toBe(bytes.length)
  expect(f.invoke('commit', intent.id)).toMatchObject({ state: 'queued' })
  expect(f.invoke('list')).toHaveLength(1)
  expect([...f.handlers.keys()]).not.toContain('hermes:durable-send:confirm')
  expect(() => f.invoke('begin', { ...intent, id: randomUUID(), sourcePath: '/etc/passwd' })).toThrow()
})

test.each(['subframe', 'external-navigation', 'other-file', 'unowned-window'])(
  'denies %s before any spool file is created',
  reason => {
    const f = fixture()

    if (reason === 'subframe') {
      Object.defineProperty(f.event, 'senderFrame', { value: { url: f.frame.url } })
    }

    if (reason === 'external-navigation') {
      f.frame.url = 'https://example.invalid/index.html'
    }

    if (reason === 'other-file') {
      f.frame.url = 'file:///private/other.html'
    }

    if (reason === 'unowned-window') {
      f.event.sender = { ...f.sender } as never
    }

    expect(() => f.invoke('list')).toThrow('denied')
    expect(fs.readdirSync(f.home)).toEqual([])
  }
)
