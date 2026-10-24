import { EventEmitter } from 'node:events'

import type { BrowserWindow, IpcMain, IpcMainInvokeEvent } from 'electron'
import { describe, expect, it, vi } from 'vitest'

import type { PlayerSnapshot } from '../src/lib/player-contract'

import { parsePlayerAgentRequest, parsePlayerCommand, registerPlayerIpc } from './player-ipc'

describe('player IPC ownership boundary', () => {
  it('admits only the registered chat top frame at its exact renderer page', async () => {
    const handlers = new Map<string, (...args: unknown[]) => unknown>()

    const ipc = {
      handle: (name: string, handler: (...args: unknown[]) => unknown) => handlers.set(name, handler),
      removeHandler: (name: string) => handlers.delete(name)
    }

    const frame = { url: 'file:///app/dist/index.html?win=chat#/' }
    const contents = { id: 7, mainFrame: frame, isDestroyed: () => false, getZoomFactor: () => 1.5, send: vi.fn() }
    const window = Object.assign(new EventEmitter(), { webContents: contents, isDestroyed: () => false })

    const host = {
      request: vi.fn(async () => ({ status: 'closed' }) as PlayerSnapshot),
      linkAwareness: vi.fn(async (_owner, scope) => ({ linked: true, linkId: 'opaque-link', scope })),
      unlinkAwareness: vi.fn(async () => ({ linked: false, linkId: null, scope: null })),
      agentRequest: vi.fn(async (_owner, request) => ({ operation: request.operation.type })),
      setupMediaHelpers: vi.fn(async () => ({ available: false, reason: 'media-helpers-not-installed' })),
      revealMediaJob: vi.fn(async () => false),
      clearLocalData: vi.fn(async () => ({ status: 'closed' }) as PlayerSnapshot),
      subscribe: vi.fn(() => vi.fn()),
      ownerClosed: vi.fn()
    }

    const bridge = registerPlayerIpc(ipc as unknown as IpcMain, host)
    bridge.registerOwner(window as unknown as BrowserWindow, 'file:///app/dist/index.html')
    const request = handlers.get('hermes:player:request')!
    const event = { sender: contents, senderFrame: frame } as unknown as IpcMainInvokeEvent

    await request(event, { type: 'open', source: 'tv' })
    expect(host.request).toHaveBeenCalledWith(window, { type: 'open', source: 'tv' })
    await request(event, {
      type: 'bounds',
      bounds: { x: 20, y: 40, width: 480, height: 300 },
      visible: true,
      revision: 1
    })
    expect(host.request).toHaveBeenLastCalledWith(window, {
      type: 'bounds',
      bounds: { x: 30, y: 60, width: 720, height: 450 },
      visible: true,
      revision: 1
    })
    const scope = { connectionId: 'local', profileId: 'default', sessionId: 'session-1' }
    await handlers.get('hermes:player:link-awareness')!(event, scope)
    expect(host.linkAwareness).toHaveBeenCalledWith(window, scope)
    await handlers.get('hermes:player:agent-request')!(event, { scope, operation: { type: 'context' } })
    expect(host.agentRequest).toHaveBeenCalledWith(window, { scope, operation: { type: 'context' } })
    expect(() =>
      handlers.get('hermes:player:agent-request')!(event, {
        scope,
        operation: { type: 'context', clientWindowId: 7 }
      })
    ).toThrow('Invalid player agent request')
    expect(() => request({ ...event, senderFrame: { url: frame.url } }, { type: 'close' })).toThrow()
    expect(() => request({ ...event, sender: { ...contents, id: 8 } }, { type: 'close' })).toThrow()
    frame.url = 'file:///other/index.html'
    expect(() => request(event, { type: 'close' })).toThrow()
    frame.url = 'https://tv.youtube.com/'
    expect(() => request(event, { type: 'close' })).toThrow()
    expect(host.request).toHaveBeenCalledTimes(2)

    window.emit('closed')
    expect(host.ownerClosed).toHaveBeenCalledWith(window)
    expect(() => request(event, { type: 'status' })).toThrow()
    bridge.dispose()
    expect(handlers.size).toBe(0)
  })

  it('rejects arbitrary navigation and malformed geometry before they reach the manager', () => {
    expect(parsePlayerCommand({ type: 'open', source: 'youtube' })).toEqual({ type: 'open', source: 'youtube' })
    expect(parsePlayerCommand({ type: 'open', source: 'netflix' })).toEqual({ type: 'open', source: 'netflix' })
    const bounds = { x: 20, y: 70, width: 480, height: 300 }
    expect(parsePlayerCommand({ type: 'bounds', bounds, visible: true, revision: 3 })).toMatchObject({ bounds })

    for (const command of [
      { type: 'open', source: 'fixture' },
      { type: 'open', source: 'youtube-tv' },
      { type: 'open', source: ['youtube'] },
      { type: 'open', source: 'tv', url: 'https://example.com' },
      { type: 'evaluate', script: 'process.env' },
      { type: 'bounds', bounds: { ...bounds, width: Infinity }, visible: true, revision: 3 },
      { type: 'bounds', bounds, visible: true, revision: -1 },
      { type: 'pin', pinned: 'yes' }
    ]) {
      expect(() => parsePlayerCommand(command)).toThrow('Invalid player request')
    }
  })

  it('accepts only closed agent operations scoped to a concrete chat and native context token', () => {
    const scope = { connectionId: 'local', profileId: 'default', sessionId: 'session-123' }
    expect(parsePlayerAgentRequest({ scope, operation: { type: 'context' } })).toEqual({
      scope,
      operation: { type: 'context' }
    })
    expect(
      parsePlayerAgentRequest({
        scope,
        operation: {
          type: 'control',
          contextToken: 'x'.repeat(32),
          navigationGeneration: 2,
          action: 'pause'
        }
      }).operation.type
    ).toBe('control')

    for (const value of [
      { scope: { ...scope, connectionId: null }, operation: { type: 'context' } },
      { scope, operation: { type: 'context', url: 'https://youtube.com/watch?v=abc' } },
      {
        scope,
        operation: { type: 'control', contextToken: 'caller-nominated', navigationGeneration: 2, action: 'play' }
      },
      {
        scope,
        operation: { type: 'control', contextToken: 'x'.repeat(32), navigationGeneration: 2, action: 'evaluate' }
      },
      {
        scope,
        operation: { type: 'download.start', contextToken: 'x'.repeat(32), url: 'https://example.com/video.mp4' }
      },
      { scope, operation: { type: 'job.cancel', jobId: '../secret' } }
    ]) {
      expect(() => parsePlayerAgentRequest(value)).toThrow('Invalid player agent request')
    }
  })
})
