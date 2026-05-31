import { afterEach, describe, expect, it, vi } from 'vitest'

import { isDemoMode } from '../lib/demo-flag'

import { bridge } from './bridge'
import { control, DemoSocket } from './gateway'

afterEach(() => {
  window.localStorage.clear()
  window.location.hash = ''
  control.socket = null
})

describe('isDemoMode', () => {
  it('is off by default', () => {
    expect(isDemoMode()).toBe(false)
  })

  it('enables via the ?demo=1 hash query', () => {
    window.location.hash = '#/?demo=1'
    expect(isDemoMode()).toBe(true)
  })

  it('enables via localStorage', () => {
    window.localStorage.setItem('hermes.demo', '1')
    expect(isDemoMode()).toBe(true)
  })
})

describe('bridge.api (REST router)', () => {
  it('serves the session list from fixtures', async () => {
    const res = await bridge.api<{ sessions: Array<{ id: string }>; total: number }>({ path: '/api/sessions' })
    expect(res.total).toBeGreaterThan(0)
    expect(res.sessions.some(s => s.id === 's-release')).toBe(true)
  })

  it('serves messages for a known session', async () => {
    const res = await bridge.api<{ session_id: string; messages: unknown[] }>({
      path: '/api/sessions/s-release/messages'
    })

    expect(res.session_id).toBe('s-release')
    expect(res.messages.length).toBeGreaterThan(0)
  })

  it('serves config and an empty default for unknown GETs', async () => {
    expect(await bridge.api<{ display: unknown }>({ path: '/api/config' })).toHaveProperty('display')
    expect(await bridge.api({ path: '/api/unknown' })).toEqual({})
  })
})

describe('fake gateway', () => {
  it('answers a JSON-RPC request from the handler map', async () => {
    const socket = new DemoSocket('ws://demo')

    const result = new Promise<unknown>(resolve => {
      socket.onmessage = ev => {
        const frame = JSON.parse(ev.data) as { id?: number; result?: unknown }

        if (frame.id === 7) {
          resolve(frame.result)
        }
      }
    })

    socket.send(JSON.stringify({ id: 7, method: 'setup.runtime_check' }))
    expect(await result).toEqual({ ok: true })
  })

  it('playTurn emits a well-formed streaming sequence', async () => {
    vi.useFakeTimers()

    const events: Array<{ type: string }> = []
    control.socket = {
      deliver: (data: string) => {
        events.push((JSON.parse(data) as { params: { type: string } }).params)
      }
    } as unknown as DemoSocket

    const done = control.playTurn({ reasoning: 'r', tool: { name: 't', tool_id: '1' }, reply: ['ab'] }, 'sid')

    await vi.runAllTimersAsync()
    await done

    const types = events.map(e => e.type)
    expect(types[0]).toBe('message.start')
    expect(types).toContain('reasoning.delta')
    expect(types).toContain('tool.start')
    expect(types).toContain('tool.complete')
    expect(types).toContain('message.delta')
    expect(types.at(-1)).toBe('message.complete')

    vi.useRealTimers()
  })
})
