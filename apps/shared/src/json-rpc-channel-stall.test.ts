import { describe, expect, it, vi } from 'vitest'

import { JsonRpcRequestChannel, type JsonRpcTransport } from './json-rpc-channel.js'

const spyTransport = () => {
  const sent: string[] = []
  const transport: JsonRpcTransport = { send: text => void sent.push(text) }

  return { sent, transport }
}

/**
 * Renderer-stall skew vs the heartbeat deadline (#69930):
 *
 * The desktop renderer keeps its gateway WebSocket alive with a 15s
 * `gateway.ping` / 45s deadline heartbeat on the JS event loop. A renderer
 * main-thread stall (heavy render, GC, a large session list) delays BOTH the
 * ping sends and the deadline check itself. When the loop finally resumes,
 * the first timer tick compares wall-clock `Date.now()` against
 * `lastLivenessAt` and declares the transport dead — even though the
 * backend was alive and streaming the whole time and the stall, not the
 * socket, produced the silence. The client then force-closes a healthy
 * socket (clean 1001), reconnects, and repeats: reporters saw 572
 * client-initiated closes/day with lifetimes of 35–145s.
 *
 * The fix: a tick that observes a gap far past the heartbeat cadence proves
 * the HOST loop stalled between ticks — silence during the stall is not
 * transport silence, so the deadline window restarts at the observed tick
 * instead of failing the transport on resume. A genuinely dead transport
 * still fails one deadline after the loop resumes.
 *
 * A real stall delays the timer callback itself (the clock runs while the
 * loop is blocked), so these tests simulate it by advancing the mock clock
 * WITHOUT letting the interval fire — `vi.advanceTimersByTime` runs the
 * overdue callback at the end, with `Date.now()` already past the gap.
 */
describe('heartbeat deadline under renderer-stall skew (#69930)', () => {
  it('does not fail a healthy transport when the event loop stalled between pings', async () => {
    vi.useFakeTimers()

    try {
      const failures: string[] = []

      const channel = new JsonRpcRequestChannel({
        heartbeatDeadlineMs: 45_000,
        heartbeatIntervalMs: 15_000,
        heartbeatLiveness: 'any-inbound',
        onHeartbeatFailure: error => void failures.push(error.message)
      })

      const { sent, transport } = spyTransport()

      channel.attach(transport)
      channel.startHeartbeat()

      // Healthy window: ping + inbound frame each cycle.
      for (let cycle = 0; cycle < 3; cycle += 1) {
        await vi.advanceTimersByTimeAsync(15_000)
        channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'status.update' } }))
      }

      expect(failures).toEqual([])

      // The renderer stalls: the wall clock keeps running but the blocked
      // loop misses every tick (simulated with setSystemTime — fake timers
      // otherwise play missed intervals back on their original schedule,
      // which a real stall never does).
      vi.setSystemTime(Date.now() + 150_000)
      await vi.advanceTimersByTimeAsync(15_000)

      // The stall itself is not transport silence: the backend streamed a
      // frame during it, which the loop delivers on resume.
      channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'status.update' } }))

      expect(failures).toEqual([])

      // The heartbeat is still armed: further healthy cycles ping normally.
      const pingsBefore = sent.filter(frame => frame.includes('gateway.ping')).length
      await vi.advanceTimersByTimeAsync(15_000)
      expect(sent.filter(frame => frame.includes('gateway.ping')).length).toBeGreaterThan(pingsBefore)
    } finally {
      vi.useRealTimers()
    }
  })

  it('still fails a genuinely silent transport once the post-stall deadline expires', async () => {
    vi.useFakeTimers()

    try {
      const failures: string[] = []

      const channel = new JsonRpcRequestChannel({
        heartbeatDeadlineMs: 45_000,
        heartbeatIntervalMs: 15_000,
        heartbeatLiveness: 'any-inbound',
        onHeartbeatFailure: error => void failures.push(error.message)
      })

      const { sent, transport } = spyTransport()

      channel.attach(transport)
      channel.startHeartbeat()

      for (let cycle = 0; cycle < 2; cycle += 1) {
        await vi.advanceTimersByTimeAsync(15_000)
        channel.handleFrame(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'status.update' } }))
      }

      expect(failures).toEqual([])

      // A renderer stall AND a dead backend: nothing arrives after resume.
      vi.setSystemTime(Date.now() + 150_000)
      await vi.advanceTimersByTimeAsync(15_000)

      // The stall was forgiven; the post-stall window is not. With no
      // inbound frame after the resume, the deadline expires and the
      // transport fails.
      await vi.advanceTimersByTimeAsync(60_000)

      expect(failures).toHaveLength(1)

      // Failure stops the timer: no further pings after the report.
      const pings = sent.length
      await vi.advanceTimersByTimeAsync(45_000)
      expect(sent.length).toBe(pings)
    } finally {
      vi.useRealTimers()
    }
  })
})
