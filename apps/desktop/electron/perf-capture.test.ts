import { describe, expect, it, vi } from 'vitest'

import { captureTraceOnce, MAX_TRACE_MS, MIN_TRACE_MS, traceDurationMs } from './perf-capture'

function tracer(overrides: Partial<{ start: () => Promise<void>; stop: () => Promise<string> }> = {}) {
  return {
    startRecording: vi.fn(overrides.start ?? (async () => {})),
    stopRecording: vi.fn(overrides.stop ?? (async () => '{"traceEvents":[]}'))
  }
}

describe('traceDurationMs', () => {
  it('ignores an unset or nonsense window', () => {
    expect(traceDurationMs(undefined)).toBeNull()
    expect(traceDurationMs('')).toBeNull()
    expect(traceDurationMs('soon')).toBeNull()
    expect(traceDurationMs('0')).toBeNull()
    expect(traceDurationMs('-5')).toBeNull()
  })

  it('clamps into a window a trace can actually hold', () => {
    expect(traceDurationMs('8000')).toBe(8000)
    expect(traceDurationMs('10')).toBe(MIN_TRACE_MS)
    expect(traceDurationMs('999999')).toBe(MAX_TRACE_MS)
  })
})

describe('captureTraceOnce', () => {
  it('records, stops and writes the trace', async () => {
    const t = tracer()
    const write = vi.fn(async () => {})

    const written = await captureTraceOnce({ tracer: t, durationMs: 8000, outFile: '/tmp/t.json', write, sleep: async () => {} })

    expect(written).toBe('/tmp/t.json')
    expect(t.startRecording).toHaveBeenCalledTimes(1)
    expect(t.startRecording.mock.calls[0][0].categories).toContain('cc')
    expect(t.stopRecording).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledWith('/tmp/t.json', '{"traceEvents":[]}')
  })

  it('pulls in the trace file when the tracer answers with a path', async () => {
    const t = tracer({ stop: async () => '/tmp/chrome-trace-out' })
    const write = vi.fn(async () => {})
    const read = vi.fn(async () => '{"traceEvents":[1]}')

    const written = await captureTraceOnce({
      tracer: t,
      durationMs: 1000,
      outFile: '/tmp/t.json',
      write,
      read,
      sleep: async () => {}
    })

    expect(written).toBe('/tmp/t.json')
    expect(read).toHaveBeenCalledWith('/tmp/chrome-trace-out')
    expect(write).toHaveBeenCalledWith('/tmp/t.json', '{"traceEvents":[1]}')
  })

  it('writes nothing and stops the tracer when recording is refused', async () => {
    const t = tracer({
      start: async () => {
        throw new Error('already tracing')
      }
    })

    const write = vi.fn(async () => {})

    const written = await captureTraceOnce({ tracer: t, durationMs: 8000, outFile: '/tmp/t.json', write, sleep: async () => {} })

    expect(written).toBeNull()
    expect(write).not.toHaveBeenCalled()
    expect(t.stopRecording).not.toHaveBeenCalled()
  })

  it('stops the tracer when the capture itself fails mid-flight', async () => {
    const t = tracer({
      stop: async () => {
        throw new Error('no trace')
      }
    })

    const write = vi.fn(async () => {})

    const written = await captureTraceOnce({ tracer: t, durationMs: 8000, outFile: '/tmp/t.json', write, sleep: async () => {} })

    expect(written).toBeNull()
    expect(write).not.toHaveBeenCalled()
    // The first stop throws, so the finally block's guard must not double-stop.
    expect(t.stopRecording).toHaveBeenCalledTimes(1)
  })
})
