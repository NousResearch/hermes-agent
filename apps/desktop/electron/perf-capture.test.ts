import { describe, expect, it, vi } from 'vitest'

import { captureTraceOnce, MAX_TRACE_MS, MIN_TRACE_MS, traceDurationMs } from './perf-capture'

function tracer(overrides: Partial<{ start: () => Promise<void>; stop: () => Promise<string> }> = {}) {
  return {
    startRecording: vi.fn(overrides.start ?? (async () => {})),
    stopRecording: vi.fn(overrides.stop ?? (async () => '{"traceEvents":[]}'))
  }
}

const quiet = () => {}

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

    const result = await captureTraceOnce({ tracer: t, durationMs: 8000, outFile: '/tmp/t.json', write, sleep: quiet })

    expect(result).toEqual({ ok: true, path: '/tmp/t.json' })
    expect(t.startRecording).toHaveBeenCalledTimes(1)
    expect(t.startRecording.mock.calls[0][0].categories).toContain('cc')
    expect(t.stopRecording).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledWith('/tmp/t.json', '{"traceEvents":[]}')
  })

  it('pulls in the trace file and deletes the tracer copy', async () => {
    const t = tracer({ stop: async () => '/tmp/chrome-trace-out' })
    const write = vi.fn(async () => {})
    const read = vi.fn(async () => '{"traceEvents":[1]}')
    const remove = vi.fn(async () => {})

    const result = await captureTraceOnce({
      tracer: t,
      durationMs: 1000,
      outFile: '/tmp/t.json',
      write,
      read,
      remove,
      sleep: quiet
    })

    expect(result).toEqual({ ok: true, path: '/tmp/t.json' })
    expect(read).toHaveBeenCalledWith('/tmp/chrome-trace-out')
    expect(write).toHaveBeenCalledWith('/tmp/t.json', '{"traceEvents":[1]}')
    // Two 100MB copies per capture is not acceptable: the tracer's copy goes.
    expect(remove).toHaveBeenCalledWith('/tmp/chrome-trace-out')
  })

  it('names the failure when the tracer refuses to start', async () => {
    const t = tracer({
      start: async () => {
        throw new Error('already tracing')
      }
    })

    const write = vi.fn(async () => {})

    const result = await captureTraceOnce({ tracer: t, durationMs: 8000, outFile: '/tmp/t.json', write, sleep: quiet })

    expect(result).toEqual({ ok: false, reason: 'start-refused' })
    expect(write).not.toHaveBeenCalled()
    expect(t.stopRecording).not.toHaveBeenCalled()
  })

  it('reports a stop that throws instead of blaming the tracer for refusing', async () => {
    const t = tracer({
      stop: async () => {
        throw new Error('no trace')
      }
    })

    const write = vi.fn(async () => {})

    const result = await captureTraceOnce({ tracer: t, durationMs: 8000, outFile: '/tmp/t.json', write, sleep: quiet })

    expect(result).toEqual({ ok: false, reason: 'stop-failed' })
    expect(write).not.toHaveBeenCalled()
    // The first stop throws, so the finally guard must not double-stop.
    expect(t.stopRecording).toHaveBeenCalledTimes(1)
  })

  it('bounds a stopRecording that never settles', async () => {
    const t = tracer({ stop: () => new Promise<string>(() => {}) })
    const write = vi.fn(async () => {})

    const result = await captureTraceOnce({
      tracer: t,
      durationMs: 1000,
      outFile: '/tmp/t.json',
      write,
      sleep: quiet,
      stopTimeoutMs: 10
    })

    expect(result).toEqual({ ok: false, reason: 'stop-timeout' })
    expect(write).not.toHaveBeenCalled()
  })

  it('reports an empty trace and a failed copy distinctly', async () => {
    const empty = tracer({ stop: async () => '' })
    const write = vi.fn(async () => {})

    expect(await captureTraceOnce({ tracer: empty, durationMs: 1000, outFile: '/tmp/t.json', write, sleep: quiet })).toEqual({
      ok: false,
      reason: 'empty'
    })

    const unreadable = tracer({ stop: async () => '/tmp/gone' })

    expect(
      await captureTraceOnce({
        tracer: unreadable,
        durationMs: 1000,
        outFile: '/tmp/t.json',
        write,
        read: async () => {
          throw new Error('ENOENT')
        },
        sleep: quiet
      })
    ).toEqual({ ok: false, reason: 'read-failed' })

    const fullDisk = tracer()

    const failingWrite = vi.fn(async () => {
      throw new Error('ENOSPC')
    })

    expect(
      await captureTraceOnce({ tracer: fullDisk, durationMs: 1000, outFile: '/tmp/t.json', write: failingWrite, sleep: quiet })
    ).toEqual({ ok: false, reason: 'write-failed' })
  })
})
