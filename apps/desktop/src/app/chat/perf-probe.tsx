import { Profiler, type ProfilerOnRenderCallback, type ReactNode } from 'react'

import { $messages, setBusy, setMessages } from '@/store/session'

type Sample = {
  id: string
  phase: string
  actualDuration: number
  baseDuration: number
  startTime: number
  commitTime: number
}

type SyntheticDriverHandle = { stop: () => void }

declare global {
  interface Window {
    __PERF_PROBE__?: {
      samples: Sample[]
      enabled: boolean
      clear: () => void
      summary: () => Record<string, { count: number; total: number; max: number; p50: number; p95: number }>
    }
    __PERF_DRIVE__?: {
      /** Inject an assistant message and grow it by `chunk` every `intervalMs`. Returns a stop handle. */
      stream: (opts?: { chunk?: string; intervalMs?: number; totalTokens?: number }) => SyntheticDriverHandle
      /**
       * Replace the transcript with `turns` synthetic user/assistant pairs of
       * realistic mixed markdown, then resolve with the ms elapsed from the
       * `setMessages` commit to the second animation frame (a mount+paint
       * proxy). Used by the `transcript` perf scenario. `reset()` restores.
       */
      loadTranscript: (turns?: number) => Promise<number>
      reset: () => void
      snapshotMsgs: () => number
    }
  }
}

if (typeof window !== 'undefined' && !window.__PERF_PROBE__) {
  const samples: Sample[] = []
  window.__PERF_PROBE__ = {
    samples,
    enabled: false,
    clear: () => {
      samples.length = 0
    },
    summary: () => {
      const byId = new Map<string, number[]>()

      for (const s of samples) {
        const k = `${s.id}:${s.phase}`
        const arr = byId.get(k) ?? []
        arr.push(s.actualDuration)
        byId.set(k, arr)
      }

      const out: Record<string, { count: number; total: number; max: number; p50: number; p95: number }> = {}

      for (const [k, arr] of byId) {
        arr.sort((a, b) => a - b)
        const total = arr.reduce((a, b) => a + b, 0)
        out[k] = {
          count: arr.length,
          total: Math.round(total * 100) / 100,
          max: Math.round(arr[arr.length - 1] * 100) / 100,
          p50: Math.round(arr[Math.floor(arr.length * 0.5)] * 100) / 100,
          p95: Math.round(arr[Math.floor(arr.length * 0.95)] * 100) / 100
        }
      }

      return out
    }
  }
}

const onRender: ProfilerOnRenderCallback = (id, phase, actualDuration, baseDuration, startTime, commitTime) => {
  const probe = typeof window !== 'undefined' ? window.__PERF_PROBE__ : undefined

  if (!probe || !probe.enabled) {
    return
  }

  probe.samples.push({ id, phase, actualDuration, baseDuration, startTime, commitTime })

  if (probe.samples.length > 5000) {
    probe.samples.splice(0, probe.samples.length - 5000)
  }
}

if (typeof window !== 'undefined' && !window.__PERF_DRIVE__) {
  // Synthetic stream driver — pushes tokens through the live $messages atom so the
  // assistant-ui runtime + react tree sees them exactly as a real LLM stream would.
  // Driven by the perf harness (scripts/perf/) when no live LLM credit is available.
  let baseline: ReturnType<typeof $messages.get> | null = null
  let activeHandle: SyntheticDriverHandle | null = null

  const stop = () => {
    activeHandle = null
    setBusy(false)
  }

  // One synthetic turn's worth of mixed markdown — prose, a list, a fenced
  // code block, inline code, a link, and a short table — so a loaded transcript
  // exercises the same render cost (Streamdown blocks, code cards) a real one
  // would. Kept deterministic (seeded by index) so runs are comparable.
  const syntheticTurn = (i: number): ReturnType<typeof $messages.get> => {
    const user = {
      id: `perf-u-${i}`,
      role: 'user' as const,
      parts: [
        { type: 'text' as const, text: `Question ${i}: how does the widget in module ${i} handle back-pressure?` }
      ],
      timestamp: Date.now()
    }

    const assistant = {
      id: `perf-a-${i}`,
      role: 'assistant' as const,
      parts: [
        {
          type: 'text' as const,
          text: [
            `## Answer ${i}`,
            '',
            `The widget buffers writes and applies a bounded queue. Key points for module \`${i}\`:`,
            '',
            '- It coalesces bursts into a single flush.',
            '- Back-pressure propagates via a `Promise` that resolves on drain.',
            '- See [the design note](https://example.com/design) for the state machine.',
            '',
            '```ts',
            `function flush${i}(items: number[]) {`,
            '  return items.reduce((a, b) => a + b, 0)',
            '}',
            '```',
            '',
            '| stage | cost |',
            '|---|---|',
            '| enqueue | O(1) |',
            '| flush | O(n) |',
            ''
          ].join('\n')
        }
      ],
      timestamp: Date.now(),
      pending: false
    }

    return [user, assistant]
  }

  window.__PERF_DRIVE__ = {
    snapshotMsgs: () => $messages.get().length,
    loadTranscript: (turns = 200) => {
      if (!baseline) {
        baseline = $messages.get()
      }

      const next: ReturnType<typeof $messages.get> = []

      for (let i = 0; i < turns; i += 1) {
        next.push(...syntheticTurn(i))
      }

      const t0 = performance.now()
      setMessages(next)

      return new Promise<number>(resolve => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            resolve(performance.now() - t0)
          })
        })
      })
    },
    reset: () => {
      activeHandle?.stop()

      if (baseline) {
        setMessages(baseline)
      }

      baseline = null
      setBusy(false)
    },
    stream: ({
      chunk = 'word ',
      intervalMs = 16,
      totalTokens = 400,
      // Mimic `use-message-stream.scheduleDeltaFlush` — batch token deltas
      // into at-most one $messages update every `flushMinMs` ms, exactly as
      // the real gateway path does. With this on, the synthetic harness's
      // numbers actually reflect what a real LLM stream of the same token
      // rate would feel like. Set to 0 to bypass and apply every token
      // immediately (worst-case).
      flushMinMs = 0
    }: { chunk?: string; intervalMs?: number; totalTokens?: number; flushMinMs?: number } = {}) => {
      activeHandle?.stop()
      const current = $messages.get()

      if (!baseline) {
        baseline = current
      }

      const msgId = `synthetic-${Date.now()}`
      // Seed an empty assistant message — assistant-ui will see it grow.
      setMessages([
        ...current,
        {
          id: msgId,
          role: 'assistant',
          parts: [{ type: 'text', text: '' }],
          timestamp: Date.now(),
          pending: true
        }
      ])
      setBusy(true)

      let pushed = 0
      let pendingDelta = ''
      let lastFlushAt = 0
      let timer: ReturnType<typeof setTimeout> | null = null
      let flushHandle: number | null = null

      const applyDelta = (delta: string) => {
        if (!delta) {
          return
        }

        setMessages(prev =>
          prev.map(m => {
            if (m.id !== msgId) {
              return m
            }

            const head = m.parts.slice(0, -1)
            const last = m.parts.at(-1)
            const lastText = last && last.type === 'text' ? last.text : ''

            return {
              ...m,
              parts: [...head, { type: 'text', text: lastText + delta }]
            }
          })
        )
      }

      const flushNow = () => {
        flushHandle = null
        lastFlushAt = performance.now()
        const delta = pendingDelta
        pendingDelta = ''
        applyDelta(delta)
      }

      const scheduleFlush = () => {
        if (flushHandle !== null) {
          return
        }

        if (flushMinMs <= 0) {
          flushNow()

          return
        }

        const since = performance.now() - lastFlushAt
        const wait = Math.max(0, flushMinMs - since)
        flushHandle =
          wait <= 0 && typeof requestAnimationFrame === 'function'
            ? requestAnimationFrame(flushNow)
            : (setTimeout(flushNow, wait) as unknown as number)
      }

      const handle: SyntheticDriverHandle = {
        stop: () => {
          if (timer) {
            clearTimeout(timer)
          }

          timer = null

          if (flushHandle !== null) {
            clearTimeout(flushHandle)
            cancelAnimationFrame?.(flushHandle)
          }

          flushHandle = null

          if (pendingDelta) {
            applyDelta(pendingDelta)
            pendingDelta = ''
          }

          activeHandle = null
          // Mark message finalized.
          setMessages(prev => prev.map(m => (m.id === msgId ? { ...m, pending: false } : m)))
          setBusy(false)
        }
      }

      activeHandle = handle

      const tick = () => {
        if (activeHandle !== handle) {
          return
        }

        if (pushed >= totalTokens) {
          if (pendingDelta) {
            flushNow()
          }

          handle.stop()

          return
        }

        pushed += 1

        if (flushMinMs > 0) {
          pendingDelta += chunk
          scheduleFlush()
        } else {
          applyDelta(chunk)
        }

        timer = setTimeout(tick, intervalMs)
      }

      timer = setTimeout(tick, intervalMs)

      return handle
    }
  }

  // Suppress dead-import warning.
  void stop
}

export function PerfProbe({ id, children }: { id: string; children: ReactNode }) {
  return (
    <Profiler id={id} onRender={onRender}>
      {children}
    </Profiler>
  )
}
