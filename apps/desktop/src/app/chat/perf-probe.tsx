import { Profiler, type ProfilerOnRenderCallback, type ReactNode } from 'react'

import { $terminalTakeover, setTerminalTakeover } from '@/app/right-sidebar/store'
import { writeAgentTerminalChunk } from '@/app/right-sidebar/terminal/agent-terminal-stream'
import {
  $activeTerminalId,
  $terminals,
  createTerminal,
  ensureAgentTerminal,
  selectTerminal,
  type TerminalEntry
} from '@/app/right-sidebar/terminal/terminals'
import { $repoStatusByCwd } from '@/store/coding-status'
import { $gateway } from '@/store/gateway'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $currentCwd,
  $messages,
  $selectedStoredSessionId,
  $sessions,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setCurrentCwdTransient,
  setMessages,
  setSelectedStoredSessionId
} from '@/store/session'
import { dropTranscriptTail, saveTranscriptTail } from '@/store/transcript-tail-cache'

import { SESSION_OPEN_MARKS } from '../session/hooks/use-session-actions/session-open-marks'
import {
  runSessionOpenPerfFixture,
  type SessionOpenPerfFixtureCleanup
} from '../session/hooks/use-session-actions/session-open-perf-fixture'

import { selectFirstPaintTranscriptProof, waitForSessionTranscriptDom } from './session-open-dom-proof'

type Sample = {
  id: string
  phase: string
  actualDuration: number
  baseDuration: number
  startTime: number
  commitTime: number
}

type SyntheticDriverHandle = { stop: () => void }

type SessionSwitchFixtureResult = {
  /** The current resume/session.info protocol has no deferred-prewarm completion signal. */
  agentReady: null
  /** DOM observation after cache.commit; this is intentionally not a production performance mark. */
  cachePaintReady?: number
  marks: Partial<Record<(typeof SESSION_OPEN_MARKS)[number], number>>
}

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
      /**
       * Whether the active gateway socket is open. The perf harness waits on
       * this before measuring so background reconnect churn (a booting/absent
       * backend) doesn't contaminate frame-pacing numbers.
       */
      connected: () => boolean
      /** Mount files + multiple xterms for the synthetic right-pane scenario. */
      rightPaneSetup: (opts: { cwd: string; terminals?: number }) => { procId: string; terminalIds: string[] }
      rightPaneGit: (path: string, kind?: 'added' | 'conflicted' | 'modified') => void
      rightPaneReset: () => void
      rightPaneSelect: (id: string) => void
      rightPaneWrite: (procId: string, chunk: string) => void
      sessionSwitch: (opts?: { delayRuntimeMs?: number; verifiedCache?: boolean }) => Promise<SessionSwitchFixtureResult>
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
  let sessionSwitchRound = 0

  let rightPaneBaseline: null | {
    activeTerminalId: null | string
    cwd: string
    repoStatusByCwd: ReturnType<typeof $repoStatusByCwd.get>
    takeover: boolean
    terminals: readonly TerminalEntry[]
  } = null

  const stop = () => {
    activeHandle = null
    setBusy(false)
  }

  const resetRightPane = () => {
    if (!rightPaneBaseline) {
      return
    }

    setTerminalTakeover(rightPaneBaseline.takeover)
    $terminals.set(rightPaneBaseline.terminals)
    $activeTerminalId.set(rightPaneBaseline.activeTerminalId)
    $repoStatusByCwd.set(rightPaneBaseline.repoStatusByCwd)
    setCurrentCwdTransient(rightPaneBaseline.cwd)
    rightPaneBaseline = null
  }

  const marksForCurrentRound = (): SessionSwitchFixtureResult['marks'] =>
    Object.fromEntries(
      SESSION_OPEN_MARKS.flatMap(name => {
        const entries = performance.getEntriesByName(name)
        // A cold open can use the verified tail twice (the immediate paint and
        // a later fallback). The controlled cache-paint metric is intentionally
        // the first actual cache commit followed by rAF observation, never the
        // later bookkeeping commit.
        const entry = name === 'hermes.session.cache.commit' ? entries.at(0) : entries.at(-1)

        return entry ? [[name, entry.startTime]] : []
      })
    )

  const waitForSessionOpenMark = async (name: (typeof SESSION_OPEN_MARKS)[number]): Promise<number> => {
    const deadline = performance.now() + 5_000

    while (performance.now() < deadline) {
      const entry = performance.getEntriesByName(name).at(-1)

      if (entry) {
        return entry.startTime
      }

      await new Promise<void>(resolve => window.setTimeout(resolve, 0))
    }

    throw new Error(`session-switch fixture did not observe ${name}`)
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
    connected: () => {
      try {
        return $gateway.get()?.connectionState === 'open'
      } catch {
        return false
      }
    },
    rightPaneGit: (path, kind = 'modified') => {
      const file = {
        conflicted: kind === 'conflicted',
        path,
        staged: false,
        unstaged: kind === 'modified',
        untracked: kind === 'added'
      }

      const cwd = $currentCwd.get().trim()
      $repoStatusByCwd.set({
        ...$repoStatusByCwd.get(),
        [cwd]: {
          added: 0,
          ahead: 0,
          behind: 0,
          branch: 'perf',
          changed: 1,
          conflicted: kind === 'conflicted' ? 1 : 0,
          defaultBranch: 'main',
          detached: false,
          files: [file],
          removed: 0,
          staged: 0,
          unstaged: kind === 'modified' ? 1 : 0,
          untracked: kind === 'added' ? 1 : 0
        }
      })
    },
    rightPaneReset: resetRightPane,
    rightPaneSelect: selectTerminal,
    rightPaneSetup: ({ cwd, terminals = 3 }) => {
      resetRightPane()
      rightPaneBaseline = {
        activeTerminalId: $activeTerminalId.get(),
        cwd: $currentCwd.get(),
        repoStatusByCwd: $repoStatusByCwd.get(),
        takeover: $terminalTakeover.get(),
        terminals: $terminals.get()
      }

      setCurrentCwdTransient(cwd)
      const terminalIds = [createTerminal(cwd)]
      let procId = ''

      for (let index = 1; index < Math.max(1, terminals); index += 1) {
        procId = `right-pane-perf-${Date.now()}-${index}`
        const id = ensureAgentTerminal(procId, `perf output ${index}`)

        if (id) {
          terminalIds.push(id)
        }
      }

      if (procId) {
        selectTerminal(terminalIds.at(-1) ?? terminalIds[0])
      }

      setTerminalTakeover(true)

      return { procId, terminalIds }
    },
    rightPaneWrite: (procId, chunk) => writeAgentTerminalChunk(procId, chunk),
    sessionSwitch: async ({ delayRuntimeMs = 0, verifiedCache = false } = {}) => {
      // Controlled rounds clear only their fixed local marks; never wipe
      // browser navigation, React, or other user performance entries.
      for (const name of SESSION_OPEN_MARKS) {
        performance.clearMarks(name)
      }

      sessionSwitchRound += 1

      const storedSessionId = `perf-session-${sessionSwitchRound}`

      const authority = {
        connectionId: '',
        displayRevision: 1,
        lineageRootId: `perf-root-${sessionSwitchRound}`,
        profile: 'default',
        resolvedTipId: storedSessionId
      }

      // Verified-cache mode must retain the ENTIRE authoritative latest page;
      // otherwise a conditional unchanged response would prove only the 40-row
      // storage suffix. REST-publish mode keeps the heavier 80-turn payload.
      const transcript = Array.from({ length: verifiedCache ? 20 : 80 }, (_, index) => syntheticTurn(index)).flat()
      const originalActiveSessionId = $activeSessionId.get()
      const originalAwaitingResponse = $awaitingResponse.get()
      const originalBusy = $busy.get()
      const originalMessages = $messages.get()
      const originalSelectedStoredSessionId = $selectedStoredSessionId.get()
      const originalSessions = $sessions.get()
      const fixtureTimestamp = Math.floor(Date.now() / 1_000)
      let fixtureCleanup: SessionOpenPerfFixtureCleanup = () => undefined

      // Match a listed row to a real v3 local-cache entry. The mounted
      // useSessionActions hook below performs the real cache read, display
      // hydration, reconciliation, and fixed production marks. This fixture
      // supplies only deterministic no-network gateway/REST responses.
      $sessions.set([
        {
          _lineage_root_id: authority.lineageRootId,
          connection_id: authority.connectionId,
          display_revision: authority.displayRevision,
          ended_at: null,
          id: authority.resolvedTipId,
          input_tokens: 0,
          is_active: false,
          last_active: fixtureTimestamp,
          message_count: transcript.length,
          model: null,
          output_tokens: 0,
          preview: null,
          profile: authority.profile,
          source: 'desktop',
          started_at: fixtureTimestamp,
          title: 'Session-open performance fixture',
          tool_call_count: 0
        } satisfies (typeof originalSessions)[number]
      ])
      const pagination = {
        limit: 120,
        offset: 0,
        order: 'latest' as const,
        returned: transcript.length
      }
      saveTranscriptTail(storedSessionId, transcript, authority, { pagination })

      try {
        const result = runSessionOpenPerfFixture({
          delayRuntimeMs,
          fetchLatest: async knownDisplayRevision =>
            verifiedCache && knownDisplayRevision === authority.displayRevision
              ? {
                  display_revision: authority.displayRevision,
                  lineage_root_id: authority.lineageRootId,
                  messages: [],
                  pagination: { ...pagination, returned: 0 },
                  resolved_tip_id: authority.resolvedTipId,
                  session_id: storedSessionId,
                  unchanged: true
                }
              : {
                  display_revision: authority.displayRevision,
                  lineage_root_id: authority.lineageRootId,
                  messages: transcript.map(message => ({
                    content: message.parts,
                    role: message.role,
                    timestamp: message.timestamp
                  })),
                  pagination,
                  resolved_tip_id: authority.resolvedTipId,
                  session_id: storedSessionId
                },
          requestGateway: async method => {
            if (method !== 'session.resume') {
              throw new Error(`session-switch fixture received unexpected gateway method: ${method}`)
            }

            await new Promise<void>(resolve => window.setTimeout(resolve, Math.max(0, delayRuntimeMs)))

            return {
              info: {},
              messages: [],
              messages_omitted: true,
              resumed: storedSessionId,
              session_id: `perf-runtime-${sessionSwitchRound}`,
              session_key: storedSessionId,
              todo_state: { items: [] }
            }
          },
          storedSessionId
        })

        let cachePaintReady: number | undefined

        if (verifiedCache) {
          await waitForSessionOpenMark('hermes.session.cache.commit')
          const proof = selectFirstPaintTranscriptProof(transcript)

          if (!proof) {
            throw new Error('session-switch fixture could not select a first-paint transcript proof')
          }

          cachePaintReady = await waitForSessionTranscriptDom({
            ...proof,
            expectedStoredSessionId: storedSessionId,
            root: document.body
          })
        }

        fixtureCleanup = await result
        await Promise.all([
          ...(verifiedCache ? [] : [waitForSessionOpenMark('hermes.session.rest.commit')]),
          waitForSessionOpenMark('hermes.session.resume.ready'),
          waitForSessionOpenMark('hermes.session.history.ready')
        ])

        // No agent-ready mark: current protocol cannot report deferred agent
        // prewarm completion, and deriving one from the mock delay is dishonest.
        return { agentReady: null, cachePaintReady, marks: marksForCurrentRound() }
      } finally {
        fixtureCleanup()
        dropTranscriptTail(storedSessionId, authority)
        $sessions.set(originalSessions)
        setSelectedStoredSessionId(originalSelectedStoredSessionId)
        setActiveSessionId(originalActiveSessionId)
        setMessages(originalMessages)
        setBusy(originalBusy)
        setAwaitingResponse(originalAwaitingResponse)
      }
    },
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
      resetRightPane()

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
