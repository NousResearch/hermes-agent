import type { ChatMessage } from '@/lib/chat-messages'
import { setBusy, setMessages } from '@/store/session'

const HEARTBEAT_INTERVAL_MS = 100
const PAINT_VALIDATION_TIMEOUT_MS = 5_000

interface HeartbeatSample {
  at: number
  gapMs: number
  source: 'interval' | 'animation-frame' | 'long-task'
}

interface FixtureTurn {
  class: 'plain' | 'code' | 'tools' | 'mixed'
  messages: ChatMessage[]
}

interface ShortSessionDriver {
  fixtures: FixtureTurn[]
  heartbeatSamples: HeartbeatSample[]
  load: (count: number) => Promise<{
    discoveredAssistantMessages: number
    expectedAssistantMessages: number
    expectedCodeCards: number
    expectedTools: number
    expectedUserIds: string[]
    messageRecords: number
    paintedAssistantMessages: number
    paintedCodeCards: number
    paintedTools: number
    paintedUserIds: string[]
    syntheticTurns: number
  }>
  manifest: () => Promise<{
    bytes: number
    classes: FixtureTurn['class'][]
    messageRecords: number
    parts: number
    sha256: string
    tools: number
    syntheticTurns: number
  }>
  checkpoint: () => void
  reset: () => void
  summary: () => { maxGapMs: number; samples: HeartbeatSample[] }
}

declare global {
  interface Window {
    __SHORT_SESSION_HANG_REPRO__?: ShortSessionDriver
  }
}

const text = (value: string) => ({ text: value, type: 'text' as const })

function turn(index: number, fixtureClass: FixtureTurn['class'], assistantParts: ChatMessage['parts']): FixtureTurn {
  const timestamp = 1_700_000_000_000 + index * 1_000

  return {
    class: fixtureClass,
    messages: [
      {
        id: `short-session-u-${index}`,
        parts: [text(`Deterministic ${fixtureClass} prompt ${index}: preserve renderer responsiveness.`)],
        role: 'user',
        timestamp
      },
      {
        id: `short-session-a-${index}`,
        parts: assistantParts,
        pending: false,
        role: 'assistant',
        timestamp: timestamp + 1
      }
    ]
  }
}

const fixtures: FixtureTurn[] = [
  turn(1, 'plain', [text('Plain response one. The transcript should remain selectable and scrollable.')]),
  turn(2, 'code', [
    text('Code response two.\n\n```ts\nexport const bounded = (n: number) => Math.min(8, Math.max(1, n))\n```')
  ]),
  turn(3, 'tools', [
    {
      args: { path: '/tmp/hermes-short-session-fixture.txt' },
      argsText: '{"path":"/tmp/hermes-short-session-fixture.txt"}',
      result: { content: 'deterministic fixture output', ok: true },
      toolCallId: 'short-session-tool-3',
      toolName: 'read_file',
      type: 'tool-call'
    }
  ]),
  turn(4, 'mixed', [
    text(
      'Mixed response four starts with prose and a compact table.\n\n| field | value |\n|---|---|\n| stable | yes |'
    ),
    {
      args: { command: 'printf deterministic' },
      argsText: '{"command":"printf deterministic"}',
      result: { output: 'deterministic', status: 0 },
      toolCallId: 'short-session-tool-4',
      toolName: 'terminal',
      type: 'tool-call'
    },
    text('\nThe renderer must paint narration after the tool result.')
  ]),
  turn(5, 'plain', [text('Plain response five is the first required responsiveness checkpoint.')]),
  turn(6, 'code', [
    text('Code response six.\n\n```python\ndef heartbeat(now, previous):\n    return max(0, now - previous)\n```')
  ]),
  turn(7, 'tools', [
    {
      args: { query: 'deterministic local fixture' },
      argsText: '{"query":"deterministic local fixture"}',
      result: { matches: ['fixture-7'], ok: true },
      toolCallId: 'short-session-tool-7',
      toolName: 'search_files',
      type: 'tool-call'
    }
  ]),
  turn(8, 'mixed', [
    text('Mixed response eight is the final checkpoint. `inline code` and **markdown** remain interactive.'),
    {
      args: { path: '/tmp' },
      argsText: '{"path":"/tmp"}',
      result: { entries: ['hermes-short-session-fixture.txt'], ok: true },
      toolCallId: 'short-session-tool-8',
      toolName: 'list_directory',
      type: 'tool-call'
    }
  ])
]

if (typeof window !== 'undefined' && !window.__SHORT_SESSION_HANG_REPRO__) {
  const heartbeatSamples: HeartbeatSample[] = []
  let lastInterval = performance.now()
  let lastFrame = performance.now()
  let maxGapMsSoFar = 0
  let sampleWindowStartedAt = performance.now()

  const record = (sample: HeartbeatSample) => {
    if (document.visibilityState !== 'visible' || sample.at < sampleWindowStartedAt) {
      return
    }

    maxGapMsSoFar = Math.max(maxGapMsSoFar, sample.gapMs)
    heartbeatSamples.push(sample)

    if (heartbeatSamples.length > 2_000) {
      heartbeatSamples.splice(0, heartbeatSamples.length - 2_000)
    }
  }

  window.setInterval(() => {
    const now = performance.now()

    if (document.visibilityState === 'visible') {
      record({ at: now, gapMs: now - lastInterval, source: 'interval' })
    }

    lastInterval = now
  }, HEARTBEAT_INTERVAL_MS)

  const frame = (now: number) => {
    if (document.visibilityState === 'visible') {
      record({ at: now, gapMs: now - lastFrame, source: 'animation-frame' })
    }

    lastFrame = now
    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      const now = performance.now()
      lastInterval = now
      lastFrame = now
    }
  })

  const afterTwoFramesWhileVisible = () =>
    new Promise<void>((resolve, reject) => {
      if (document.visibilityState !== 'visible') {
        reject(new Error('renderer became hidden before paint validation'))

        return
      }

      let settled = false
      let deadline: number | undefined

      const finish = (error?: Error) => {
        if (settled) {
          return
        }

        settled = true

        if (deadline !== undefined) {
          window.clearTimeout(deadline)
        }

        document.removeEventListener('visibilitychange', onVisibilityChange)

        if (error) {
          reject(error)
        } else {
          resolve()
        }
      }

      const onVisibilityChange = () => {
        if (document.visibilityState !== 'visible') {
          finish(new Error('renderer became hidden during paint validation'))
        }
      }

      document.addEventListener('visibilitychange', onVisibilityChange)
      deadline = window.setTimeout(
        () => finish(new Error('paint validation did not observe two frames')),
        PAINT_VALIDATION_TIMEOUT_MS
      )
      requestAnimationFrame(() => {
        if (document.visibilityState !== 'visible') {
          finish(new Error('renderer became hidden before the first paint frame'))

          return
        }

        requestAnimationFrame(() => {
          if (document.visibilityState !== 'visible') {
            finish(new Error('renderer became hidden before the second paint frame'))

            return
          }

          finish()
        })
      })
    })

  try {
    const observer = new PerformanceObserver(list => {
      for (const entry of list.getEntries()) {
        record({ at: entry.startTime, gapMs: entry.duration, source: 'long-task' })
      }
    })

    observer.observe({ entryTypes: ['longtask'] })
  } catch {
    // Long Task API availability is a soft diagnostic signal.
  }

  const checkpoint = () => {
    const now = performance.now()

    sampleWindowStartedAt = now
    heartbeatSamples.length = 0
    maxGapMsSoFar = 0
    lastInterval = now
    lastFrame = now
  }

  const reset = () => {
    checkpoint()
    setBusy(false)
    setMessages([])
  }

  window.__SHORT_SESSION_HANG_REPRO__ = {
    fixtures,
    heartbeatSamples,
    load: async count => {
      if (!Number.isFinite(count)) {
        throw new RangeError('count must be a finite number')
      }

      const bounded = Math.min(fixtures.length, Math.max(1, Math.trunc(count)))
      const next = fixtures.slice(0, bounded).flatMap(fixture => fixture.messages)
      const expectedUserIds = next.filter(message => message.role === 'user').map(message => message.id)
      const expectedAssistantIds = next.filter(message => message.role === 'assistant').map(message => message.id)
      const expectedAssistantMessages = expectedAssistantIds.length
      const expectedTools = next.flatMap(message => message.parts).filter(part => part.type === 'tool-call').length

      const expectedCodeCards = next
        .flatMap(message => message.parts)
        .filter(part => part.type === 'text')
        .reduce((total, part) => total + Math.floor((part.text.match(/```/g)?.length ?? 0) / 2), 0)

      setBusy(false)
      setMessages(next)

      await afterTwoFramesWhileVisible()

      const isPainted = (element: Element) => {
        const style = getComputedStyle(element)

        return style.display !== 'none' && style.visibility !== 'hidden' && element.getClientRects().length > 0
      }

      const paintedUserIds = expectedUserIds.filter(id => {
        const element = document.querySelector(`[data-message-id="${CSS.escape(id)}"]`)

        return element ? isPainted(element) : false
      })

      const assistantRoots = [
        ...new Set(
          expectedUserIds.flatMap(id => {
            const user = document.querySelector(`[data-message-id="${CSS.escape(id)}"]`)
            const turn = user?.closest('[data-slot="aui_turn-pair"]')

            return turn ? [...turn.querySelectorAll('[data-slot="aui_assistant-message-root"]')] : []
          })
        )
      ]

      const paintedTools = assistantRoots
        .flatMap(root => [...root.querySelectorAll('[data-tool-row]')])
        .filter(isPainted).length

      const paintedCodeCards = assistantRoots
        .flatMap(root => [...root.querySelectorAll('[data-slot="code-card"]')])
        .filter(isPainted).length

      const paintedAssistantMessages = assistantRoots.filter(root => {
        if (!isPainted(root)) {
          return false
        }

        const hasText = Boolean(root.textContent?.trim())
        const hasPaintedTool = [...root.querySelectorAll('[data-tool-row]')].some(isPainted)
        const hasPaintedCode = [...root.querySelectorAll('[data-slot="code-card"]')].some(isPainted)

        return hasText || hasPaintedTool || hasPaintedCode
      }).length

      return {
        discoveredAssistantMessages: assistantRoots.length,
        expectedAssistantMessages,
        expectedCodeCards,
        expectedTools,
        expectedUserIds,
        messageRecords: next.length,
        paintedAssistantMessages,
        paintedCodeCards,
        paintedTools,
        paintedUserIds,
        syntheticTurns: bounded
      }
    },
    manifest: async () => {
      const serialized = JSON.stringify(fixtures)
      const bytes = new TextEncoder().encode(serialized)
      const digest = await crypto.subtle.digest('SHA-256', bytes)

      return {
        bytes: bytes.byteLength,
        classes: fixtures.map(fixture => fixture.class),
        messageRecords: fixtures.reduce((total, fixture) => total + fixture.messages.length, 0),
        parts: fixtures.reduce(
          (total, fixture) => total + fixture.messages.reduce((count, message) => count + message.parts.length, 0),
          0
        ),
        sha256: [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join(''),
        tools: fixtures.reduce(
          (total, fixture) =>
            total +
            fixture.messages.reduce(
              (count, message) => count + message.parts.filter(part => part.type === 'tool-call').length,
              0
            ),
          0
        ),
        syntheticTurns: fixtures.length
      }
    },
    checkpoint,
    reset,
    summary: () => ({
      maxGapMs: maxGapMsSoFar,
      samples: [...heartbeatSamples]
    })
  }
}
