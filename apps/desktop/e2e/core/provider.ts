/**
 * Scripted, recording OpenAI-compatible provider for the core Desktop suite.
 *
 * Unlike the shared tests-js mock (trigger keywords + module-global counters,
 * one of the reasons the old lane drifted), every reply here is keyed by the
 * unique marker in the turn's own user message, and the step within a turn is
 * derived from the request itself (assistant messages after the last user
 * message). Two scenarios can never consume each other's script, and a replay
 * or retry of the same request gets the same answer.
 *
 * Every chunk actually written is recorded so the oracle can assert
 * "rendered == persisted == what the provider streamed".
 */

import http from 'node:http'
import type { AddressInfo } from 'node:net'

export interface Gate {
  open: () => void
  opened: Promise<void>
}

export function gate(): Gate {
  let open = () => {}

  const opened = new Promise<void>(resolve => {
    open = resolve
  })

  return { open, opened }
}

export interface ToolCallSpec {
  name: string
  args: Record<string, unknown>
}

/** One provider completion. Chunks are streamed in order: reasoning, text, tool calls. */
export interface Step {
  reasoning?: string[]
  text?: string[]
  toolCalls?: ToolCallSpec[]
  /** Hold the stream after the first text (or reasoning) chunk until the gate opens. */
  holdAfterFirstChunk?: Gate
}

export interface RecordedCompletion {
  marker: null | string
  step: number
  stream: boolean
  sentText: string
  sentReasoning: string
  toolCalls: string[]
  finished: boolean
  /** The client (the backend) hung up before the stream ended, e.g. an interrupt/steer. */
  aborted: boolean
  body: any
}

export interface ScriptedProvider {
  url: string
  /** Register the steps a turn keyed by `marker` answers with (step i = i-th completion of that turn). */
  script: (marker: string, steps: Step[]) => void
  completions: RecordedCompletion[]
  /** Resolves once the first chunk of `marker`'s step has been written to the wire. */
  streamStarted: (marker: string, step?: number) => Promise<void>
  close: () => Promise<void>
}

const MARKER_RE = /\bU\d+-[a-z0-9]+\b/

function textOf(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }

  if (Array.isArray(content)) {
    return content.map(part => (typeof part?.text === 'string' ? part.text : '')).join('')
  }

  return ''
}

function turnPosition(messages: any[]): { marker: null | string; step: number } {
  let lastUser = -1

  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === 'user') {
      lastUser = i

      break
    }
  }

  if (lastUser < 0) {
    return { marker: null, step: 0 }
  }

  const marker = MARKER_RE.exec(textOf(messages[lastUser].content))?.[0] ?? null
  const step = messages.slice(lastUser + 1).filter(m => m?.role === 'assistant').length

  return { marker, step }
}

function chunk(model: string, delta: Record<string, unknown>, finish: null | string = null): string {
  return `data: ${JSON.stringify({
    id: 'core-e2e',
    object: 'chat.completion.chunk',
    created: 0,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }]
  })}\n\n`
}

const tick = () => new Promise(resolve => setTimeout(resolve, 15))

export function startScriptedProvider(): Promise<ScriptedProvider> {
  const scripts = new Map<string, Step[]>()
  const completions: RecordedCompletion[] = []
  const started = new Map<string, Gate>()

  const startedGate = (key: string) => {
    let g = started.get(key)

    if (!g) {
      g = gate()
      started.set(key, g)
    }

    return g
  }

  async function streamStep(res: http.ServerResponse, model: string, step: Step, rec: RecordedCompletion) {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
    res.on('close', () => {
      if (!res.writableFinished) {
        rec.aborted = true
      }
    })
    let first = true

    const afterChunk = async () => {
      if (first) {
        first = false
        startedGate(`${rec.marker}#${rec.step}`).open()

        if (step.holdAfterFirstChunk) {
          await step.holdAfterFirstChunk.opened
        }
      }

      await tick()
    }

    for (const piece of step.reasoning ?? []) {
      if (rec.aborted) {
        return
      }

      res.write(chunk(model, { reasoning_content: piece }))
      rec.sentReasoning += piece
      await afterChunk()
    }

    for (const piece of step.text ?? []) {
      if (rec.aborted) {
        return
      }

      res.write(chunk(model, { content: piece }))
      rec.sentText += piece
      await afterChunk()
    }

    const calls = step.toolCalls ?? []

    if (rec.aborted) {
      return
    }

    if (calls.length > 0) {
      res.write(
        chunk(model, {
          tool_calls: calls.map((call, index) => ({
            index,
            id: `call_core_${rec.marker}_${rec.step}_${index}`,
            type: 'function',
            function: { name: call.name, arguments: JSON.stringify(call.args) }
          }))
        })
      )
      rec.toolCalls.push(...calls.map(call => call.name))
      await afterChunk()
    }

    res.write(chunk(model, {}, calls.length > 0 ? 'tool_calls' : 'stop'))
    res.write('data: [DONE]\n\n')
    res.end()
    rec.finished = true
  }

  function plainJson(res: http.ServerResponse, model: string, step: Step, rec: RecordedCompletion) {
    const text = (step.text ?? []).join('')
    const calls = step.toolCalls ?? []
    rec.sentText = text
    rec.sentReasoning = (step.reasoning ?? []).join('')
    rec.toolCalls.push(...calls.map(call => call.name))
    rec.finished = true
    startedGate(`${rec.marker}#${rec.step}`).open()
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(
      JSON.stringify({
        id: 'core-e2e',
        object: 'chat.completion',
        created: 0,
        model,
        choices: [
          {
            index: 0,
            finish_reason: calls.length > 0 ? 'tool_calls' : 'stop',
            message: {
              role: 'assistant',
              content: text || null,
              ...(rec.sentReasoning ? { reasoning_content: rec.sentReasoning } : {}),
              ...(calls.length > 0
                ? {
                    tool_calls: calls.map((call, index) => ({
                      id: `call_core_${rec.marker}_${rec.step}_${index}`,
                      type: 'function',
                      function: { name: call.name, arguments: JSON.stringify(call.args) }
                    }))
                  }
                : {})
            }
          }
        ],
        usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 }
      })
    )
  }

  const server = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url?.startsWith('/v1/models')) {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(
        JSON.stringify({ object: 'list', data: [{ id: 'mock-model', object: 'model', created: 0, owned_by: 'core' }] })
      )

      return
    }

    if (req.method !== 'POST' || !req.url?.startsWith('/v1/chat/completions')) {
      res.writeHead(404, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'not found' }))

      return
    }

    let raw = ''
    req.on('data', (data: Buffer) => {
      raw += data.toString()
    })
    req.on('end', () => {
      let body: any = {}

      try {
        body = JSON.parse(raw)
      } catch {
        body = {}
      }

      const messages: any[] = Array.isArray(body.messages) ? body.messages : []
      const { marker, step: stepIndex } = turnPosition(messages)
      const steps = marker ? scripts.get(marker) : undefined
      // Unscripted traffic (auxiliary calls, a marker-less prompt) gets a
      // fixed short answer; it is recorded, never matched to a scenario.
      const step: Step = steps?.[Math.min(stepIndex, steps.length - 1)] ?? { text: ['ok'] }
      const model = typeof body.model === 'string' ? body.model : 'mock-model'

      const rec: RecordedCompletion = {
        marker: steps ? marker : null,
        step: stepIndex,
        stream: body.stream === true,
        sentText: '',
        sentReasoning: '',
        toolCalls: [],
        finished: false,
        aborted: false,
        body
      }

      completions.push(rec)

      if (rec.stream) {
        void streamStep(res, model, step, rec).catch(() => res.destroy())
      } else {
        plainJson(res, model, step, rec)
      }
    })
  })

  return new Promise((resolve, reject) => {
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as AddressInfo
      resolve({
        url: `http://127.0.0.1:${port}`,
        script: (marker, steps) => {
          scripts.set(marker, steps)
        },
        completions,
        streamStarted: (marker, step = 0) => startedGate(`${marker}#${step}`).opened,
        close: () =>
          new Promise<void>(done => {
            server.closeAllConnections?.()
            server.close(() => done())
          })
      })
    })
  })
}
