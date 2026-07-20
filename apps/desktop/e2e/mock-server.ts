/**
 * Minimal OpenAI-compatible mock inference server for E2E tests.
 *
 * Implements just enough of the /v1/* surface for `hermes serve` to resolve a
 * provider, list models, and stream a canned chat completion back to the
 * desktop app — without any real LLM.
 *
 * Endpoints:
 *   GET  /v1/models             → { data: [{ id, ... }] }
 *   POST /v1/chat/completions   → streaming (SSE) or non-streaming response
 *
 * The canned response is a short, deterministic assistant message. Tool-call
 * requests are not simulated — the E2E tests only need the chat surface to
 * prove the full boot → gateway → inference → renderer chain works.
 */

import http from 'node:http'
import type { ServerResponse } from 'node:http'

/** A canned assistant reply used for every chat completion request. */
export const MOCK_REPLY = 'Hello from the mock inference server! The full boot chain is working.'

export interface MockServerOptions {
  /** Pause the matching stream after its first token for session-switch E2E coverage. */
  holdFirstStreamForPrompt?: string
}

export interface MockServer {
  port: number
  url: string
  receivedPrompts: string[]
  waitForHeldStream: () => Promise<void>
  releaseHeldStream: () => void
  close: () => Promise<void>
}

// ─── Multi-turn interim script ─────────────────────────────────────────
//
// When the user's message contains the trigger keyword, the mock server
// walks through a scripted sequence of responses that exercise the
// interim-assistant-message fix (#65919) across several patterns:
//
//   1. text + single tool_call  → should produce an interim message
//   2. text + single tool_call  → another interim message
//   3. no text + tool_call       → NO interim (no visible text alongside tools)
//   4. text + single tool_call  → another interim message
//   5. final answer (stop)      → message.complete, different from all interims
//
// Each "turn" is one API call. The agent executes the tool after each
// tool_calls response, then re-calls the API, advancing to the next turn.

export interface ScriptedTurn {
  /** Assistant text content to stream. Empty string = no visible text. */
  text: string
  /** Tool calls to emit. Empty array = final turn (finish_reason: stop). */
  toolCalls?: Array<{
    name: string
    args: Record<string, unknown>
  }>
}

const INTERIM_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Let me start by planning the approach.',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '1', content: 'Plan', status: 'in_progress' }] } }],
  },
  {
    text: 'Now checking the details before answering.',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '2', content: 'Check details', status: 'in_progress' }] } }],
  },
  {
    // No visible text alongside this tool call — should NOT produce an
    // interim message. The agent fires _emit_interim_assistant_message
    // but _interim_assistant_visible_text returns "" so it's a no-op.
    text: '',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '3', content: 'Silent step', status: 'completed' }] } }],
  },
  {
    text: 'Found something interesting worth noting.',
    toolCalls: [{ name: 'todo', args: { todos: [{ id: '4', content: 'Note finding', status: 'completed' }] } }],
  },
  {
    // Final answer — different from all interim texts.
    text: 'All done! Here is the complete summary of what I found.',
  },
]

/** Per-server request counter so we can walk through the script turns. */
let _scriptIndex = 0

/** Reset the script index (called between tests via restartMockServer). */
function resetScriptIndex(): void {
  _scriptIndex = 0
}

/**
 * Start the mock server on an ephemeral port.
 *
 * @returns a handle with `port`, `url`, received user prompts, and `close()`.
 */
export function startMockServer(options: MockServerOptions = {}): Promise<MockServer> {
  return new Promise((resolve, reject) => {
    const receivedPrompts: string[] = []
    let resolveHeldStreamStarted: (() => void) | null = null
    let releaseHeldStream: (() => void) | null = null
    const heldStreamStarted = new Promise<void>(resolveHeld => {
      resolveHeldStreamStarted = resolveHeld
    })
    const heldStreamReleased = new Promise<void>(resolveRelease => {
      releaseHeldStream = resolveRelease
    })
    const server = http.createServer((req, res) => {
      // CORS headers — the Electron renderer doesn't need them, but they
      // don't hurt and make the server usable from a browser context too.
      res.setHeader('Access-Control-Allow-Origin', '*')
      res.setHeader('Access-Control-Allow-Headers', '*')
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

      if (req.method === 'OPTIONS') {
        res.writeHead(204)
        res.end()
        return
      }

      // GET /v1/models — return a single fake model.
      if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'Content-Type': 'application/json' })
        res.end(
          JSON.stringify({
            object: 'list',
            data: [
              {
                id: 'mock-model',
                object: 'model',
                created: 0,
                owned_by: 'mock',
              },
            ],
          }),
        )
        return
      }

      // POST /v1/chat/completions — return a canned response.
      if (req.method === 'POST' && req.url?.startsWith('/v1/chat/completions')) {
        let body = ''

        req.on('data', (chunk: Buffer) => {
          body += chunk.toString()
        })

        req.on('end', () => {
          let parsed: any = {}

          try {
            parsed = JSON.parse(body)
          } catch {
            // malformed JSON — treat as non-streaming with defaults
          }

          const lastUserMessage = [...(parsed.messages ?? [])]
            .reverse()
            .find((message: { role?: unknown }) => message?.role === 'user')

          if (typeof lastUserMessage?.content === 'string') {
            receivedPrompts.push(lastUserMessage.content)
          }

          const stream = parsed.stream === true
          const model = parsed.model || 'mock-model'

          // Detect the interim-message test trigger: the user's message
          // contains a specific keyword. The mock walks through the
          // INTERIM_SCRIPT turns in sequence.
          //
          // The trigger keyword is chosen so normal chat tests (which send
          // "Hello, can you hear me?" etc.) never hit this path.
          const messages: any[] = Array.isArray(parsed.messages) ? parsed.messages : []
          const lastUserMsg = [...messages].reverse().find(m => m?.role === 'user')
          const userText = typeof lastUserMsg?.content === 'string' ? lastUserMsg.content : ''
          const isInterimTrigger = userText.includes('E2E_INTERIM_TRIGGER')

          if (isInterimTrigger) {
            const turn = INTERIM_SCRIPT[_scriptIndex] ?? INTERIM_SCRIPT[INTERIM_SCRIPT.length - 1]
            _scriptIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (stream) {
            const holdThisStream = Boolean(
              options.holdFirstStreamForPrompt && typeof lastUserMessage?.content === 'string' &&
                lastUserMessage.content.includes(options.holdFirstStreamForPrompt),
            )
            streamTextResponse(res, model, MOCK_REPLY, holdThisStream ? () => {
              resolveHeldStreamStarted?.()
              return heldStreamReleased
            } : undefined)
          } else {
            nonStreamingTextResponse(res, model, MOCK_REPLY)
          }
        })

        req.on('error', () => {
          res.writeHead(400)
          res.end('Bad request')
        })
        return
      }

      // Fallback — 404 for anything else
      res.writeHead(404, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: 'Not found' }))
    })

    server.on('error', reject)

    server.listen(0, '127.0.0.1', () => {
      const addr = server.address()
      if (addr === null || typeof addr === 'string') {
        reject(new Error('Failed to get server address'))
        return
      }

      const port = addr.port
      const url = `http://127.0.0.1:${port}`

      resolve({
        port,
        url,
        receivedPrompts,
        waitForHeldStream: () => heldStreamStarted,
        releaseHeldStream: () => releaseHeldStream?.(),
        close: () =>
          new Promise((resolveClose, rejectClose) => {
            server.close((err) => {
              if (err) {
                rejectClose(err)
              } else {
                resolveClose()
              }
            })
          }),
      })
    })
  })
}

// ─── Response helpers ──────────────────────────────────────────────────

/** SSE chunk shape for a streaming chat completion. */
function sseChunk(model: string, delta: Record<string, unknown>, finishReason: string | null = null): string {
  return `data: ${JSON.stringify({
    id: 'mock-completion',
    object: 'chat.completion.chunk',
    created: 0,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })}\n\n`
}

/**
 * Stream a plain text response (no tool calls) as SSE, finishing with
 * `finish_reason: "stop"`. This is the default canned-reply path.
 */
function streamTextResponse(
  res: ServerResponse,
  model: string,
  text: string,
  waitForRelease?: () => Promise<void>,
): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  })

  const words = text.split(' ')
  let i = 0

  const sendChunk = (): void => {
    if (i >= words.length) {
      res.write(sseChunk(model, {}, 'stop'))
      res.write('data: [DONE]\n\n')
      res.end()
      return
    }

    const word = i === 0 ? words[i] : ' ' + words[i]
    res.write(sseChunk(model, { content: word }))
    i++
    if (waitForRelease && i === 1) {
      waitForRelease().then(() => setTimeout(sendChunk, 20))
      return
    }
    setTimeout(sendChunk, 20)
  }

  sendChunk()
}

/** Non-streaming plain text response. */
function nonStreamingTextResponse(res: ServerResponse, model: string, text: string): void {
  res.writeHead(200, { 'Content-Type': 'application/json' })
  res.end(
    JSON.stringify({
      id: 'mock-completion',
      object: 'chat.completion',
      created: 0,
      model,
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: text },
          finish_reason: 'stop',
        },
      ],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    }),
  )
}

/**
 * Stream a single scripted turn: first the text content (word by word),
 * then a chunk carrying the tool_calls (if any), with the appropriate
 * finish_reason.
 *
 * If the turn has no text and no tool calls, it's an empty final response.
 * If it has text but no tool calls, it's a final answer (finish_reason: stop).
 * If it has tool calls (with or without text), finish_reason is "tool_calls".
 */
function streamScriptedTurn(
  res: ServerResponse,
  model: string,
  turn: ScriptedTurn,
): void {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  })

  const hasToolCalls = turn.toolCalls && turn.toolCalls.length > 0
  const finishReason = hasToolCalls ? 'tool_calls' : 'stop'

  // If there's no text to stream, go straight to the tool_calls / finish.
  if (!turn.text) {
    if (hasToolCalls) {
      res.write(
        sseChunk(model, {
          tool_calls: turn.toolCalls!.map((tc, idx) => ({
            index: idx,
            id: `call_e2e_${_scriptIndex}_${idx}`,
            type: 'function',
            function: { name: tc.name, arguments: JSON.stringify(tc.args) },
          })),
        }, finishReason),
      )
    } else {
      res.write(sseChunk(model, {}, finishReason))
    }
    res.write('data: [DONE]\n\n')
    res.end()
    return
  }

  // Stream the text word by word, then emit tool_calls if present.
  const words = turn.text.split(' ')
  let i = 0

  const sendChunk = (): void => {
    if (i >= words.length) {
      // All text streamed — emit tool_calls if present, then finish.
      if (hasToolCalls) {
        res.write(
          sseChunk(model, {
            tool_calls: turn.toolCalls!.map((tc, idx) => ({
              index: idx,
              id: `call_e2e_${_scriptIndex}_${idx}`,
              type: 'function',
              function: { name: tc.name, arguments: JSON.stringify(tc.args) },
            })),
          }, finishReason),
        )
      } else {
        res.write(sseChunk(model, {}, finishReason))
      }
      res.write('data: [DONE]\n\n')
      res.end()
      return
    }

    const word = i === 0 ? words[i] : ' ' + words[i]
    res.write(sseChunk(model, { content: word }))
    i++
    setTimeout(sendChunk, 20)
  }

  sendChunk()
}

/** Non-streaming version of a scripted turn. */
function nonStreamingScriptedTurn(
  res: ServerResponse,
  model: string,
  turn: ScriptedTurn,
): void {
  const hasToolCalls = turn.toolCalls && turn.toolCalls.length > 0
  const finishReason = hasToolCalls ? 'tool_calls' : 'stop'

  const message: Record<string, unknown> = { role: 'assistant' }
  if (turn.text) {
    message.content = turn.text
  }
  if (hasToolCalls) {
    message.tool_calls = turn.toolCalls!.map((tc, idx) => ({
      id: `call_e2e_${_scriptIndex}_${idx}`,
      type: 'function',
      function: { name: tc.name, arguments: JSON.stringify(tc.args) },
    }))
  }

  res.writeHead(200, { 'Content-Type': 'application/json' })
  res.end(
    JSON.stringify({
      id: 'mock-completion',
      object: 'chat.completion',
      created: 0,
      model,
      choices: [{ index: 0, message, finish_reason: finishReason }],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    }),
  )
}

/**
 * Restart the mock server's script index so each test starts from turn 0.
 * Call this between tests that use the interim trigger.
 */
export function restartMockServer(): void {
  resetScriptIndex()
}

/**
 * The interim script's text constants, exported for test assertions.
 * Each entry is the visible text of one turn. Turns with empty text
 * produce no interim message and are excluded from this list.
 */
export const INTERIM_TEXTS = {
  /** All interim texts that should appear as sealed messages when the flag is ON. */
  interims: INTERIM_SCRIPT
    .filter((t) => t.text && t.toolCalls)
    .map((t) => t.text),
  /** The final answer text. */
  finalText: INTERIM_SCRIPT[INTERIM_SCRIPT.length - 1].text,
  /** Text that should NOT produce an interim (empty-text tool turn). */
  silentTurnIndex: INTERIM_SCRIPT.findIndex((t) => !t.text && t.toolCalls),
} as const
