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

import fs from 'node:fs'
import http from 'node:http'
import type { ServerResponse } from 'node:http'
import os from 'node:os'
import nodePath from 'node:path'

/** A canned assistant reply used for every chat completion request. */
export const MOCK_REPLY = 'Hello from the mock inference server! The full boot chain is working.'

export interface MockServerOptions {
  /** Pause the matching stream after its first token for session-switch E2E coverage. */
  holdFirstStreamForPrompt?: string
/** Pause the first completion whose request JSON contains this text. */
holdFirstCompletionContaining?: string
/** Absolute sandbox path written by the verify-on-stop scripted tool call. */
verificationWritePath?: string
/**
 * Sentinel path that ends the E2E_SIDEBAR_CROSS background process.
 *
 * Without it that process is a bare `sleep 5`, which races the agent turn and
 * the 4s auto-dismiss linger — see `createBackgroundReleaseHandle`. Pass a
 * handle's `path` to let the test decide when the process exits.
 */
backgroundReleasePath?: string
}

export interface MockServer {
  port: number
  url: string
  receivedPrompts: string[]
  waitForHeldStream: () => Promise<void>
  waitForHeldCompletion: () => Promise<void>
  releaseHeldStream: () => void
  heldCompletionCount: () => number
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

/** Per-server counter for the sidebar-states script (independent from _scriptIndex). */
let _sidebarScriptIndex = 0

/** Per-server counter for the cross-session sidebar script. */
let _sidebarCrossIndex = 0

/** Per-server counter for the queue-stop script. */
let _queueStopIndex = 0

/** Per-server counter for the correction/session-switch script. */
let _correctionSwitchIndex = 0

/** Per-server counter for the verify-on-stop script. */
let _verificationStopIndex = 0

/** Per-server counter for the task-panel warm-resume script. */
let _taskPanelResumeIndex = 0

/** User messages received by the mock, for E2E assertions on real submits. */
const _receivedUserTexts: string[] = []

/** Reset the script indices (called between tests via restartMockServer). */
function resetScriptIndex(): void {
  _scriptIndex = 0
  _sidebarScriptIndex = 0
  _sidebarCrossIndex = 0
  _queueStopIndex = 0
  _correctionSwitchIndex = 0
  _verificationStopIndex = 0
  _taskPanelResumeIndex = 0
  _receivedUserTexts.length = 0
}

/** Return the user prompts the real backend submitted to this mock server. */
export function receivedUserTexts(): readonly string[] {
  return _receivedUserTexts
}

// ─── Sidebar-states script ─────────────────────────────────────────────
//
// A separate trigger (E2E_SIDEBAR_TRIGGER) exercises the desktop sidebar's
// background-process and subagent states. The mock returns tool_calls that
// the agent executes for real — `terminal(background=true)` spawns a real
// (but trivial) background process, and `delegate_task` spawns a real
// subagent that calls the mock server and gets the canned reply.
//
// Turn 1: text + terminal(bg=true) + delegate_task → tools execute
// Turn 2: final answer → message.complete, dot transitions

const SIDEBAR_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Let me run a background task and delegate some work.',
    toolCalls: [
      {
        name: 'terminal',
        args: {
          command: 'echo "background process output" && sleep 1 && echo "done"',
          background: true,
          notify_on_complete: true,
        },
      },
      {
        name: 'delegate_task',
        args: {
          goal: 'Summarize the test results',
          context: 'This is a test subagent for the sidebar states E2E test.',
        },
      },
    ],
  },
  {
    text: 'All tasks complete. The background process finished and the subagent returned its summary.',
  },
]

// ─── Sidebar cross-session script ──────────────────────────────────────
//
// E2E_SIDEBAR_CROSS starts a long background process plus a subagent so the
// tests can:
//   1. See the background dot while the subagent runs.
//   2. Open a different session and see session A's dot transition to
//      "finished unread" when the background process completes.
//
// The background process must outlive the agent turn — the whole point is a
// dot that is still "running" after the final answer lands. A fixed `sleep`
// cannot guarantee that: on a loaded CI runner the turn (two model round
// trips + a real subagent delegation) can take longer than the sleep, the
// process exits early, the 4s success linger elapses, and the dot is gone
// before the test looks. That is a wall-clock race between three independent
// timers, and it made this the flakiest spec in the suite.
//
// When `backgroundReleasePath` is set the process instead blocks until the
// test creates that sentinel file, so the test — not the clock — decides when
// the dot clears. `sleep 5` remains the fallback for callers that don't pass
// a handle.
function sidebarCrossBgCommand(releasePath?: string): string {
  if (!releasePath) {
    return 'echo "long bg output" && sleep 5 && echo "finished"'
  }
  // Bounded wait (60s): if a test forgets to release (or crashes mid-way),
  // the process still exits instead of hanging the worker until the suite
  // times out.
  const quoted = JSON.stringify(releasePath)
  return [
    'echo "long bg output"',
    `for _ in $(seq 1 600); do [ -e ${quoted} ] && break; sleep 0.1; done`,
    'echo "finished"',
  ].join(' && ')
}

function sidebarCrossScript(releasePath?: string): ScriptedTurn[] {
  return [
    {
      text: 'Starting a long background task and delegating work.',
      toolCalls: [
        {
          name: 'terminal',
          args: {
            command: sidebarCrossBgCommand(releasePath),
            background: true,
            notify_on_complete: true,
          },
        },
        {
          name: 'delegate_task',
          args: {
            goal: 'Analyze cross-session state',
            context: 'Testing that the background dot updates across sessions.',
          },
        },
      ],
    },
    {
      text: 'Both tasks are running in the background now.',
    },
  ]
}

const SIDEBAR_CROSS_SCRIPT: ScriptedTurn[] = sidebarCrossScript()

const QUEUE_STOP_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Starting a task that will keep this turn active.',
    toolCalls: [{ name: 'clarify', args: { question: 'Keep working?', choices: ['Yes', 'No'] } }],
  },
  { text: 'The paused task completed.' },
]

// The reported correction arrived while a foreground tool was still running.
// Keep that boundary open long enough for the renderer to redirect the turn,
// then let the next model request complete normally.
const CORRECTION_SWITCH_SCRIPT: ScriptedTurn[] = [
  {
    text: 'Checking the long-running task before I continue.',
    toolCalls: [{ name: 'terminal', args: { command: 'sleep 5' } }],
  },
  { text: 'The corrected task finished.' },
]

export const CORRECTION_SWITCH_TRIGGER = 'E2E_CORRECTION_SWITCH_TRIGGER'

/**
 * Drives a real code edit followed by two finish attempts. Hermes should add
 * its synthetic verify-on-stop continuation after each finish attempt until
 * the bounded verifier gives up. The mock's request capture proves the nudge
 * reached the model; desktop must never render it as chat content.
 */
function verificationStopScript(writePath: string): ScriptedTurn[] {
  return [
  {
    text: 'I will make the requested code change.',
    toolCalls: [{
      name: 'write_file',
      args: {
        path: writePath,
        content: 'def changed_by_e2e():\n    return "changed"\n',
      },
    }],
  },
  { text: 'The code edit is complete.' },
  { text: 'I cannot provide fresh verification evidence for that edit.' },
  ]
}

export const VERIFICATION_STOP_TRIGGER = 'E2E_VERIFY_ON_STOP_TRIGGER'
export const VERIFICATION_STOP_TEXT = 'I cannot provide fresh verification evidence for that edit.'

/**
 * A marker that makes the mock emit a real blocking clarify tool call. Tests
 * use it to hold a turn open while exercising busy-composer interactions.
 */
export const BLOCKING_CLARIFY_TRIGGER = 'E2E_BLOCKING_CLARIFY_TRIGGER'
export const BLOCKING_CLARIFY_QUESTION = 'Keep this test turn running?'

/**
 * A long live response with a five-row todo card, held open by a foreground tool.
 * The transcript is deliberately taller than the viewport so warm-session
 * tests can detect when re-opening the session leaves it above the true bottom.
 */
export const TASK_PANEL_RESUME_TRIGGER = 'E2E_TASK_PANEL_RESUME_TRIGGER'
export const TASK_PANEL_RESUME_TEXT = Array.from(
  { length: 24 },
  (_, index) => `Task-panel clearance line ${index + 1}: inspect the restored working session geometry.`,
).join('\n\n')

const TASK_PANEL_RESUME_SCRIPT: ScriptedTurn[] = [
  {
    text: TASK_PANEL_RESUME_TEXT,
    toolCalls: [
      {
        name: 'todo',
        args: {
          todos: [
            { id: 'design', content: 'Design the restored layout', status: 'completed' },
            { id: 'implement', content: 'Implement the measured clearance', status: 'in_progress' },
            { id: 'verify', content: 'Verify the latest message stays visible', status: 'pending' },
            { id: 'review', content: 'Review the visual regression', status: 'pending' },
            { id: 'ship', content: 'Ship the focused fix', status: 'pending' },
          ],
        },
      },
      {
        name: 'terminal',
        args: { command: 'sleep 60' },
      },
    ],
  },
]

const BLOCKING_CLARIFY_TURN: ScriptedTurn = {
  text: '',
  toolCalls: [{ name: 'clarify', args: { question: BLOCKING_CLARIFY_QUESTION, choices: ['Yes', 'No'] } }],
}

/**
 * A marker that makes the mock emit a blocking BATCH clarify tool call
 * (multi-question form). Regression coverage for the duplicated-card bug:
 * the tool.start row and the clarify.request row carry different ids and a
 * batch payload has no top-level question, so the correlation key must come
 * from the question list or the card mounts twice.
 */
export const BATCH_CLARIFY_TRIGGER = 'E2E_BATCH_CLARIFY_TRIGGER'
export const BATCH_CLARIFY_QUESTIONS = [
  { question: 'Pick a batch drink?', choices: ['Coffee', 'Tea'] },
  { question: 'Pick a batch time?', choices: ['Morning', 'Night'] },
]

const BATCH_CLARIFY_TURN: ScriptedTurn = {
  text: '',
  toolCalls: [{ name: 'clarify', args: { questions: BATCH_CLARIFY_QUESTIONS } }],
}

function includesBatchClarifyTrigger(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.includes(BATCH_CLARIFY_TRIGGER)
  }

  if (Array.isArray(value)) {
    return value.some(includesBatchClarifyTrigger)
  }

  if (value && typeof value === 'object') {
    return Object.values(value).some(includesBatchClarifyTrigger)
  }

  return false
}

function includesBlockingClarifyTrigger(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.includes(BLOCKING_CLARIFY_TRIGGER)
  }

  if (Array.isArray(value)) {
    return value.some(includesBlockingClarifyTrigger)
  }

  if (value && typeof value === 'object') {
    return Object.values(value).some(includesBlockingClarifyTrigger)
  }

  return false
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
    let heldCompletionCount = 0
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
          const holdThisCompletion = Boolean(
            options.holdFirstCompletionContaining &&
            heldCompletionCount === 0 &&
            JSON.stringify(parsed).includes(options.holdFirstCompletionContaining),
          )

          // Detect the interim-message test trigger: the user's message
          // contains a specific keyword. The mock walks through the
          // INTERIM_SCRIPT turns in sequence.
          //
          // The trigger keyword is chosen so normal chat tests (which send
          // "Hello, can you hear me?" etc.) never hit this path.
          const messages: any[] = Array.isArray(parsed.messages) ? parsed.messages : []
          const lastUserMsg = [...messages].reverse().find(m => m?.role === 'user')
          const userText = typeof lastUserMsg?.content === 'string' ? lastUserMsg.content : ''
          if (userText) {
            _receivedUserTexts.push(userText)
          }
          const isInterimTrigger = userText.includes('E2E_INTERIM_TRIGGER')
          const isSidebarTrigger = userText.includes('E2E_SIDEBAR_TRIGGER')
          const isSidebarCrossTrigger = userText.includes('E2E_SIDEBAR_CROSS')
          const isQueueStopTrigger = userText.includes('E2E_QUEUE_STOP_TRIGGER')
          const isTaskPanelResumeTrigger = userText.includes(TASK_PANEL_RESUME_TRIGGER)
          const isVerificationStopTrigger = messages.some(
            message => typeof message?.content === 'string' && message.content.includes(VERIFICATION_STOP_TRIGGER),
          )
          const isCorrectionSwitchTrigger = messages.some(
            message => typeof message?.content === 'string' && message.content.includes(CORRECTION_SWITCH_TRIGGER),
          )

          if (isTaskPanelResumeTrigger) {
            const turn =
              TASK_PANEL_RESUME_SCRIPT[_taskPanelResumeIndex] ??
              TASK_PANEL_RESUME_SCRIPT[TASK_PANEL_RESUME_SCRIPT.length - 1]
            _taskPanelResumeIndex++
            const respond = () => {
              if (stream) {
                streamScriptedTurn(res, model, turn)
              } else {
                nonStreamingScriptedTurn(res, model, turn)
              }
            }

            if (holdThisCompletion) {
              heldCompletionCount++
              resolveHeldStreamStarted?.()
              void heldStreamReleased.then(respond)
            } else {
              respond()
            }
            return
          }

          if (includesBatchClarifyTrigger(parsed.messages)) {
            // Only the FIRST completion of the conversation scripts the batch
            // clarify. The trigger text stays in message history, so once the
            // answered tool result is present the turn falls through to the
            // canned reply — otherwise the mock loops the quiz forever.
            const hasToolResult = Array.isArray(parsed.messages)
              && parsed.messages.some((message: { role?: string }) => message?.role === 'tool')

            if (!hasToolResult) {
              if (stream) {
                streamScriptedTurn(res, model, BATCH_CLARIFY_TURN)
              } else {
                nonStreamingScriptedTurn(res, model, BATCH_CLARIFY_TURN)
              }
              return
            }
          }

          if (includesBlockingClarifyTrigger(parsed.messages)) {
            if (stream) {
              streamScriptedTurn(res, model, BLOCKING_CLARIFY_TURN)
            } else {
              nonStreamingScriptedTurn(res, model, BLOCKING_CLARIFY_TURN)
            }
            return
          }

          if (isQueueStopTrigger) {
            const turn = QUEUE_STOP_SCRIPT[_queueStopIndex] ?? QUEUE_STOP_SCRIPT[QUEUE_STOP_SCRIPT.length - 1]
            _queueStopIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isVerificationStopTrigger) {
            const script = verificationStopScript(options.verificationWritePath ?? 'e2e-verification-target.py')
            const turn = script[_verificationStopIndex] ?? script[script.length - 1]
            _verificationStopIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isCorrectionSwitchTrigger) {
            const turn = CORRECTION_SWITCH_SCRIPT[_correctionSwitchIndex] ?? CORRECTION_SWITCH_SCRIPT[CORRECTION_SWITCH_SCRIPT.length - 1]
            _correctionSwitchIndex++
            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isSidebarCrossTrigger) {
            const script = sidebarCrossScript(options.backgroundReleasePath)
            const turn = script[_sidebarCrossIndex] ?? script[script.length - 1]
            _sidebarCrossIndex++

            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

          if (isSidebarTrigger) {
            const turn = SIDEBAR_SCRIPT[_sidebarScriptIndex] ?? SIDEBAR_SCRIPT[SIDEBAR_SCRIPT.length - 1]
            _sidebarScriptIndex++

            if (stream) {
              streamScriptedTurn(res, model, turn)
            } else {
              nonStreamingScriptedTurn(res, model, turn)
            }
            return
          }

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
            streamTextResponse(res, model, MOCK_REPLY, holdThisStream || holdThisCompletion ? () => {
              if (holdThisCompletion) {
                heldCompletionCount++
              }
              resolveHeldStreamStarted?.()
              return heldStreamReleased
            } : undefined)
          } else {
            if (holdThisCompletion) {
              heldCompletionCount++
              resolveHeldStreamStarted?.()
              void heldStreamReleased.then(() => nonStreamingTextResponse(res, model, MOCK_REPLY))
            } else {
              nonStreamingTextResponse(res, model, MOCK_REPLY)
            }
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
        waitForHeldCompletion: () => heldStreamStarted,
        releaseHeldStream: () => releaseHeldStream?.(),
        heldCompletionCount: () => heldCompletionCount,
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

/** Test-controlled lifetime for the E2E_SIDEBAR_CROSS background process. */
export interface BackgroundReleaseHandle {
  /** Sentinel path — pass as `backgroundReleasePath` to `startMockServer`. */
  path: string
  /** End the background process now (creates the sentinel). */
  release: () => void
  /** Remove the sentinel if it still exists. Safe to call twice. */
  cleanup: () => void
}

/**
 * Create a sentinel that keeps the E2E_SIDEBAR_CROSS background process alive
 * until the test explicitly releases it.
 *
 * The cross-session sidebar tests need a background process that is still
 * RUNNING after the agent turn finishes — that is the state under test (a
 * session whose turn is done but whose background work is not). With a fixed
 * `sleep`, three independent clocks race: the sleep, the agent turn (two model
 * round trips plus a real subagent delegation), and the 4s success linger
 * before a finished task auto-dismisses. When a loaded CI runner makes the
 * turn slower than the sleep, the process is already gone and the assertion
 * samples an empty sidebar. Observed on CI 2026-07-26 across two unrelated
 * PRs: the "should appear" poll needed 7.5s to see the dot, by which point
 * `sleep 5` had exited.
 *
 * With a sentinel there is one clock and the test owns it:
 *
 * ```ts
 * const release = createBackgroundReleaseHandle()
 * const mock = await startMockServer({ backgroundReleasePath: release.path })
 * // ... assert the dot is visible; it cannot vanish on its own ...
 * release.release()   // now, and only now, the process exits
 * ```
 */
export function createBackgroundReleaseHandle(): BackgroundReleaseHandle {
  const path = nodePath.join(
    os.tmpdir(),
    `hermes-e2e-bg-release-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
  )
  return {
    path,
    release: () => {
      try {
        fs.writeFileSync(path, 'release')
      } catch {
        // The process also has a bounded fallback wait; a failed write must
        // not crash the test before its real assertions run.
      }
    },
    cleanup: () => {
      try {
        fs.rmSync(path, { force: true })
      } catch {
        // Best-effort — the sentinel lives in the OS temp dir.
      }
    },
  }
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

/** The sidebar-states script's text constants, exported for test assertions. */
export const SIDEBAR_TEXTS = {
  /** The interim text from turn 1 (alongside tool calls). */
  interimText: SIDEBAR_SCRIPT[0].text,
  /** The final answer text. */
  finalText: SIDEBAR_SCRIPT[SIDEBAR_SCRIPT.length - 1].text,
  /** The background process command (for asserting process.list entries). */
  bgCommand: 'echo "background process output" && sleep 1 && echo "done"',
  /** The subagent's goal (for asserting subagent panel state). */
  subagentGoal: 'Summarize the test results',
} as const

/** The cross-session sidebar script's text constants. */
export const SIDEBAR_CROSS_TEXTS = {
  /** The interim text from turn 1. */
  interimText: SIDEBAR_CROSS_SCRIPT[0].text,
  /** The final answer text. */
  finalText: SIDEBAR_CROSS_SCRIPT[SIDEBAR_CROSS_SCRIPT.length - 1].text,
  /**
   * The default (unheld) background process command. Tests that pass a
   * `backgroundReleasePath` get a sentinel-waiting command instead — see
   * `createBackgroundReleaseHandle`.
   */
  bgCommand: sidebarCrossBgCommand(),
  /** The subagent's goal. */
  subagentGoal: 'Analyze cross-session state',
} as const
