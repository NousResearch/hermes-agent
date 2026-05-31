// Fake JSON-RPC gateway for demo mode. Replaces window.WebSocket with a socket
// that answers the renderer's RPC requests from fixtures and can push
// server→client events (the same shapes the live backend emits). Exposes a
// control surface used by window.__demo to drive a scripted turn.
import { CONFIG, MODEL_OPTIONS } from './fixtures'

type Json = Record<string, unknown>
type GatewayEvent = { type: string; session_id?: string; payload?: Json }

const OPEN = 1

function sessionInfo(session_id: string, cwd = '~/code/hermes-agent'): Json {
  return {
    session_id,
    stored_session_id: session_id,
    resumed: session_id,
    message_count: 0,
    messages: [],
    info: {
      model: MODEL_OPTIONS.model,
      provider: MODEL_OPTIONS.provider,
      cwd,
      branch: 'main',
      running: false,
      tools: {},
      skills: []
    }
  }
}

// method -> (params) => result
const rpc: Record<string, (params?: Json) => Json> = {
  'setup.status': () => ({ provider_configured: true }),
  'setup.runtime_check': () => ({ ok: true }),
  'config.get': () => CONFIG,
  'model.options': () => MODEL_OPTIONS,
  'model.get': () => ({ provider: MODEL_OPTIONS.provider, model: MODEL_OPTIONS.model }),
  'model.current': () => ({ provider: MODEL_OPTIONS.provider, model: MODEL_OPTIONS.model }),
  'reload.env': () => ({ ok: true }),
  'context.suggestions': () => ({ suggestions: [] }),
  'complete.path': () => ({ completions: [] }),
  'prompt.submit': () => ({ ok: true }),
  'prompt.cancel': () => ({ ok: true }),
  'session.resume': p => sessionInfo((p?.session_id as string) || 'demo-session'),
  'session.create': p => sessionInfo('demo-session', (p?.cwd as string) || '~/code/hermes-agent')
}

export interface DemoTurn {
  reasoning?: string
  tool?: { name: string; tool_id: string; args?: Json; result?: Json }
  reply: string[]
}

export interface DemoControl {
  socket: DemoSocket | null
  termCb: ((data: string) => void) | null
  emit(event: GatewayEvent): boolean
  term(data: string): void
  playTurn(turn: DemoTurn, sessionId?: string): Promise<void>
}

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

export const control: DemoControl = {
  socket: null,
  termCb: null,
  emit(event) {
    if (!this.socket) {
      return false
    }

    this.socket.deliver(JSON.stringify({ method: 'event', params: event }))

    return true
  },
  term(data) {
    this.termCb?.(data)
  },
  async playTurn(turn, sessionId = 'demo-session') {
    this.emit({ type: 'message.start', session_id: sessionId })
    await sleep(400)

    if (turn.reasoning) {
      this.emit({ type: 'reasoning.delta', session_id: sessionId, payload: { text: turn.reasoning } })
      await sleep(900)
    }

    if (turn.tool) {
      this.emit({
        type: 'tool.start',
        session_id: sessionId,
        payload: { name: turn.tool.name, tool_id: turn.tool.tool_id, args: turn.tool.args }
      })
      await sleep(900)
      this.emit({
        type: 'tool.complete',
        session_id: sessionId,
        payload: { name: turn.tool.name, tool_id: turn.tool.tool_id, result: turn.tool.result }
      })
      await sleep(700)
    }

    const full = turn.reply.join('')

    for (let i = 0; i < full.length; i += 3) {
      this.emit({ type: 'message.delta', session_id: sessionId, payload: { text: full.slice(i, i + 3) } })
      await sleep(24)
    }

    this.emit({ type: 'message.complete', session_id: sessionId, payload: { text: full } })
  }
}

export class DemoSocket extends EventTarget {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3

  url: string
  readyState = 0
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: Event) => void) | null = null
  onerror: ((ev: Event) => void) | null = null

  constructor(url: string) {
    super()
    this.url = url
    control.socket = this
    // open on the next tick so the connect()'s 'open' listener is attached first
    setTimeout(() => {
      this.readyState = OPEN
      const ev = new Event('open')
      this.dispatchEvent(ev)
      this.onopen?.(ev)
    }, 0)
  }

  deliver(data: string) {
    const ev = new MessageEvent('message', { data })
    this.dispatchEvent(ev)
    this.onmessage?.(ev)
  }

  send(raw: string) {
    let frame: { id?: number | string; method?: string; params?: Json }

    try {
      frame = JSON.parse(raw)
    } catch {
      return
    }

    if (frame.id != null) {
      const handler = frame.method ? rpc[frame.method] : undefined
      let result: Json = {}

      try {
        if (handler) {
          result = handler(frame.params) || {}
        }
      } catch {
        // ignore handler errors in demo mode
      }

      setTimeout(() => this.deliver(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result })), 0)
    }
  }

  close() {
    this.readyState = 3
    const ev = new Event('close')
    this.dispatchEvent(ev)
    this.onclose?.(ev)
  }
}

export function installGateway(): DemoControl {
  window.WebSocket = DemoSocket as unknown as typeof WebSocket

  return control
}
