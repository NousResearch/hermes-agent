import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process'
import * as path from 'node:path'
import { createInterface } from 'node:readline'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')
const DEFAULT_TIMEOUT_MS = 60_000

interface JsonRpcError {
  code?: number
  message?: string
}

interface JsonRpcFrame {
  error?: JsonRpcError
  id?: number
  method?: string
  params?: {
    payload?: unknown
    session_id?: string
    type?: string
  }
  result?: unknown
}

interface CreatedSession {
  session_id: string
  stored_session_id: string
}

export interface RealSessionSpec {
  /** Human-visible sidebar title, persisted by the first completed turn. */
  title: string
  /** Each item becomes one real user prompt followed by the mock provider's reply. */
  turns: readonly string[]
}

export interface RealSession {
  /** Runtime-only TUI session id, valid only while the builder process is alive. */
  runtimeId: string
  /** Durable SessionDB id that desktop resumes after the builder exits. */
  sessionId: string
}

/**
 * Creates durable desktop session history through the real TUI gateway and
 * AIAgent loop, using the E2E mock provider configured in `hermesHome`.
 *
 * This intentionally uses the shipped stdio JSON-RPC transport instead of
 * importing SessionDB or launching Electron. The desktop's WebSocket backend
 * dispatches the same `tui_gateway.server` methods.
 */
export class RealSessionBuilder {
  private readonly child: ChildProcessWithoutNullStreams
  private nextRequestId = 0
  private readonly pending = new Map<number, { reject: (reason: Error) => void; resolve: (value: unknown) => void }>()
  private readonly events: JsonRpcFrame[] = []
  private readonly eventWaiters: Array<{
    predicate: (frame: JsonRpcFrame) => boolean
    reject: (reason: Error) => void
    resolve: (frame: JsonRpcFrame) => void
  }> = []
  private readonly stderr: string[] = []
  private closed = false

  private constructor(hermesHome: string) {
    this.child = spawn('uv', ['run', '--active', '--no-sync', 'python', '-m', 'tui_gateway.entry'], {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        HERMES_HOME: hermesHome,
        PYTHONPATH: REPO_ROOT,
      },
      stdio: 'pipe',
    })

    createInterface({ input: this.child.stdout }).on('line', line => this.handleLine(line))
    createInterface({ input: this.child.stderr }).on('line', line => {
      this.stderr.push(line)
      if (this.stderr.length > 80) this.stderr.shift()
    })
    this.child.once('error', error => this.failAll(new Error(`real-session gateway failed to start: ${error.message}`)))
    this.child.once('exit', (code, signal) => {
      if (!this.closed) {
        this.failAll(new Error(`real-session gateway exited unexpectedly (${signal ?? code ?? 'unknown'}):\n${this.stderr.join('\n')}`))
      }
    })
  }

  static async start(hermesHome: string): Promise<RealSessionBuilder> {
    const builder = new RealSessionBuilder(hermesHome)
    await builder.waitForEvent(frame => frame.params?.type === 'gateway.ready')
    return builder
  }

  async createSession(spec: RealSessionSpec): Promise<RealSession> {
    if (spec.turns.length === 0) {
      throw new Error('RealSessionBuilder requires at least one turn so the real agent creates a durable session row')
    }

    const created = await this.request<CreatedSession>('session.create', {
      cols: 120,
      cwd: REPO_ROOT,
      source: 'desktop',
      title: spec.title,
    })
    const runtimeId = requireString(created, 'session_id')
    const sessionId = requireString(created, 'stored_session_id')

    for (const text of spec.turns) {
      const completion = this.waitForEvent(
        frame => frame.params?.type === 'message.complete' && frame.params.session_id === runtimeId,
      )
      await this.request('prompt.submit', { session_id: runtimeId, text })
      const frame = await completion
      const status = readString(frame.params?.payload, 'status')
      if (status !== 'complete') {
        throw new Error(`real session turn failed with status ${status ?? 'unknown'}: ${JSON.stringify(frame.params?.payload)}`)
      }
    }

    await this.request('session.close', { session_id: runtimeId })
    return { runtimeId, sessionId }
  }

  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    this.child.stdin.end()
    await new Promise<void>(resolve => {
      const timeout = setTimeout(() => {
        this.child.kill('SIGTERM')
        resolve()
      }, 5_000)
      this.child.once('exit', () => {
        clearTimeout(timeout)
        resolve()
      })
    })
  }

  private request<T = unknown>(method: string, params: Record<string, unknown>): Promise<T> {
    const id = ++this.nextRequestId
    return this.withTimeout(new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: value => resolve(value as T), reject })
      this.child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`, error => {
        if (error) {
          this.pending.delete(id)
          reject(error)
        }
      })
    }), `request ${method}`)
  }

  private waitForEvent(predicate: (frame: JsonRpcFrame) => boolean): Promise<JsonRpcFrame> {
    const index = this.events.findIndex(predicate)
    if (index >= 0) {
      return Promise.resolve(this.events.splice(index, 1)[0])
    }
    return this.withTimeout(new Promise<JsonRpcFrame>((resolve, reject) => {
      this.eventWaiters.push({ predicate, resolve, reject })
    }), 'gateway event')
  }

  private handleLine(line: string): void {
    let frame: JsonRpcFrame
    try {
      frame = JSON.parse(line) as JsonRpcFrame
    } catch {
      return
    }

    if (typeof frame.id === 'number') {
      const pending = this.pending.get(frame.id)
      if (!pending) return
      this.pending.delete(frame.id)
      if (frame.error) {
        pending.reject(new Error(`JSON-RPC error ${frame.error.code ?? 'unknown'}: ${frame.error.message ?? 'unknown error'}`))
      } else {
        pending.resolve(frame.result)
      }
      return
    }

    if (frame.method !== 'event') return
    const waiter = this.eventWaiters.find(candidate => candidate.predicate(frame))
    if (!waiter) {
      this.events.push(frame)
      return
    }
    this.eventWaiters.splice(this.eventWaiters.indexOf(waiter), 1)
    waiter.resolve(frame)
  }

  private withTimeout<T>(promise: Promise<T>, operation: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timed out after ${DEFAULT_TIMEOUT_MS / 1000}s waiting for ${operation}:\n${this.stderr.join('\n')}`)), DEFAULT_TIMEOUT_MS)
      promise.then(value => {
        clearTimeout(timer)
        resolve(value)
      }, error => {
        clearTimeout(timer)
        reject(error)
      })
    })
  }

  private failAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
    for (const waiter of this.eventWaiters) waiter.reject(error)
    this.eventWaiters.length = 0
  }
}

function readString(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== 'object') return undefined
  const candidate = (value as Record<string, unknown>)[key]
  return typeof candidate === 'string' ? candidate : undefined
}

function requireString(value: unknown, key: string): string {
  const candidate = readString(value, key)
  if (!candidate) throw new Error(`Gateway response omitted required ${key}: ${JSON.stringify(value)}`)
  return candidate
}
