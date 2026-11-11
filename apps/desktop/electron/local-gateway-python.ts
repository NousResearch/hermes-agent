import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process'
import { statSync } from 'node:fs'
import path from 'node:path'

import { type GatewayTicketPurpose, ticketFailure } from './local-gateway'
import { hiddenWindowsChildOptions, windowsShellCommand } from './windows-child-options'

interface TicketEndpoint {
  profile_id: string
  instance_id: string
  runtime_protocol: number
  /** Multiplexer home whose control socket mints tickets for a served secondary. */
  control_home?: string | null
}

// Reuse the runtime's bounded control client: POSIX home/ACL/socket checks and
// Windows pipe server PID/SID validation have one canonical implementation.
const TICKET_SCRIPT = `
import json, os, sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
with redirect_stdout(sys.stderr):
    from hermes_cli.gateway_client import _session_ticket
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, _private_node
while True:
    line = sys.stdin.buffer.readline(65537)
    if not line:
        break
    if len(line) > 65536 or not line.endswith(b'\\n'):
        break
    request = json.loads(line)
    try:
        with redirect_stdout(sys.stderr):
            endpoint = SimpleNamespace(**{'control_home': None, **request['endpoint']})
            home = Path(endpoint.profile_id)
            if str(home.resolve()) != endpoint.profile_id or endpoint.runtime_protocol != 1:
                raise ValueError('invalid ticket endpoint')
            if os.name != 'nt':
                _private_node(home, kind='directory', home=True)
            ticket = _session_ticket(home, endpoint, purpose=request['purpose'],
                                     scope='host' if request['purpose'] == 'interactive' else None)
        reply = {'id': request['id'], 'ticket': ticket}
    except DiscoveryError as exc:
        reply = {'id': request['id'], 'error': exc.reason}
    except Exception:
        reply = {'id': request['id'], 'error': 'ticket_failed'}
    print(json.dumps(reply), flush=True)
`

interface PendingTicket {
  resolve: (ticket: string) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout>
}

/** How much of the helper's stderr is kept as the reason a mint failed. */
const TICKET_STDERR_TAIL = 400

const bridges = new Map<string, TicketBridge>()
let bridgeGeneration = 0

/** Reuse one private helper per runtime/profile; every ticket still verifies the live owner. */
class TicketBridge {
  private child: ChildProcessWithoutNullStreams
  private pending = new Map<number, PendingTicket>()
  private sequence = 0
  private output = ''
  private idleTimer?: ReturnType<typeof setTimeout>
  private closed = false
  /** Why this bridge is gone plus the helper's own last words: with `dispose()` rejecting every
   *  waiter with the bare message, this was the only cause a failed mint ever had. */
  private exitReason = 'the ticket bridge was closed'
  private stderrTail = ''

  constructor(command: string, cwd: string, env: NodeJS.ProcessEnv, private key: string) {
    this.child = spawn(command, ['-P', '-u', '-c', TICKET_SCRIPT], hiddenWindowsChildOptions({
      cwd, env, shell: false, stdio: ['pipe', 'pipe', 'pipe']
    }))
    // Read, never discard: a helper that dies on startup (a `python3` too old for `-P`, a checkout
    // missing from PYTHONPATH) says why on stderr, and that line is the whole diagnosis.
    this.child.stderr.setEncoding('utf8')
    this.child.stderr.on('data', data => { this.stderrTail = `${this.stderrTail}${String(data)}`.slice(-TICKET_STDERR_TAIL) })
    this.child.stdout.setEncoding('utf8')
    this.child.stdout.on('data', data => this.receive(String(data)))
    this.child.on('error', error => this.dispose(`the ticket bridge failed to start: ${error.message}`))
    this.child.stdin.on('error', error => this.dispose(`the ticket bridge request pipe failed: ${error.message}`))
    this.child.on('close', code => this.dispose(`the ticket bridge exited (code ${code ?? 'unknown'})`))
  }

  request(endpoint: TicketEndpoint, purpose: GatewayTicketPurpose): Promise<string> {
    clearTimeout(this.idleTimer)

    return new Promise((resolve, reject) => {
      if (this.closed) { reject(ticketFailure(this.exitReason));

 return }

      const id = ++this.sequence
      const request = JSON.stringify({ id, endpoint, purpose }) + '\n'

      if (Buffer.byteLength(request) > 65536) { reject(new Error('Gateway ticket request too large'));

 return }

      const timer = setTimeout(() => this.dispose('the ticket bridge did not answer within 10000ms'), 10_000)
      this.pending.set(id, { resolve, reject, timer })
      // Private stdin carries identity as data. EOF also retires the helper if Desktop exits.
      this.child.stdin.write(request)
    })
  }

  private receive(data: string): void {
    this.output += data

    if (this.output.length > 65536) { this.dispose();

 return }

    let end: number

    while ((end = this.output.indexOf('\n')) >= 0) {
      const line = this.output.slice(0, end)
      this.output = this.output.slice(end + 1)

      try {
        const reply = JSON.parse(line)
        const waiter = this.pending.get(reply.id)

        if (!waiter) { this.dispose();

 return }

        this.pending.delete(reply.id)
        clearTimeout(waiter.timer)

        if (typeof reply.ticket === 'string' && reply.ticket) { waiter.resolve(reply.ticket) }
        // The helper's own verdict (`DiscoveryError.reason`, `ticket_failed`) stayed a side property
        // that nothing printed; it is the cause, so it belongs in the message too. `reason` is kept
        // for `mintGroupWritableHome` / `plugin-ws-url`, which route a refusal by it.
        else { waiter.reject(Object.assign(ticketFailure(typeof reply.error === 'string' ? reply.error : ''),
          typeof reply.error === 'string' ? { reason: reply.error } : {})) }
      } catch { this.dispose();

 return }
    }

    if (!this.pending.size) { this.idleTimer = setTimeout(() => this.dispose(), 30_000) }
  }

  dispose(reason = 'the ticket bridge was closed'): void {
    if (this.closed) { return }
    this.closed = true
    clearTimeout(this.idleTimer)
    // The helper's stderr is the only record of why it went away, and both shapes matter: a CLI
    // usage error opens with the real problem (`Unknown option: -P`) and trails off into `usage:`,
    // a Python crash ends with the exception (`ModuleNotFoundError: No module named 'hermes_cli'`)
    // under a traceback. Keep the first line and the last one rather than choosing between them.
    const lines = this.stderrTail.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
    const detail = lines.length > 1 ? `${lines[0]} … ${lines[lines.length - 1]}` : lines[0]
    this.exitReason = detail ? `${reason}: ${detail}` : reason

    if (bridges.get(this.key) === this) { bridges.delete(this.key) }

    for (const waiter of this.pending.values()) {
      clearTimeout(waiter.timer)
      waiter.reject(ticketFailure(this.exitReason))
    }

    this.pending.clear()
    this.child.kill('SIGKILL')
  }
}

export function closeGatewayTicketBridges(): void {
  bridgeGeneration++

  for (const bridge of bridges.values()) { bridge.dispose() }
}

interface ResolvedTicketBackend {
  command: string
  kind: string
  shell?: boolean
  env?: NodeJS.ProcessEnv
  root?: string
  /** A bundled CLI still launches normally, but ticket minting needs its interpreter. */
  ticketCommand?: string
  ticketEnv?: NodeJS.ProcessEnv
}

const RUNTIME_COMMAND_TIMEOUT_MS = 30_000

/** Ask an installation launcher for its interpreter; it owns ABI and store selection. */
function launcherRuntimePython(backend: ResolvedTicketBackend & { root: string }): Promise<string> {
  return new Promise((resolve, reject) => {
    const fail = (detail: string) => reject(new Error(`Gateway ticket bootstrap could not resolve the installed Hermes Python runtime: ${detail}`))

    // A .cmd override is a shell backend: quote it as the backend spawn does, or cmd.exe
    // splits a launcher path with a space at the first one (#74064).
    const child = spawn(windowsShellCommand(backend.command, Boolean(backend.shell)), ['--print-runtime-command'], hiddenWindowsChildOptions({
      cwd: backend.root, env: { ...process.env, ...backend.env }, shell: Boolean(backend.shell), stdio: ['ignore', 'pipe', 'ignore']
    }))

    let output = ''
    const timer = setTimeout(() => child.kill('SIGKILL'), RUNTIME_COMMAND_TIMEOUT_MS)
    child.stdout!.setEncoding('utf8')
    child.stdout!.on('data', data => {
      output += String(data)

      if (output.length > 65536) { child.kill('SIGKILL') }
    })
    child.on('error', error => { clearTimeout(timer); fail(error.message) })
    child.on('close', code => {
      clearTimeout(timer)

      if (code !== 0) { fail(`launcher exited ${code ?? 'without a code'}; refresh this installation's launcher`);

 return }

      let command: unknown

      try { command = JSON.parse(output) } catch { fail('launcher did not report a runtime command');

 return }

      const python = Array.isArray(command) ? command[0] : undefined

      if (typeof python !== 'string' || !path.isAbsolute(python) || python.includes('\0') || !isFile(python)) {
        fail('launcher reported an invalid runtime command');

 return
      }

      resolve(python)
    })
  })
}

function isFile(candidate: string): boolean {
  try { return statSync(candidate).isFile() } catch { return false }
}

/**
 * The helper needs a bare interpreter. A Python backend is one; an installed
 * launcher backend names its own. The helper is stdlib-only, so the checkout on
 * PYTHONPATH is its whole import closure (-P still keeps the cwd out).
 */
export async function ticketRuntimeFor(backend: ResolvedTicketBackend): Promise<{ command: string; env?: NodeJS.ProcessEnv }> {
  if (backend.ticketCommand) {
    return { command: backend.ticketCommand, env: { ...backend.env, ...backend.ticketEnv } }
  }

  if (backend.kind === 'python' && !backend.shell) {
    return backend
  }

  if (backend.kind === 'command' && backend.root) {
    const python = await launcherRuntimePython({ ...backend, root: backend.root })

    return { command: python, env: { ...backend.env, PYTHONPATH: backend.root } }
  }

  throw new Error('Gateway ticket bootstrap requires the installed Hermes Python runtime')
}

export function createGatewayTicketResolver(resolveBackend: () => Promise<ResolvedTicketBackend>, cwd: () => string) {
  let cached: Promise<{ command: string; env?: NodeJS.ProcessEnv }> | undefined
  let cachedGeneration = -1

  return async (endpoint: TicketEndpoint, purpose: GatewayTicketPurpose): Promise<string> => {
    const generation = bridgeGeneration

    if (!cached || cachedGeneration !== generation) {
      cachedGeneration = generation

      const flight = resolveBackend().then(ticketRuntimeFor).catch(error => {
        if (cached === flight) { cached = undefined }
        throw error
      })

      cached = flight
    }

    const backend = await cached

    if (generation !== bridgeGeneration) { throw new Error('Gateway ticket runtime changed; retry connection') }

    return mintGatewayTicketWithPython(backend, cwd(), endpoint, purpose)
  }
}

export function mintGatewayTicketWithPython(
  backend: { command: string; env?: NodeJS.ProcessEnv },
  cwd: string,
  endpoint: TicketEndpoint,
  purpose: GatewayTicketPurpose
): Promise<string> {
  const env = { ...process.env, ...backend.env, HERMES_HOME: endpoint.profile_id,
    PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' }

  const key = JSON.stringify([backend.command, cwd, Object.entries(env).sort(([a], [b]) => a.localeCompare(b))])
  let bridge = bridges.get(key)

  if (!bridge) {
    bridge = new TicketBridge(backend.command, cwd, env, key)
    bridges.set(key, bridge)
  }

  return bridge.request(endpoint, purpose)
}
