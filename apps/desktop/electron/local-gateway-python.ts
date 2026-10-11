import { type ChildProcessWithoutNullStreams, spawn } from 'node:child_process'

import { hiddenWindowsChildOptions } from './windows-child-options'

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

  constructor(command: string, cwd: string, env: NodeJS.ProcessEnv, private key: string) {
    this.child = spawn(command, ['-P', '-u', '-c', TICKET_SCRIPT], hiddenWindowsChildOptions({
      cwd, env, shell: false, stdio: ['pipe', 'pipe', 'pipe']
    }))
    this.child.stderr.resume()
    this.child.stdout.setEncoding('utf8')
    this.child.stdout.on('data', data => this.receive(String(data)))
    this.child.on('error', () => this.dispose())
    this.child.stdin.on('error', () => this.dispose())
    this.child.on('close', () => this.dispose())
  }

  request(endpoint: TicketEndpoint, purpose: 'interactive' | 'native-http'): Promise<string> {
    clearTimeout(this.idleTimer)

    return new Promise((resolve, reject) => {
      if (this.closed) { reject(new Error('Gateway ticket bootstrap failed'));

 return }

      const id = ++this.sequence
      const request = JSON.stringify({ id, endpoint, purpose }) + '\n'

      if (Buffer.byteLength(request) > 65536) { reject(new Error('Gateway ticket request too large'));

 return }

      const timer = setTimeout(() => this.dispose(), 10_000)
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
        else { waiter.reject(Object.assign(new Error('Gateway ticket bootstrap failed'),
          typeof reply.error === 'string' ? { reason: reply.error } : {})) }
      } catch { this.dispose();

 return }
    }

    if (!this.pending.size) { this.idleTimer = setTimeout(() => this.dispose(), 30_000) }
  }

  dispose(): void {
    if (this.closed) { return }
    this.closed = true
    clearTimeout(this.idleTimer)

    if (bridges.get(this.key) === this) { bridges.delete(this.key) }

    for (const waiter of this.pending.values()) {
      clearTimeout(waiter.timer)
      waiter.reject(new Error('Gateway ticket bootstrap failed'))
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
}

export function createGatewayTicketResolver(resolveBackend: () => Promise<ResolvedTicketBackend>, cwd: () => string) {
  let cached: Promise<ResolvedTicketBackend> | undefined
  let cachedGeneration = -1

  return async (endpoint: TicketEndpoint, purpose: 'interactive' | 'native-http'): Promise<string> => {
    const generation = bridgeGeneration

    if (!cached || cachedGeneration !== generation) {
      cachedGeneration = generation

      const flight = resolveBackend().then(backend => {
        if (backend.kind !== 'python' || backend.shell) {
          throw new Error('Gateway ticket bootstrap requires the installed Hermes Python runtime')
        }

        return backend
      }).catch(error => {
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
  purpose: 'interactive' | 'native-http'
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
