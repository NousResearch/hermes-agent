import { spawn } from 'node:child_process'

import { hiddenWindowsChildOptions } from './windows-child-options'

interface TicketEndpoint {
  profile_id: string
  instance_id: string
  runtime_protocol: number
  /** Multiplexer home whose control socket mints tickets for a served secondary. */
  control_home?: string | null
}

// Reuse the runtime's SID-validated, deadline-bounded pipe client. No Node pipe
// connection may bypass GetNamedPipeServerProcessId / token-owner validation.
const TICKET_SCRIPT = `
import json, sys
from pathlib import Path
from types import SimpleNamespace
from hermes_cli.gateway_client import _session_ticket
request = json.loads(sys.stdin.buffer.read(65537))
endpoint = SimpleNamespace(control_home=None, **request['endpoint'])
home = Path(endpoint.profile_id)
if str(home.resolve()) != endpoint.profile_id or endpoint.runtime_protocol != 1:
    raise ValueError('invalid ticket endpoint')
ticket = _session_ticket(home, endpoint, purpose=request['purpose'])
sys.stdout.write(json.dumps({'ticket': ticket}))
`

export function mintGatewayTicketWithPython(
  backend: { command: string; env?: NodeJS.ProcessEnv },
  cwd: string,
  endpoint: TicketEndpoint,
  purpose: 'interactive' | 'native-http'
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(backend.command, ['-P', '-c', TICKET_SCRIPT], hiddenWindowsChildOptions({
      cwd,
      env: { ...process.env, ...backend.env, HERMES_HOME: endpoint.profile_id, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      shell: false,
      stdio: ['pipe', 'pipe', 'pipe']
    }))

    let stdout = ''
    const fail = () => reject(new Error('Gateway ticket bootstrap failed'))
    const timer = setTimeout(() => { child.kill(); fail() }, 10_000)
    child.on('error', fail)
    child.stdin.on('error', fail)
    child.stderr.resume()
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', data => {
      stdout += data

      if (stdout.length > 65536) { child.kill(); fail() }
    })
    child.on('close', code => {
      clearTimeout(timer)

      if (code !== 0) {
        fail()

        return
      }

      try {
        const reply = JSON.parse(stdout)

        if (typeof reply.ticket !== 'string' || !reply.ticket) { throw new Error() }
        resolve(reply.ticket)
      } catch { fail() }
    })
    // Identity is data on private stdin, never interpolated into code or a shell.
    child.stdin.end(JSON.stringify({ endpoint, purpose }))
  })
}
