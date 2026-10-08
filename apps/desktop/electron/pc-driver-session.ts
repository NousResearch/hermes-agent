import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'

/** One MCP transport per conversation: element snapshots never cross sessions. */
export class PcDriverSession {
  private child: ChildProcessWithoutNullStreams
  private nextId = 0
  private buffer = ''
  private pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>()
  private closed = false
  readonly ready: Promise<void>

  constructor(command: string, args = ['mcp']) {
    this.child = spawn(command, args, {
      windowsHide: true,
      env: { ...process.env, CUA_DRIVER_RS_TELEMETRY_ENABLED: '0' }
    })
    this.child.stdout.setEncoding('utf8')
    this.child.stdout.on('data', chunk => this.read(String(chunk)))
    // Drain stderr without logging window contents or credentials.
    this.child.stderr.resume()
    this.child.on('error', () => this.close('PC driver failed to start.'))
    this.child.on('exit', () => this.close('PC driver exited; action outcome may be unknown. Do not replay.'))
    this.ready = this.request('initialize', {
      protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'hermes-desktop-device-bridge', version: '0.1.0' }
    }).then(() => { this.write({ jsonrpc: '2.0', method: 'notifications/initialized' }) })
  }

  private write(message: unknown) {
    if (this.closed) throw new Error('PC driver session is closed.')
    this.child.stdin.write(JSON.stringify(message) + '\n')
  }

  private read(chunk: string) {
    this.buffer += chunk
    if (Buffer.byteLength(this.buffer) > 16 * 1024 * 1024) {
      this.close('PC driver response exceeds the 16 MB limit. No replay.')
      return
    }
    for (;;) {
      const newline = this.buffer.indexOf('\n')
      if (newline < 0) return
      const line = this.buffer.slice(0, newline)
      this.buffer = this.buffer.slice(newline + 1)
      let message: any
      try { message = JSON.parse(line) } catch { this.close('Invalid PC driver response. No replay.'); return }
      if (message.method && message.id !== undefined) {
        // No implicit sampling, filesystem roots or authorization escalation.
        this.write({ jsonrpc: '2.0', id: message.id, error: { code: -32601, message: 'Host requests are not supported.' } })
        continue
      }
      const pending = this.pending.get(message.id)
      if (!pending) continue
      clearTimeout(pending.timer)
      this.pending.delete(message.id)
      if (message.error) pending.reject(new Error(String(message.error.message || 'PC driver error')))
      else pending.resolve(message.result)
    }
  }

  private request(method: string, params: unknown): Promise<any> {
    return new Promise((resolve, reject) => {
      if (this.closed) { reject(new Error('PC driver session is closed.')); return }
      const id = ++this.nextId
      const timer = setTimeout(() => this.close('PC driver timed out; action outcome is unknown. Inspect fresh state before acting again.'), 45000)
      this.pending.set(id, { resolve, reject, timer })
      try { this.write({ jsonrpc: '2.0', id, method, params }) }
      catch (error) { this.close(String(error)) }
    })
  }

  async call(name: string, args: Record<string, unknown>) {
    await this.ready
    return this.request('tools/call', { name, arguments: args })
  }

  close(reason = 'PC access revoked.') {
    if (this.closed) return
    this.closed = true
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer)
      pending.reject(new Error(reason))
    }
    this.pending.clear()
    this.child.stdin.end()
    // EOF is the driver's documented runtime teardown; kill only if it lingers.
    const timer = setTimeout(() => this.child.kill(), 1500)
    timer.unref()
  }
}
