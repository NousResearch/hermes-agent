// Live agent-terminal output, pushed from the backend as `agent.terminal.output`
// events (see tui_gateway `_wire_agent_terminal_output`). Chunks route straight
// to the matching read-only xterm, keyed by process id — no polling, no tail
// truncation. A capped per-proc backlog lets a tab opened mid-stream replay what
// it missed, and lets a closed-then-reopened tab restore its history.

type Writer = (chunk: string) => void

const writers = new Map<string, Writer>()
const backlog = new Map<string, string>()
const commandHeaders = new Map<string, string>()
const lastSnapshots = new Map<string, string>()
const seededCommands = new Set<string>()

const MAX_BACKLOG = 256_000

/** A live agent terminal registers its xterm write and replays the backlog.
 *  Returns an idempotent unregister. */
export function registerAgentTerminalWriter(procId: string, write: Writer): () => void {
  writers.set(procId, write)

  const history = backlog.get(procId)

  if (history) {
    write(history)
  }

  return () => {
    if (writers.get(procId) === write) {
      writers.delete(procId)
    }
  }
}

/** Append a streamed chunk: buffer it (capped) for future opens and write it to
 *  the live terminal, if one is mounted. */
export function writeAgentTerminalChunk(procId: string, chunk: string): void {
  if (!procId || !chunk) {
    return
  }

  const next = (backlog.get(procId) ?? '') + chunk
  backlog.set(procId, next.length > MAX_BACKLOG ? next.slice(-MAX_BACKLOG) : next)
  writers.get(procId)?.(chunk)
}

/** Seed the tab with the command immediately, so an agent terminal never opens
 *  as an empty void while stdout is still pending or not yet observed. */
export function seedAgentTerminalCommand(procId: string, command: string): void {
  const trimmed = command.trim()

  if (!procId || !trimmed || seededCommands.has(procId)) {
    return
  }

  seededCommands.add(procId)
  const header = `$ ${trimmed}\r\n`
  commandHeaders.set(procId, header)
  writeAgentTerminalChunk(procId, header)
}

/** Ingest a full output snapshot from process.list/status-stack. This is the
 *  fallback for older/not-yet-restarted gateways and a seed for tabs opened
 *  after output already exists. If it extends our current backlog, append only
 *  the delta; if the registry's rolling tail slid, reset to that tail. */
export function syncAgentTerminalSnapshot(procId: string, output: string): void {
  if (!procId || !output) {
    return
  }

  const current = backlog.get(procId) ?? ''
  const header = commandHeaders.get(procId) ?? ''
  const body = header && current.startsWith(header) ? current.slice(header.length) : current
  const previous = lastSnapshots.get(procId) ?? ''

  if (output === previous || output === body || body.endsWith(output)) {
    lastSnapshots.set(procId, output)

    return
  }

  if (output.startsWith(body)) {
    writeAgentTerminalChunk(procId, output.slice(body.length))
    lastSnapshots.set(procId, output)

    return
  }

  if (output.startsWith(previous)) {
    writeAgentTerminalChunk(procId, output.slice(previous.length))
    lastSnapshots.set(procId, output)

    return
  }

  const next = `${header}${output}`.slice(-MAX_BACKLOG)
  lastSnapshots.set(procId, output)
  backlog.set(procId, next)
  writers.get(procId)?.(`\x1bc${next}`)
}
