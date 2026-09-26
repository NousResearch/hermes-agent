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

// Every `terminal(background=true)` mints a NEW process id, and nothing here ever
// forgot one: a finished process kept its backlog and last snapshot for the life of
// the renderer. Retention grew linearly with the number of commands ever run —
// replaying every process through `registerAgentTerminalWriter` measured 256K chars
// held per process (~512 KB, JS strings being UTF-16), so 100 -> 25.6M chars, 200 ->
// 51.2M, 400 -> 102.4M (~195 MB). One shape of the unbounded renderer growth in #77311.
//
// Bounded like the other renderer caches (`lib/lru-cache`, `chat/shiki-highlight-cache`):
// an entry ceiling AND a total-character ceiling, since one busy process can hold as
// much as twenty quiet ones. A process whose terminal is MOUNTED is never evicted —
// that is on screen, not a cache. Everything dropped is regenerable: the next
// `syncAgentTerminalSnapshot` re-seeds the tab from the registry's rolling tail.
const MAX_TRACKED_PROCS = 24
const MAX_TOTAL_CHARS = 2_000_000

/** Forget one process entirely. The four maps are evicted TOGETHER: `lastSnapshots`
 *  is the delta fence for `backlog`, so dropping one without the other would make the
 *  next snapshot diff against a tail that is no longer there. */
function forgetProc(procId: string): void {
  backlog.delete(procId)
  commandHeaders.delete(procId)
  lastSnapshots.delete(procId)
  seededCommands.delete(procId)
}

/** Drop the least-recently-written unmounted processes until both ceilings hold.
 *  `backlog` insertion order is the LRU clock (writers re-insert on every chunk). */
function evictColdProcs(): void {
  const total = () => {
    let chars = 0

    for (const [proc, text] of backlog) {
      chars += text.length + (lastSnapshots.get(proc)?.length ?? 0)
    }

    return chars
  }

  if (backlog.size <= MAX_TRACKED_PROCS && total() <= MAX_TOTAL_CHARS) {
    return
  }

  for (const proc of [...backlog.keys()]) {
    if (backlog.size <= MAX_TRACKED_PROCS && total() <= MAX_TOTAL_CHARS) {
      return
    }

    if (!writers.has(proc)) {
      forgetProc(proc)
    }
  }
}

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
  // delete-then-set moves this process to the tail: a plain re-set would keep its
  // original slot and make the oldest-first eviction below pick a live process.
  backlog.delete(procId)
  backlog.set(procId, next.length > MAX_BACKLOG ? next.slice(-MAX_BACKLOG) : next)
  writers.get(procId)?.(chunk)
  evictColdProcs()
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
    evictColdProcs()

    return
  }

  // The fence is updated BEFORE the write so the write's eviction pass counts it:
  // after, it would re-create a fence for a process that pass just forgot.
  if (output.startsWith(previous)) {
    lastSnapshots.set(procId, output)
    writeAgentTerminalChunk(procId, output.slice(previous.length))

    return
  }

  if (output.startsWith(body)) {
    lastSnapshots.set(procId, output)
    writeAgentTerminalChunk(procId, output.slice(body.length))

    return
  }

  // A reset is a write like any chunk: refresh recency and answer to the ceilings.
  const next = `${header}${output}`.slice(-MAX_BACKLOG)
  lastSnapshots.set(procId, output)
  backlog.delete(procId)
  backlog.set(procId, next)
  writers.get(procId)?.(`\x1bc${next}`)
  evictColdProcs()
}
