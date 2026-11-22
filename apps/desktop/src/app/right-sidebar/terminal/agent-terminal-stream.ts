// Live agent-terminal output, pushed from the backend as `agent.terminal.output`
// events (see tui_gateway `_wire_agent_terminal_output`). Chunks route straight
// to the matching read-only xterm, keyed by process id — no polling, no tail
// truncation. A capped per-proc backlog lets a tab opened mid-stream replay what
// it missed, and lets a closed-then-reopened tab restore its history.

type Writer = (chunk: string) => void

const writers = new Map<string, Writer>()

/**
 * Capped backlog kept as a chunk deque, not one string.
 *
 * `(backlog + chunk).slice(-MAX)` copied the whole 256K tail twice for EVERY
 * output event once the cap was reached — a busy agent terminal allocates
 * hundreds of KB per few-byte chunk. Chunks are appended by reference and
 * trimmed by dropping whole entries from the head, so a chunk write is O(1)
 * apart from the rare compaction. The string is materialized only for the
 * readers that need it (mount replay, snapshot diffing), which run per tab
 * open / poll rather than per output event.
 */
interface BacklogEntry {
  chunks: string[]
  /** Dead prefix: chunks before this index have been evicted. */
  head: number
  length: number
}

const backlog = new Map<string, BacklogEntry>()
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

/** Materialize the backlog and keep it as the entry's only chunk. A separate
 *  memo would hold a second full copy beside the chunks that the character
 *  ceiling never counts; collapsing keeps one copy and makes repeat reads free. */
function backlogText(entry: BacklogEntry): string {
  if (entry.head !== 0 || entry.chunks.length !== 1) {
    entry.chunks = [entry.chunks.slice(entry.head).join('')]
    entry.head = 0
  }

  return entry.chunks[0]
}

/** Recency order is the Map's insertion order; callers re-insert on every write. */
function setBacklog(procId: string, text: string): void {
  backlog.delete(procId)
  backlog.set(procId, { chunks: [text], head: 0, length: text.length })
}

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
    write(backlogText(history))
  }

  return () => {
    if (writers.get(procId) === write) {
      writers.delete(procId)
      evictColdProcs()
    }
  }
}

/** Append a streamed chunk: buffer it (capped) for future opens and write it to
 *  the live terminal, if one is mounted. */
export function writeAgentTerminalChunk(procId: string, chunk: string): void {
  if (!procId || !chunk) {
    return
  }

  const entry = backlog.get(procId) ?? { chunks: [], head: 0, length: 0 }
  entry.chunks.push(chunk)
  entry.length += chunk.length

  // Evict whole chunks from the head while the remainder still covers the cap.
  // A single oversized chunk is the only case that needs a copy.
  while (entry.chunks.length - entry.head > 1 && entry.length - entry.chunks[entry.head].length >= MAX_BACKLOG) {
    entry.length -= entry.chunks[entry.head].length
    // Release the evicted string now; keeping the dead slot alive until the
    // 1k-entry compaction would hide up to head-many chunks from the character
    // budget that is supposed to bound this cache.
    entry.chunks[entry.head] = ''
    entry.head += 1
  }

  if (entry.head > 1024 && entry.head * 2 > entry.chunks.length) {
    entry.chunks = entry.chunks.slice(entry.head)
    entry.head = 0
  }

  // The loop above only drops whole chunks, so the retained tail can exceed
  // the cap by less than one head chunk. Trim that overhang from the oldest
  // chunk (newest characters stay). This is a bounded slice, never a join; a
  // single oversized chunk lands here too.
  const overhang = entry.length - MAX_BACKLOG

  if (overhang > 0) {
    entry.chunks[entry.head] = entry.chunks[entry.head].slice(overhang)
    entry.length = MAX_BACKLOG
  }

  // delete-then-set moves this process to the tail: a plain re-set would keep its
  // original slot and make the oldest-first eviction below pick a live process.
  backlog.delete(procId)
  backlog.set(procId, entry)
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

  const entry = backlog.get(procId)
  const current = entry ? backlogText(entry) : ''
  const header = commandHeaders.get(procId) ?? ''
  const body = header && current.startsWith(header) ? current.slice(header.length) : current
  const previous = lastSnapshots.get(procId) ?? ''

  if (output === previous || output === body || body.endsWith(output)) {
    lastSnapshots.set(procId, output)
    evictColdProcs()

    return
  }

  // The live backlog (body) is the authoritative on-screen state and is always at
  // least as long as the last snapshot fence, so check its prefix-delta FIRST: the
  // fence-first order would re-append bytes already streamed by live chunks and
  // print them twice. The fence is updated BEFORE the write so the write's eviction
  // pass counts it: after, it would re-create a fence for a process that pass just
  // forgot.
  if (output.startsWith(body)) {
    lastSnapshots.set(procId, output)
    writeAgentTerminalChunk(procId, output.slice(body.length))

    return
  }

  if (output.startsWith(previous)) {
    lastSnapshots.set(procId, output)
    writeAgentTerminalChunk(procId, output.slice(previous.length))

    return
  }

  // A reset is a write like any chunk: refresh recency and answer to the ceilings.
  const next = `${header}${output}`.slice(-MAX_BACKLOG)
  lastSnapshots.set(procId, output)
  setBacklog(procId, next)
  writers.get(procId)?.(`\x1bc${next}`)
  evictColdProcs()
}
