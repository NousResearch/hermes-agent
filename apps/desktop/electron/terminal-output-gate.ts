export interface TerminalExitPayload {
  code: number | null
  signal: string | null
}

interface TerminalOutputGateOptions {
  onExitFlushed: () => void
  sendData: (data: string) => void
  sendExit: (payload: TerminalExitPayload) => void
}

const MAX_PENDING_OUTPUT = 1024 * 1024

// Attached-path coalescing: node-pty emits many small chunks under fast
// output (`ls -R`, `cat`), and every webContents.send is a cross-process
// post. Buffer up to one frame (or 4KB, whichever comes first) instead of
// one IPC per chunk.
const ATTACHED_FLUSH_MS = 16
const ATTACHED_FLUSH_BYTES = 4096

export function createTerminalOutputGate({ onExitFlushed, sendData, sendExit }: TerminalOutputGateOptions) {
  let attached = false
  let disposed = false
  // Pre-attach backlog, bounded. Chunks are kept as a list and trimmed from
  // the front — never `${pending}${chunk}`.slice(), which re-copies the whole
  // backlog per chunk (O(n²) under a fast pre-attach burst).
  let pendingChunks: string[] = []
  let pendingBytes = 0
  let attachedData = ''
  let attachedBytes = 0
  let flushTimer: ReturnType<typeof setTimeout> | null = null
  let pendingExit: TerminalExitPayload | null = null

  const clearFlushTimer = () => {
    if (flushTimer !== null) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
  }

  const flushAttachedData = () => {
    clearFlushTimer()

    if (attachedData) {
      const data = attachedData
      attachedData = ''
      attachedBytes = 0
      sendData(data)
    }
  }

  const flushExit = (payload: TerminalExitPayload) => {
    // Buffered output precedes the exit, in order.
    flushAttachedData()
    sendExit(payload)
    onExitFlushed()
  }

  const scheduleFlush = () => {
    if (flushTimer !== null || disposed) {
      return
    }

    flushTimer = setTimeout(() => {
      flushTimer = null

      if (!disposed) {
        flushAttachedData()
      }
    }, ATTACHED_FLUSH_MS)

    if (typeof flushTimer.unref === 'function') {
      flushTimer.unref()
    }
  }

  return {
    attach(): void {
      if (attached) {
        return
      }

      attached = true

      if (pendingChunks.length > 0) {
        sendData(pendingChunks.join(''))
        pendingChunks = []
        pendingBytes = 0
      }

      if (pendingExit) {
        const payload = pendingExit
        pendingExit = null
        flushExit(payload)
      }
    },
    data(chunk: string): void {
      if (disposed) {
        return
      }

      if (!attached) {
        pendingChunks.push(chunk)
        pendingBytes += chunk.length

        while (pendingBytes > MAX_PENDING_OUTPUT && pendingChunks.length > 1) {
          pendingBytes -= pendingChunks.shift()!.length
        }

        if (pendingBytes > MAX_PENDING_OUTPUT) {
          const keep = pendingChunks[0].slice(-MAX_PENDING_OUTPUT)
          pendingChunks = [keep]
          pendingBytes = keep.length
        }

        return
      }

      attachedData += chunk
      attachedBytes += chunk.length

      if (attachedBytes >= ATTACHED_FLUSH_BYTES) {
        flushAttachedData()
      } else {
        scheduleFlush()
      }
    },
    exit(payload: TerminalExitPayload): void {
      if (disposed) {
        return
      }

      if (attached) {
        flushExit(payload)

        return
      }

      pendingExit = payload
    },
    /** Flush coalesced output now (tests; terminal teardown). */
    flush(): void {
      flushAttachedData()
    },
    dispose(): void {
      disposed = true
      clearFlushTimer()
      pendingChunks = []
      pendingBytes = 0
      attachedData = ''
      attachedBytes = 0
      pendingExit = null
    }
  }
}
