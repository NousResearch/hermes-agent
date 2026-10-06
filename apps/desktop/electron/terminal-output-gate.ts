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

export function createTerminalOutputGate({ onExitFlushed, sendData, sendExit }: TerminalOutputGateOptions) {
  let attached = false
  let pendingData = ''
  let pendingExit: TerminalExitPayload | null = null

  const flushExit = (payload: TerminalExitPayload) => {
    sendExit(payload)
    onExitFlushed()
  }

  return {
    attach(): void {
      if (attached) {
        return
      }

      attached = true

      if (pendingData) {
        sendData(pendingData)
        pendingData = ''
      }

      if (pendingExit) {
        const payload = pendingExit
        pendingExit = null
        flushExit(payload)
      }
    },
    data(chunk: string): void {
      if (attached) {
        sendData(chunk)

        return
      }

      pendingData = `${pendingData}${chunk}`.slice(-MAX_PENDING_OUTPUT)
    },
    exit(payload: TerminalExitPayload): void {
      if (attached) {
        flushExit(payload)

        return
      }

      pendingExit = payload
    }
  }
}
