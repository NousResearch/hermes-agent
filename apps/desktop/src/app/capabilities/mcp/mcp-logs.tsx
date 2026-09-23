import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { LogTail } from '@/components/chat/log-tail'
import { getLogs } from '@/hermes'
import { startCompletionPoll } from '@/lib/completion-poll'
import { $activeGatewayProfile } from '@/store/profile'

export const LOG_POLL_MS = 2000

const STDIO_MARKER_RE = /^===== \[.*\] starting MCP server '(.+)' =====$/

export function filterStdioSections(lines: string[], server: string): string[] {
  const out: string[] = []
  let inSection = false

  for (const line of lines) {
    const marker = STDIO_MARKER_RE.exec(line.trim())

    if (marker) {
      inSection = marker[1] === server
    }

    if (inSection) {
      out.push(line)
    }
  }

  return out
}

export type McpLogSource = 'agent' | 'stdio'

export function McpLogs({
  emptyLabel,
  server,
  source
}: {
  emptyLabel: string
  server: null | string
  source: McpLogSource
}) {
  const [lines, setLines] = useState<null | string[]>(null)
  const activeProfile = useStore($activeGatewayProfile)

  useEffect(() => {
    setLines(null)

    return startCompletionPoll({
      delayMs: LOG_POLL_MS,
      poll: async () => {
        const response =
          source === 'stdio'
            ? await getLogs({ file: 'mcp', lines: 500 })
            : await getLogs({ file: 'agent', lines: 300, search: server ?? 'mcp' })

        return source === 'stdio' && server ? filterStdioSections(response.lines, server) : response.lines
      },
      publish: setLines
    })
  }, [server, source, activeProfile])

  return <LogTail emptyLabel={emptyLabel} lines={lines} />
}
