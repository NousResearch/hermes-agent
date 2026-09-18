// The MCP output channel — Cursor's "MCP Logs" equivalent.
//
// Extracted from `mcp-tab.tsx` so the Connectors page's `Advanced` section can
// pin the same pane under a single server without a second poller.

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { LogTail } from '@/components/chat/log-tail'
import { getLogs } from '@/hermes'
import { startCompletionPoll } from '@/lib/completion-poll'
import { $activeGatewayProfile } from '@/store/profile'

export const LOG_POLL_MS = 2000

const STDIO_MARKER_RE = /^===== \[.*\] starting MCP server '(.+)' =====$/

// Keep only the stdio-log sections belonging to one server. The shared file
// has no per-line tags — sections start at that server's session marker and
// run until the next marker (any server's).
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

/** Scope follows the caller's selected server (all servers otherwise); the
 *  source control lives in whichever pane header hosts this. Body is the app's
 *  tool-output surface: CodeCardBody typography + hover-reveal copy. */
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
  // A profile switch reroutes getLogs to the new backend; keying the effect on
  // the active profile tears down the old poll (stop suppresses a late
  // publish) so profile A's logs never flash in B.
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
