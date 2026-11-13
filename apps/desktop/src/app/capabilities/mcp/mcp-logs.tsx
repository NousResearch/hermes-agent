import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { getLogs } from '@/hermes'
import { startCompletionPoll } from '@/lib/completion-poll'
import { $activeGatewayProfile } from '@/store/profile'

import { useSettingsOwner } from '../../settings/profile-scope'

export const LOG_POLL_MS = 2000

// Current banner: `<asctime> ===== starting …`; files written before it use `===== [<time>] starting …`.
const STDIO_MARKER_RE = /^(?:\d{4}-\d{2}-\d{2} [\d:,]+ )?===== (?:\[.*\] )?starting MCP server '(.+)' =====$/

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

export function useMcpLogLines(server: null | string, source: McpLogSource): null | string[] {
  const [lines, setLines] = useState<null | string[]>(null)
  const activeProfile = useStore($activeGatewayProfile)
  // An agent page tails its owner's logs, not the app's active profile's.
  const owner = useSettingsOwner()

  useEffect(() => {
    setLines(null)

    return startCompletionPoll({
      delayMs: LOG_POLL_MS,
      poll: async () => {
        const response =
          source === 'stdio'
            ? await getLogs({ file: 'mcp', lines: 500 }, owner)
            : await getLogs({ file: 'agent', lines: 300, search: server ?? 'mcp' }, owner)

        return source === 'stdio' && server ? filterStdioSections(response.lines, server) : response.lines
      },
      publish: setLines
    })
  }, [server, source, activeProfile, owner])

  return lines
}
