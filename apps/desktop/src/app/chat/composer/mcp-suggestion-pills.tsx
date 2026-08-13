import { useState } from 'react'

import { composerFloatingPill } from '@/components/chat/composer-dock'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { addMcpServer, authMcpServer, getMcpOAuthFlow, removeMcpServer } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { brandFor, brandGlyphStyle } from '@/lib/mcp-brands'
import { completeMcpDesktopOAuth } from '@/lib/mcp-dashboard-oauth'
import { directoryEntry } from '@/lib/mcp-directory'
import { prettyName } from '@/lib/text'
import { useSessionSlice } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import { $mcpSuggestionsBySession, invalidateMcpSuggestionIndex } from '@/store/mcp-suggestions'
import { notifyError } from '@/store/notifications'

/**
 * Keyword-triggered MCP suggestion pills — the "you typed jira, want
 * Atlassian?" strip. Renders beside the micro-action badges in the floating
 * lane above the composer, same pill treatment. Session-scoped like the
 * badges: each composer shows only the pills its own draft earned.
 *
 * A click CONNECTS, right here — every directory entry is a hosted OAuth
 * remote, so the whole install is one validated config write plus the
 * browser OAuth round-trip. The pill narrates it: label → "Connecting…"
 * (click again to cancel) → "Added". No composer indirection, no agent
 * turn — the click is the consent. The `setup_mcp` transcript card remains
 * the AGENT-initiated path; both run the same flow and the same rollback:
 * a cancelled/failed connect removes the config entry it just wrote.
 *
 * No dismiss affordance ON PURPOSE. The pills are self-limiting — they only
 * exist while a trigger word/link is in the draft, vanish once the server is
 * configured, and cap at two — so a close button would mostly collect
 * accidental permanent opt-outs. The escape hatch is simply not clicking.
 *
 * Same pointer-events rule as the micro-action pills: NEVER
 * `pointer-events-none` — the pop-out drag region sits behind this strip.
 */

type PillPhase = 'done' | 'idle' | 'working'

export function McpSuggestionPills({ sessionId }: { sessionId: null | string }) {
  const { t } = useI18n()
  const copy = t.composer.mcpSuggestions
  const suggestions = useSessionSlice($mcpSuggestionsBySession, sessionId)
  const [phases, setPhases] = useState<Record<string, PillPhase>>({})
  // Cancel flags outlive renders but never trigger them (poll-boundary abort).
  const [cancels] = useState(() => new Map<string, boolean>())

  const setPhase = (server: string, phase: PillPhase) =>
    setPhases(current => ({ ...current, [server]: phase }))

  const connect = async (server: string) => {
    const known = directoryEntry(server)

    if (!known) {
      return
    }

    cancels.set(server, false)
    setPhase(server, 'working')
    triggerHaptic('selection')

    try {
      await addMcpServer({ name: known.name, url: known.url })

      try {
        await completeMcpDesktopOAuth({
          serverName: known.name,
          start: authMcpServer,
          status: async flowId => {
            const flow = await getMcpOAuthFlow(flowId)

            if (cancels.get(server)) {
              throw CANCELLED
            }

            return flow
          },
          openExternal: url => window.hermesDesktop.openExternal(url)
        })
      } catch (error) {
        // Decline/failure means "no server" — roll back the config write
        // rather than stranding an unauthorized entry (authoritative-write
        // rule). Best-effort; the primary error wins.
        await removeMcpServer(known.name).catch(() => {})
        throw error
      }

      // Tools reach the live session before the pill claims success — the
      // same write-through the Capabilities tab and the setup card use.
      await $gateway
        .get()
        ?.request('reload.mcp', { confirm: true, session_id: sessionId ?? undefined })
        .catch(() => {})

      invalidateMcpSuggestionIndex()
      triggerHaptic('submit')
      setPhase(server, 'done')
    } catch (error) {
      setPhase(server, 'idle')

      if (error !== CANCELLED) {
        notifyError(error, copy.connectFailed(prettyName(server)))
      }
    }
  }

  return suggestions.map(suggestion => {
    const brand = brandFor(suggestion.server)
    const phase = phases[suggestion.server] ?? 'idle'
    const name = prettyName(suggestion.server)

    const label = phase === 'working' ? copy.connecting(name) : phase === 'done' ? copy.added(name) : copy.label(name)

    const tip = phase === 'working' ? copy.cancelTip : phase === 'done' ? copy.addedTip : copy.tip(suggestion.keyword)

    return (
      <Tip key={suggestion.server} label={tip}>
        <button
          className={cn(composerFloatingPill, 'max-w-56', phase === 'done' && 'cursor-default')}
          onClick={() => {
            if (phase === 'working') {
              // Second click cancels a stuck flow (closed OAuth tab, etc.).
              cancels.set(suggestion.server, true)
            } else if (phase === 'idle') {
              void connect(suggestion.server)
            }
          }}
          type="button"
        >
          {phase === 'working' ? (
            <Codicon className="shrink-0 opacity-70" name="loading" size="0.75rem" spinning />
          ) : phase === 'done' ? (
            <Codicon className="shrink-0 text-emerald-400" name="check" size="0.75rem" />
          ) : brand ? (
            <brand.Icon aria-hidden className="size-3 shrink-0" style={brandGlyphStyle(brand)} />
          ) : (
            <Codicon className="shrink-0 opacity-70" name="plug" size="0.75rem" />
          )}
          <span className="truncate">{label}</span>
        </button>
      </Tip>
    )
  })
}

// Thrown by the poll wrapper when the user cancels — the rollback has its own
// path, so the catch must swallow this rather than toast it.
const CANCELLED = Symbol('mcp-pill-cancelled')
