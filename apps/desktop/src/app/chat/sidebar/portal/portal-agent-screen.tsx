import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { $lastRoster } from '@/plugins/hermes-bots/data'
import { useBots } from '@/plugins/hermes-bots/i18n'
import {
  type PortalTone,
  profileScreenCommand,
  screenBotForRoute,
  useScreenPortalState
} from '@/plugins/hermes-bots/screen-portal'

import type { SimplifiedAgent } from '../use-simplified-agents'

/** The row's mark, in a chat row's own states: the agent at work on its screen, or someone holding it. */
const MARK: Partial<Record<PortalTone, 'needs-input' | 'working'>> = {
  live: 'working',
  human: 'needs-input',
  other: 'needs-input'
}

/** An agent's live Screen (Bot Mode: its own desktop), as a row under its card while the screen is
 *  running: who holds it, and a click opens it. No row otherwise; the agent's menu opens a Screen
 *  that is stopped or not installed. Mounting it reads that agent's screen status once, so the
 *  sidebar mounts it only for the agents of the window's own machine, and only while Bot Mode is
 *  loaded. */
export function AgentScreenRow({ agent }: { agent: SimplifiedAgent }) {
  const b = useBots()
  const roster = useStore($lastRoster)
  const { connectionId, profile } = agent

  // The same row object across renders: the status read and the event listeners key on it.
  const bot = useMemo(
    () => screenBotForRoute({ connectionId, profile }, roster, (served, name) => served.profile === name),
    [connectionId, profile, roster]
  )

  const { tone } = useScreenPortalState(bot)
  const mark = MARK[tone]

  if (!mark) {
    return null
  }

  const holder = {
    live: b.screen.portalWatching,
    human: b.screen.portalYouControl,
    other: b.screen.portalOtherControls
  }[tone as 'human' | 'live' | 'other']

  return (
    <div className="portal-agent-sessions portal-agent-screen">
      <div className="portal-session-item">
        <div className="portal-session-row">
          <span aria-hidden className="portal-row-mark" data-state={mark}>
            <span>
              <span />
            </span>
          </span>
          {/* One line that gives way at its end: who holds the screen is a phrase, not a chat's age. */}
          <RowButton
            className="portal-session-open"
            onClick={() => void profileScreenCommand({ connectionId, profile }).run()}
          >
            <span className="portal-session-title">{`${b.screen.portalTitle} · ${holder}`}</span>
          </RowButton>
        </div>
      </div>
    </div>
  )
}
