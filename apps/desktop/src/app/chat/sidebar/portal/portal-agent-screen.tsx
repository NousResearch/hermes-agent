import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { $lastRoster, botSelectionKey } from '@/plugins/hermes-bots/data'
import { useBots } from '@/plugins/hermes-bots/i18n'
import {
  type PortalTone,
  profileScreenCommand,
  screenBotForRoute,
  servesProfile,
  useScreenPortalState
} from '@/plugins/hermes-bots/screen-portal'
import type { RosterRow } from '@/plugins/hermes-bots/types'

import type { SimplifiedAgent } from '../use-simplified-agents'

interface AgentScreen {
  tone: PortalTone
  /** Where the screen stands, in the plugin's own words; none while that is not known or it cannot exist. */
  says?: string
  open: () => void
}

/** The roster row that owns an agent's Screen: the one the plugin's Open Screen action resolves
 *  (its own matcher), so what a surface says about the Screen is about the one it opens. The row
 *  object holds while it stands for the same bot: the plugin's status hook reads again whenever the
 *  object it is given changes, and the roster is republished often. */
export function useScreenBot(connectionId: null | string, profile: string): RosterRow {
  const resolved = screenBotForRoute({ connectionId, profile }, useStore($lastRoster), servesProfile)
  const [bot, setBot] = useState(resolved)

  if (botSelectionKey(bot) !== botSelectionKey(resolved)) {
    setBot(resolved)
  }

  return bot
}

/** One agent's Screen (Bot Mode: its own desktop), from the plugin's state for it. Mounting this
 *  reads that agent's screen status once, then follows pushed events: call it only for an agent of
 *  the window's own machine, and only while Bot Mode is loaded. */
export function useAgentScreen(connectionId: null | string, profile: string): AgentScreen {
  const b = useBots()
  const { tone } = useScreenPortalState(useScreenBot(connectionId, profile))

  const says: Partial<Record<PortalTone, string>> = {
    live: b.screen.portalWatching,
    human: b.screen.portalYouControl,
    other: b.screen.portalOtherControls,
    off: b.screen.portalStopped,
    missing: b.screen.portalNotInstalled
  }

  return { tone, says: says[tone], open: () => void profileScreenCommand({ connectionId, profile }).run() }
}

/** The row's mark, in a chat row's own states: the agent at work on its screen, or someone holding it. */
const MARK: Partial<Record<PortalTone, 'needs-input' | 'working'>> = {
  live: 'working',
  human: 'needs-input',
  other: 'needs-input'
}

/** An agent's live Screen as a row under its card while the screen is running: who holds it, and a
 *  click opens it. No row otherwise; the agent's menu opens a Screen that is stopped or not
 *  installed. */
export function AgentScreenRow({ agent }: { agent: SimplifiedAgent }) {
  const b = useBots()
  const { tone, says, open } = useAgentScreen(agent.connectionId, agent.profile)
  const mark = MARK[tone]

  if (!mark) {
    return null
  }

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
          <RowButton className="portal-session-open" onClick={open}>
            <span className="portal-session-title">{`${b.screen.portalTitle} · ${says}`}</span>
          </RowButton>
        </div>
      </div>
    </div>
  )
}
