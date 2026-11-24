import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { $lastRoster, botSelectionKey } from '@/plugins/hermes-bots/data'
import { botsText, useBots } from '@/plugins/hermes-bots/i18n'
import {
  type PortalTone,
  profileScreenCommand,
  screenBotForRoute,
  screenTonePhrase,
  servesProfile,
  useScreenPortalState
} from '@/plugins/hermes-bots/screen-portal'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { notifyError } from '@/store/notifications'
import { $gatewayState } from '@/store/session'

import type { SimplifiedAgent } from '../use-simplified-agents'

interface AgentScreen {
  tone: PortalTone
  /** Where the screen stands, in the plugin's own words; none where it cannot exist. */
  says?: string
  /** For a running screen, the short word for who holds it. */
  holder?: string
  /** Opens the Screen; resolves to whether it did (a failure is reported, not thrown). */
  open: () => Promise<boolean>
}

/** Open an agent's Screen through the plugin's own action for a machine and profile. That action
 *  looks the agent's route up first, which can fail (an agent that is gone, a machine that does not
 *  answer): the failure is said, and the caller learns nothing opened. */
export async function openAgentScreen(connectionId: null | string, profile: string): Promise<boolean> {
  try {
    await profileScreenCommand({ connectionId, profile }).run()

    return true
  } catch (error) {
    notifyError(error, botsText().screen.menu)

    return false
  }
}

/** The roster row that owns an agent's Screen: the one the plugin's Open Screen action resolves
 *  (its own matcher), so what a surface says about the Screen is about the one it opens.
 *
 *  The plugin's status hook reads once for the row object it is given and again whenever that
 *  object changes. The roster is republished often, so the object holds while it stands for the
 *  same bot; it is renewed when the gateway's state changes, because a read made before the
 *  gateway was open fails and nothing else would ask again. */
export function useScreenBot(connectionId: null | string, profile: string): RosterRow {
  const resolved = screenBotForRoute({ connectionId, profile }, useStore($lastRoster), servesProfile)
  const gateway = useStore($gatewayState)
  const [held, setHeld] = useState({ bot: resolved, gateway })

  if (botSelectionKey(held.bot) !== botSelectionKey(resolved) || held.gateway !== gateway) {
    setHeld({ bot: resolved, gateway })
  }

  return held.bot
}

/** One agent's Screen (Bot Mode: its own desktop), from the plugin's state for it. Mounting this
 *  reads that agent's screen status once, then follows pushed events: call it only for an agent of
 *  the window's own machine, and only while Bot Mode is loaded. */
export function useAgentScreen(connectionId: null | string, profile: string): AgentScreen {
  const b = useBots()
  const { tone } = useScreenPortalState(useScreenBot(connectionId, profile))

  const { phrase, holder } = screenTonePhrase(b, tone)

  return { tone, says: phrase, holder, open: () => openAgentScreen(connectionId, profile) }
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
  const { tone, says, holder, open } = useAgentScreen(agent.connectionId, agent.profile)
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
          {/* The mark says agent or person; the meta names which, in a word. The phrase is the name. */}
          <RowButton
            aria-label={`${b.screen.portalTitle}: ${says}`}
            className="portal-session-open"
            onClick={() => void open()}
          >
            <span className="portal-session-title">{b.screen.portalTitle}</span>
          </RowButton>
          <span aria-hidden className="portal-session-meta">
            {holder}
          </span>
        </div>
      </div>
    </div>
  )
}
