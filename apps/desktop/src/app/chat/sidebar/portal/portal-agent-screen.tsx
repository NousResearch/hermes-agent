import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { $lastRoster } from '@/plugins/hermes-bots/data'
import { botsText, useBots } from '@/plugins/hermes-bots/i18n'
import { ScreenHero } from '@/plugins/hermes-bots/screen-hero'
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

/** The roster row an agent's Screen surfaces stand on: its own row, else the alias that serves
 *  its profile (an alias's screen is its profile's), else a row made up for a machine and profile
 *  the roster does not list. Rebuilt with the roster, as classic's own Screen box does: the plugin's
 *  status hook then asks again for an agent whose status has not settled, which is what recovers a
 *  read made too early. */
export function useScreenBot(connectionId: null | string, profile: string): RosterRow {
  const roster = useStore($lastRoster)

  return useMemo(() => {
    const own = screenBotForRoute({ connectionId, profile }, roster, (route, name) => route.profile === name)

    return roster.includes(own) ? own : screenBotForRoute({ connectionId, profile }, roster, servesProfile)
  }, [connectionId, profile, roster])
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

/** The agent's screen itself, while it is running: the plugin's live thumbnail (a frame every few
 *  seconds while it is on screen, none once it is gone). Nothing for a screen that is stopped, not
 *  installed or not known: the Screen row says that. The thumbnail opens the Screen as a tab beside
 *  the chat, so whatever page holds it steps aside. */
export function AgentScreenThumbnail({
  connectionId,
  profile,
  onOpened
}: {
  connectionId: null | string
  profile: string
  onOpened: () => void
}) {
  const bot = useScreenBot(connectionId, profile)
  const { tone } = useScreenPortalState(bot)

  if (tone !== 'live' && tone !== 'human' && tone !== 'other') {
    return null
  }

  return (
    // The thumbnail is the control; this only hears its click on the way up.
    <div className="portal-agent-screen-thumbnail" onClick={onOpened} role="presentation">
      <ScreenHero bot={bot} />
    </div>
  )
}
