import { host } from '@hermes/plugin-sdk'
import { useStore } from '@nanostores/react'
import { useMemo, useRef } from 'react'

import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { RowButton } from '@/components/ui/row-button'
import { $lastRoster } from '@/plugins/hermes-bots/data'
import { botsText, useBots } from '@/plugins/hermes-bots/i18n'
import { ScreenHero } from '@/plugins/hermes-bots/screen-hero'
import { openBotScreen } from '@/plugins/hermes-bots/screen-open'
import {
  type PortalTone,
  screenBotOwnFirst,
  screenTonePhrase,
  useScreenPortalState
} from '@/plugins/hermes-bots/screen-portal'
import type { RosterRow } from '@/plugins/hermes-bots/types'
import { notifyError } from '@/store/notifications'

import { useRestingHover } from '../use-resting-hover'
import type { SimplifiedAgent } from '../use-simplified-agents'

interface AgentScreen {
  /** The roster row the screen's state is read for, and the one opening it opens. */
  bot: RosterRow
  tone: PortalTone
  /** Where the screen stands, in the plugin's own words; none where it cannot exist. */
  says?: string
  /** For a running screen, the short word for who holds it. */
  holder?: string
  /** Opens the Screen; resolves to whether it did (a failure is reported, not thrown). */
  open: () => Promise<boolean>
}

/** Whether this app can open a Screen at all (an older Desktop cannot; the plugin then says so). */
const screenOpens = () => typeof host.openWorkspace === 'function'

/** Open an agent's Screen: the plugin's Open Screen for that agent's own row, under the profile its
 *  machine serves it from. Looking that up can fail (an agent that is gone, a machine that does not
 *  answer): the failure is said, and the caller learns nothing opened. */
export async function openAgentScreen(connectionId: null | string, profile: string): Promise<boolean> {
  try {
    const machine = connectionId ?? 'local'
    const served = (await host.profileRoutes()).find(route => route.connectionId === machine && route.profile === profile)
    const bot = screenBotOwnFirst({ connectionId, profile }, $lastRoster.get())

    openBotScreen(served ? { ...bot, targetProfile: served.targetProfile || served.profile } : bot, null)

    return screenOpens()
  } catch (error) {
    notifyError(error, botsText().screen.menu)

    return false
  }
}

/** The roster row an agent's Screen stands on (the plugin's own-row-first resolver, which its
 *  palette action uses too), rebuilt with the roster, as classic's own Screen box does: the
 *  plugin's status hook then asks again for an agent whose status has not settled, which is what
 *  recovers a read made too early. */
export function useScreenBot(connectionId: null | string, profile: string): RosterRow {
  const roster = useStore($lastRoster)

  return useMemo(() => screenBotOwnFirst({ connectionId, profile }, roster), [connectionId, profile, roster])
}

/** One agent's Screen (Bot Mode: its own desktop), from the plugin's state for it. Mounting this
 *  reads that agent's screen status once, then follows pushed events: call it only for an agent of
 *  the window's own machine, and only while Bot Mode is loaded. */
export function useAgentScreen(connectionId: null | string, profile: string): AgentScreen {
  const b = useBots()
  const bot = useScreenBot(connectionId, profile)
  const { tone } = useScreenPortalState(bot)
  const { phrase, holder } = screenTonePhrase(b, tone)

  return { bot, tone, says: phrase, holder, open: () => openAgentScreen(connectionId, profile) }
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
  const screen = useAgentScreen(agent.connectionId, agent.profile)
  const mark = MARK[screen.tone]

  // The row and its preview card are one component that exists only while the screen runs, so a
  // card left open when the screen stops is gone with it, not waiting for the next run.
  return mark ? <LiveScreenRow mark={mark} screen={screen} /> : null
}

function LiveScreenRow({ mark, screen }: { mark: 'needs-input' | 'working'; screen: AgentScreen }) {
  const b = useBots()
  const { bot, says, holder, open } = screen
  const preview = useRestingHover()
  const row = useRef<HTMLDivElement>(null)
  const card = useRef<HTMLDivElement>(null)

  return (
    <div className="portal-agent-sessions portal-agent-screen">
      <div className="portal-session-item">
        {/* The row opens the Screen; resting on it, or reaching it by keyboard, shows the screen
            itself beside it. The picture is mounted, and asked for, only while that card is up. */}
        <Popover onOpenChange={next => !next && preview.close()} open={preview.open !== null}>
          <PopoverAnchor asChild>
            <div className="portal-session-row" ref={row} {...preview.hoverHandlers}>
              <span aria-hidden className="portal-row-mark" data-state={mark}>
                <span>
                  <span />
                </span>
              </span>
              {/* The mark says agent or person; the meta names which, in a word. The phrase is the name. */}
              <RowButton
                aria-label={`${b.screen.portalTitle}: ${says}`}
                className="portal-session-open"
                // Focus going into the card (a press on its picture) is not leaving it.
                onBlur={event => !card.current?.contains(event.relatedTarget) && preview.close()}
                onClick={() => {
                  preview.close()
                  void open()
                }}
                // Only focus the keyboard brought: a click, or the window coming back to the front,
                // also focuses the row and is not a request to look.
                onFocus={event => event.target.matches(':focus-visible') && preview.setOpen('click')}
                onKeyDown={event => event.key === 'Escape' && preview.close()}
              >
                <span className="portal-session-title">{b.screen.portalTitle}</span>
              </RowButton>
              <span aria-hidden className="portal-session-meta">
                {holder}
              </span>
            </div>
          </PopoverAnchor>
          <PopoverContent
            align="start"
            aria-label={`${b.screen.portalTitle}: ${says}`}
            className="portal-screen-preview"
            // A press on the picture opens the Screen: the look is over, however the card was opened.
            onClick={preview.close}
            // Looking is not moving: focus stays on the row, opening and closing.
            onCloseAutoFocus={event => event.preventDefault()}
            // The row is the card's own anchor, not somewhere else: it decides when the card goes.
            onInteractOutside={event => row.current?.contains(event.target as Node) && event.preventDefault()}
            onOpenAutoFocus={event => event.preventDefault()}
            ref={card}
            side="right"
            {...preview.hoverHandlers}
          >
            <ScreenHero bot={bot} />
          </PopoverContent>
        </Popover>
      </div>
    </div>
  )
}

/** The agent's screen itself: the plugin's live thumbnail (a frame every few seconds while it is on
 *  screen, none once it is gone), for a screen that is running. Mount it only then: it draws its own
 *  placeholders otherwise. Its click opens the Screen as a tab beside the chat, so whatever page
 *  holds it steps aside; `onOpened` hears that click, and only when a Screen could open. */
export function AgentScreenThumbnail({ bot, onOpened }: { bot: RosterRow; onOpened: () => void }) {
  return (
    // The thumbnail is the control; this only hears its click on the way up.
    <div className="portal-agent-screen-thumbnail" onClick={() => screenOpens() && onOpened()} role="presentation">
      <ScreenHero bot={bot} />
    </div>
  )
}
