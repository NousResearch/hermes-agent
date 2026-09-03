/**
 * HUD switchers — the three small controls that ride the composer's controls
 * row in HUD mode: follow-the-pointer toggle, the agent pill, the room pill.
 *
 * Agents open through `openHudForProfile` (the HUD respawns against the new
 * profile's backend — a renderer adopts its backend once). Rooms are remote
 * control: the HUD asks main to open the room in the app window and keeps
 * its own session, because rooms live in the pane tree the HUD lacks.
 */

import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { GHOST_ICON_BTN } from '@/app/chat/composer/control-classes'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  dropdownMenuRow,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import type { HudLaunchOptions } from '@/lib/hud-prefs'
import { cn } from '@/lib/utils'
import {
  $hudLaunchOptions,
  $hudPrefs,
  $hudRoom,
  enterHudRoom,
  leaveHudRoom,
  openHudForProfile,
  openHudRoom,
  refreshHudLaunchOptions,
  toggleHudFollow
} from '@/store/hud'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $busy } from '@/store/session'

type HudAgent = HudLaunchOptions['agents'][number]

/** The agent's face: the Bot Mode avatar when one is set, else its role glyph
 *  (lib/agent-emoji), else the colour dot that was here before. */
function AgentGlyph({ agent }: { agent: HudAgent | undefined }) {
  if (agent?.image) {
    return <img alt="" className="size-4 shrink-0 rounded-full object-cover" src={agent.image} />
  }

  if (agent?.emoji) {
    return (
      <span aria-hidden className="shrink-0 text-[0.8125rem] leading-none">
        {agent.emoji}
      </span>
    )
  }

  return (
    <span
      aria-hidden
      className="size-2 shrink-0 rounded-full"
      style={{ background: agent?.color || 'var(--ui-text-tertiary)' }}
    />
  )
}

export function HudFollowToggle() {
  const prefs = useStore($hudPrefs)
  const { t } = useI18n()
  const h = t.hud
  const on = prefs?.follow === true
  const supported = prefs?.followSupported !== false

  return (
    <Tip label={supported ? (on ? h.followOn : h.followOff) : h.followUnsupported}>
      <Button
        aria-label={h.follow}
        aria-pressed={on}
        className={cn(GHOST_ICON_BTN, 'p-0', on && 'text-(--ui-accent)')}
        disabled={!supported}
        onClick={toggleHudFollow}
        size="icon"
        type="button"
        variant="ghost"
      >
        <Codicon name="magnet" size="0.875rem" />
      </Button>
    </Tip>
  )
}

/** Refresh the cached lists whenever a pill opens — the rows come from the
 *  primary renderer's push, so they are cheap to re-pull and may be stale. */
function useRefreshOnOpen(): (open: boolean) => void {
  return (open: boolean) => {
    if (open) {
      void refreshHudLaunchOptions()
    }
  }
}

export function HudAgentPill() {
  const { agents } = useStore($hudLaunchOptions)
  const active = normalizeProfileKey(useStore($activeGatewayProfile))
  const busy = useStore($busy)
  const { t } = useI18n()
  const h = t.hud
  const onOpenChange = useRefreshOnOpen()
  const current = agents.find(agent => normalizeProfileKey(agent.profile) === active)
  const label = current?.displayName ?? active

  return (
    <DropdownMenu onOpenChange={onOpenChange}>
      <Tip label={h.switchAgent}>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={`${h.agent}: ${label}`}
            className={cn(GHOST_ICON_BTN, 'max-w-32 gap-1 px-1.5 text-xs')}
            size="sm"
            type="button"
            variant="ghost"
          >
            <AgentGlyph agent={current} />
            <span className="truncate">{label}</span>
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>{h.agent}</DropdownMenuLabel>
        {busy ? <div className="px-2.5 pb-1 text-[0.625rem] text-(--ui-text-tertiary)">{h.busyWarning}</div> : null}
        <DropdownMenuSeparator />
        {agents.length === 0 ? (
          <DropdownMenuItem className={dropdownMenuRow} disabled>
            {h.noAgents}
          </DropdownMenuItem>
        ) : (
          agents.map(agent => {
            const key = normalizeProfileKey(agent.profile)
            const isActive = key === active

            return (
              <DropdownMenuItem
                aria-current={isActive ? 'true' : undefined}
                className={cn(dropdownMenuRow, isActive && 'font-medium')}
                disabled={!agent.reachable}
                key={key}
                onSelect={() => {
                  if (!isActive) {
                    void openHudForProfile(agent.profile)
                  }
                }}
              >
                <AgentGlyph agent={agent} />
                <span className="truncate">{agent.displayName}</span>
                {agent.title ? (
                  <span className="truncate text-[0.625rem] text-(--ui-text-tertiary)">{agent.title}</span>
                ) : null}
                {isActive ? <Codicon className="ml-auto" name="check" size="0.75rem" /> : null}
              </DropdownMenuItem>
            )
          })
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function HudRoomPill() {
  const { groups } = useStore($hudLaunchOptions)
  const room = useStore($hudRoom)
  const { t } = useI18n()
  const h = t.hud
  const onOpenChange = useRefreshOnOpen()
  const [pending, setPending] = useState<null | string>(null)

  return (
    <DropdownMenu onOpenChange={onOpenChange}>
      <Tip label={h.switchRoom}>
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={room ? h.inRoom(room) : h.room}
            aria-pressed={room !== null}
            className={cn(GHOST_ICON_BTN, room ? 'max-w-32 gap-1 px-1.5 text-xs text-(--ui-accent)' : 'p-0')}
            size={room ? 'sm' : 'icon'}
            type="button"
            variant="ghost"
          >
            <Codicon name="comment-discussion" size="0.875rem" />
            {room ? <span className="truncate">{room}</span> : null}
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>{h.room}</DropdownMenuLabel>
        {room ? (
          <>
            <DropdownMenuItem className={dropdownMenuRow} onSelect={leaveHudRoom}>
              <Codicon name="sign-out" size="0.75rem" />
              <span className="truncate">{h.leaveRoom}</span>
            </DropdownMenuItem>
            <DropdownMenuItem className={dropdownMenuRow} onSelect={() => void openHudRoom(room)}>
              <Codicon name="window" size="0.75rem" />
              <span className="truncate">{h.openRoomInApp}</span>
            </DropdownMenuItem>
          </>
        ) : null}
        <DropdownMenuSeparator />
        {groups.length === 0 ? (
          <DropdownMenuItem className={dropdownMenuRow} disabled>
            {h.noRooms}
          </DropdownMenuItem>
        ) : (
          groups.map(group => (
            <DropdownMenuItem
              className={dropdownMenuRow}
              disabled={!group.reachable || pending === group.groupId}
              key={group.groupId}
              onSelect={() => {
                setPending(group.groupId)
                void enterHudRoom(group.groupId).finally(() => setPending(null))
              }}
            >
              <span className="truncate">{group.displayName}</span>
              {room === group.groupId ? <Codicon className="ml-1" name="check" size="0.75rem" /> : null}
              {typeof group.memberCount === 'number' ? (
                <span className="ml-auto text-[0.625rem] text-(--ui-text-tertiary)">{group.memberCount}</span>
              ) : null}
            </DropdownMenuItem>
          ))
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function HudSwitchers() {
  return (
    <>
      <HudFollowToggle />
      <HudAgentPill />
      <HudRoomPill />
    </>
  )
}
