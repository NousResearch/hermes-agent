/**
 * Room mode in the HUD. While a room is entered, the HUD's composer posts
 * into that room instead of the agent session (see use-prompt-actions/submit
 * → postHudRoom) and this panel shows the room's log as it grows — pushed
 * from the app window, where Bot Mode's room engine actually runs. The HUD
 * never drives a round itself; it is a remote for the room.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { $hudRoom, $hudRoomFeed, leaveHudRoom, openHudRoom } from '@/store/hud'

export function HudRoomPanel() {
  const room = useStore($hudRoom)
  const feed = useStore($hudRoomFeed)
  const { t } = useI18n()
  const h = t.hud
  const endRef = useRef<HTMLDivElement | null>(null)
  const count = feed?.entries.length ?? 0

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end' })
  }, [count, feed?.turn])

  if (!room) {
    return null
  }

  const entries = feed?.groupId === room ? feed.entries : []
  const thinking = feed?.groupId === room && feed.running ? feed.turn : null

  return (
    <div
      aria-label={h.inRoom(room)}
      className="absolute inset-x-3 bottom-2 top-[calc(var(--hud-top-inset,0px)+var(--hud-bar-height,56px)+8px)] z-10 flex flex-col overflow-hidden rounded-lg border border-(--ui-stroke-secondary) bg-[rgb(12_14_18/0.82)] text-[#f2f5f9] shadow-lg"
      data-hud-room
      role="region"
    >
      <div className="flex items-center gap-2 border-b border-(--ui-stroke-tertiary) px-3 py-1.5 text-[0.6875rem]">
        <Codicon name="comment-discussion" size="0.75rem" />
        <span className="truncate font-medium">{room}</span>
        {feed?.members.length ? (
          <span className="truncate text-[0.625rem] text-[#aebaca]">{feed.members.join(' · ')}</span>
        ) : null}
        <span className="ml-auto flex shrink-0 items-center gap-1">
          <Button
            aria-label={h.openRoomInApp}
            className="h-6 px-1.5 text-[0.625rem]"
            onClick={() => void openHudRoom(room)}
            size="sm"
            type="button"
            variant="ghost"
          >
            {h.openRoomInApp}
          </Button>
          <Button
            aria-label={h.leaveRoom}
            className="h-6 px-1.5 text-[0.625rem]"
            onClick={leaveHudRoom}
            size="sm"
            type="button"
            variant="ghost"
          >
            {h.leaveRoom}
          </Button>
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
        {entries.length === 0 ? (
          <div className="py-4 text-center text-[0.75rem] text-[#aebaca]">{h.roomEmpty}</div>
        ) : (
          entries.map(entry => (
            <div className="mb-2 last:mb-0" key={entry.id}>
              <div className="text-[0.625rem] font-semibold tracking-wide text-[#aebaca] uppercase">
                {entry.kind === 'user' ? h.you : entry.author}
              </div>
              <div className="text-[0.8125rem] leading-snug whitespace-pre-wrap">{entry.text}</div>
            </div>
          ))
        )}
        {thinking ? <div className="mt-1 text-[0.6875rem] text-[#aebaca] italic">{h.roomThinking(thinking)}</div> : null}
        <div ref={endRef} />
      </div>
    </div>
  )
}
