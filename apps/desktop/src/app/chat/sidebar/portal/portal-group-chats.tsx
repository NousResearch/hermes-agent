import { host, useValue } from '@hermes/plugin-sdk'
import { useStore } from '@nanostores/react'
import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { Plus } from '@/lib/icons'
import { botOnConnection } from '@/plugins/hermes-bots/canonical-group-capabilities'
import {
  $disbandedGatewayRooms,
  bindGatewayGroupRoom,
  type GatewayGroupRoom,
  useGatewayGroupRooms
} from '@/plugins/hermes-bots/canonical-group-registry'
import { CanonicalGroupRow } from '@/plugins/hermes-bots/canonical-group-row'
import { CreateGroupChatDialog } from '@/plugins/hermes-bots/create-dialog'
import { $botMeta, $lastRoster } from '@/plugins/hermes-bots/data'
import { $groupChats, $groupChatWorkspace } from '@/plugins/hermes-bots/group-chat'
import { openGroupChat } from '@/plugins/hermes-bots/group-chat-view'
import { groupChatMemberBots, groupChatNames } from '@/plugins/hermes-bots/group-membership'
import { useBots } from '@/plugins/hermes-bots/i18n'
import { $activeGatewayProfile } from '@/store/profile'

import { PortalSectionHeader } from './portal-section-header'

/** The group chats a search answers with: every one while nothing is typed. */
export interface GroupChatMatches {
  rooms: GatewayGroupRoom[]
  names: string[]
}

/**
 * The gateway rooms this section lists. `useGatewayGroupRooms` reads the active agent's store, and a
 * room lives in the store of the agent that made it; this section is not one agent's, so it also
 * keeps the rooms it saw while another agent of this machine was the active one. A room's own agent
 * is the authority on it: once that agent's list no longer has it, or the agent itself is gone, it
 * goes. Agents never opened in this window list nothing here until they are.
 */
function useSeenGatewayRooms(): GatewayGroupRoom[] {
  const current = useGatewayGroupRooms()
  const connectionId = useValue(host.state.connectionId)
  const profile = useStore($activeGatewayProfile)
  const roster = useStore($lastRoster)
  const disbanded = useStore($disbandedGatewayRooms)
  const seen = useRef<GatewayGroupRoom[]>([])
  // One row a room: a backend with a single store lists the same rooms under every agent.
  const ids = new Set(current.map(room => room.room.room_id))
  // This machine's agents. None means its roster has not been read yet: a machine has at least one.
  const agents = new Set(roster.filter(row => botOnConnection(row, connectionId)).map(row => row.name))

  const earlier = seen.current.filter(
    room =>
      room.binding.connectionId === connectionId &&
      room.binding.profile !== profile &&
      (!agents.size || agents.has(room.binding.profile)) &&
      !ids.has(room.room.room_id) &&
      !disbanded.has(room.key)
  )

  // Newest first, whichever agent is active: opening another agent does not reorder the rows.
  seen.current = [...earlier, ...current].sort((a, b) => b.updatedAt - a.updatedAt || a.key.localeCompare(b.key))

  return seen.current
}

/** Bot Mode's hooks (the gateway room list polls): call only while that plugin is loaded. */
export function useGroupChatMatches(query: string): GroupChatMatches {
  const meta = useStore($botMeta)
  const rooms = useStore($groupChats)
  const gatewayRooms = useSeenGatewayRooms()
  const needle = query.trim().toLocaleLowerCase()

  return {
    rooms: gatewayRooms.filter(room => room.name.toLocaleLowerCase().includes(needle)),
    names: groupChatNames(meta, rooms).filter(name => name.toLocaleLowerCase().includes(needle))
  }
}

/** Group rooms sit outside gateway buckets: Desktop-driven rooms may span gateways. Bot Mode owns
 *  them, so the sidebar mounts this only while that plugin is loaded. */
export function PortalGroupChats({ matches, searching }: { matches: GroupChatMatches; searching: boolean }) {
  const b = useBots()
  const roster = useStore($lastRoster)
  const meta = useStore($botMeta)
  const active = useStore($groupChatWorkspace)
  const { names, rooms: visibleGatewayRooms } = matches
  const [creating, setCreating] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)
  // The dialog unmounts as it closes, so nothing returns focus for it: a dismissal hands focus back
  // to "+", while a created group chat takes it (create calls onClose, then onCreated).
  const created = useRef(false)

  const closeDialog = () => {
    created.current = false
    setCreating(false)
    requestAnimationFrame(() => {
      if (!created.current) {
        trigger.current?.focus()
      }
    })
  }

  const onCreated: typeof openGroupChat = (...args) => {
    created.current = true
    openGroupChat(...args)
  }

  // A search shows only sections that answer it.
  if (searching && visibleGatewayRooms.length + names.length === 0) {
    return null
  }

  return (
    <section aria-label={b.roster.groupChats} className="portal-section" data-slot="group-chats-section">
      <PortalSectionHeader
        action={
          <Tip label={b.group.newTitle}>
            <Button
              aria-label={b.group.newTitle}
              className="portal-section-action"
              onClick={() => setCreating(true)}
              ref={trigger}
              size="icon-xs"
              variant="ghost"
            >
              <Plus aria-hidden />
            </Button>
          </Tip>
        }
        label={b.roster.groupChats}
      />
      {visibleGatewayRooms.length + names.length > 0 && (
        <FadeScroll className="portal-section-scroll" maxHeight="12rem">
          {visibleGatewayRooms.map(room => (
            <CanonicalGroupRow
              active={active === room.key}
              key={room.key}
              onOpen={() => openGroupChat(bindGatewayGroupRoom(room), 'key')}
              room={room}
            />
          ))}
          {names.map(name => (
            <RowButton
              aria-current={active === name ? 'page' : undefined}
              className="portal-session-row"
              key={name}
              onClick={() => openGroupChat(name, 'name')}
            >
              <Codicon className="portal-group-icon" name="organization" />
              <span className="portal-session-title">{name}</span>
              <span className="portal-session-meta">
                {b.group.memberCount(groupChatMemberBots(name, roster, meta).length)}
              </span>
            </RowButton>
          ))}
        </FadeScroll>
      )}
      {creating && (
        <CreateGroupChatDialog onClose={closeDialog} onCreated={onCreated} open roster={roster} />
      )}
    </section>
  )
}
