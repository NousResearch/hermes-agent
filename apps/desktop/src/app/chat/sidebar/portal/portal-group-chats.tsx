import { useStore } from '@nanostores/react'
import { useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { Plus } from '@/lib/icons'
import { bindGatewayGroupRoom, useGatewayGroupRooms } from '@/plugins/hermes-bots/canonical-group-registry'
import { CanonicalGroupRow } from '@/plugins/hermes-bots/canonical-group-row'
import { CreateGroupChatDialog } from '@/plugins/hermes-bots/create-dialog'
import { $botMeta, $lastRoster } from '@/plugins/hermes-bots/data'
import { $groupChats, $groupChatWorkspace } from '@/plugins/hermes-bots/group-chat'
import { openGroupChat } from '@/plugins/hermes-bots/group-chat-view'
import { groupChatMemberBots, groupChatNames } from '@/plugins/hermes-bots/group-membership'
import { useBots } from '@/plugins/hermes-bots/i18n'

import { PortalSectionHeader } from './portal-section-header'

/** Group rooms sit outside gateway buckets: Desktop-driven rooms may span gateways. Bot Mode owns
 *  them, so the sidebar mounts this only while that plugin is loaded. */
export function PortalGroupChats({ query }: { query: string }) {
  const b = useBots()
  const roster = useStore($lastRoster)
  const meta = useStore($botMeta)
  const rooms = useStore($groupChats)
  const active = useStore($groupChatWorkspace)
  const gatewayRooms = useGatewayGroupRooms()
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

  const needle = query.trim().toLocaleLowerCase()
  const names = groupChatNames(meta, rooms).filter(name => name.toLocaleLowerCase().includes(needle))
  const visibleGatewayRooms = gatewayRooms.filter(room => room.name.toLocaleLowerCase().includes(needle))

  // A search shows only sections that answer it.
  if (needle && visibleGatewayRooms.length + names.length === 0) {
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
