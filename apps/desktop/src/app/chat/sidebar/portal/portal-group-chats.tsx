import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { bindGatewayGroupRoom, useGatewayGroupRooms } from '@/plugins/hermes-bots/canonical-group-registry'
import { CanonicalGroupRow } from '@/plugins/hermes-bots/canonical-group-row'
import { CreateGroupChatDialog } from '@/plugins/hermes-bots/create-dialog'
import { $botMeta, $lastRoster, primeRoster } from '@/plugins/hermes-bots/data'
import { $groupChats, $groupChatWorkspace } from '@/plugins/hermes-bots/group-chat'
import { openGroupChat } from '@/plugins/hermes-bots/group-chat-view'
import { groupChatMemberBots, groupChatNames } from '@/plugins/hermes-bots/group-membership'
import { useBots } from '@/plugins/hermes-bots/i18n'

/** Group rooms sit outside gateway buckets: Desktop-driven rooms may span gateways. */
export function PortalGroupChats({ query }: { query: string }) {
  const b = useBots()
  const roster = useStore($lastRoster)
  const meta = useStore($botMeta)
  const rooms = useStore($groupChats)
  const active = useStore($groupChatWorkspace)
  const gatewayRooms = useGatewayGroupRooms()
  const [creating, setCreating] = useState(false)
  const needle = query.trim().toLocaleLowerCase()
  const names = groupChatNames(meta, rooms).filter(name => name.toLocaleLowerCase().includes(needle))
  const visibleGatewayRooms = gatewayRooms.filter(room => room.name.toLocaleLowerCase().includes(needle))

  return (
    <section aria-label={b.roster.groupChats} className="flex min-h-10 shrink-0 flex-col gap-1">
      <div className="flex items-center justify-between px-2 py-1">
        <span className="text-[0.8125rem] text-(--ui-text-tertiary)">{b.roster.groupChats}</span>
        <Tip label={b.group.newTitle}>
          <Button aria-label={b.group.newTitle} onClick={() => {
            void primeRoster()
            setCreating(true)
          }} size="icon-xs" variant="ghost">
            <Codicon name="add" />
          </Button>
        </Tip>
      </div>
      <FadeScroll maxHeight="12rem">
        {visibleGatewayRooms.map(room => (
          <CanonicalGroupRow active={active === room.key} key={room.key}
            onOpen={() => openGroupChat(bindGatewayGroupRoom(room), 'key')} room={room} />
        ))}
        {names.map(name => (
          <RowButton aria-current={active === name ? 'page' : undefined}
            className="flex w-full min-w-0 items-center gap-2 rounded-sm px-2 py-2 text-left text-[0.8125rem] hover:bg-(--chrome-action-hover) aria-[current=page]:bg-(--chrome-action-hover)"
            key={name} onClick={() => openGroupChat(name, 'name')}>
            <Codicon name="organization" />
            <span className="min-w-0 flex-1 truncate">{name}</span>
            <span className="shrink-0 text-xs text-(--ui-text-tertiary)">
              {b.group.memberCount(groupChatMemberBots(name, roster, meta).length)}
            </span>
          </RowButton>
        ))}
      </FadeScroll>
      {creating && <CreateGroupChatDialog onClose={() => setCreating(false)} onCreated={openGroupChat} open roster={roster} />}
    </section>
  )
}
