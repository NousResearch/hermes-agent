import { useStore } from '@nanostores/react'

import { type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { useI18n } from '@/i18n'
import {
  $sessionUnreadMutes,
  isSessionUnreadMuted,
  setSessionUnreadMuted,
  type UnreadMuteRow,
  unreadMuteRowFor
} from '@/store/session-unread-mute'

interface SessionUnreadMuteItemProps {
  kit: MenuKit
  sessionId: string
  profile?: string
  target?: UnreadMuteRow
}

/** Subscribe only while a menu is open, not once for every sidebar row. */
export function SessionUnreadMuteItem({ kit, sessionId, profile, target }: SessionUnreadMuteItemProps) {
  const { t } = useI18n()
  useStore($sessionUnreadMutes)
  const row = target ?? unreadMuteRowFor(sessionId, profile)
  const muted = isSessionUnreadMuted(row)

  return renderActionItem(kit, {
    disabled: !sessionId,
    icon: muted ? 'bell' : 'bell-slash',
    label: muted ? t.sidebar.row.unmuteUnread : t.sidebar.row.muteUnread,
    onSelect: () => setSessionUnreadMuted(row, !muted)
  })
}
