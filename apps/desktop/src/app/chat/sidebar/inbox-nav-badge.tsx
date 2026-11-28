import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { $inboxUnreadCount } from '@/store/inbox'

import { INBOX_ROUTE } from '../../routes'
import type { SidebarNavItem } from '../../types'

// Every tier: scheduled-job results are for anyone who scheduled a job, and
// the row hides itself on a backend without `inbox.list`.
export const INBOX_NAV_ITEM: SidebarNavItem = {
  id: 'inbox',
  label: '',
  icon: props => <Codicon name="inbox" {...props} />,
  route: INBOX_ROUTE
}

const MAX_SHOWN = 99

/** Nav filter: the Inbox row shows only once the backend answered `inbox.list`. */
export const inboxNavShown =
  (available: boolean) =>
  (item: SidebarNavItem): boolean =>
    available || item.id !== INBOX_NAV_ITEM.id

/** Unread count on the Inbox nav row; nothing at zero. */
export function InboxNavBadge() {
  const { t } = useI18n()
  const count = useStore($inboxUnreadCount)

  if (count <= 0) {
    return null
  }

  return (
    <span
      aria-label={t.inbox.unreadCount(count)}
      className="ml-auto min-w-4 shrink-0 rounded-full bg-(--ui-control-active-background) px-1.5 text-center text-[0.625rem] font-semibold leading-4 text-foreground tabular-nums"
      data-slot="inbox-nav-badge"
      role="status"
    >
      {count > MAX_SHOWN ? `${MAX_SHOWN}+` : count}
    </span>
  )
}
