import { Badge } from '@/components/ui/badge'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Incognito } from '@/lib/icons'

import { useTemporaryChat } from './use-temporary-chat'

/** Header mark for an open temporary chat (nothing otherwise): it is not in the sidebar, so this is
 *  where it says so. */
export function TemporaryChatBadge({ className, sessionId }: { className?: string; sessionId: null | string }) {
  const { t } = useI18n()

  const temporary = useTemporaryChat(sessionId)

  if (!temporary || !sessionId) {
    return null
  }

  return (
    <Tip label={t.temporaryChat.badgeHint} placement="toolbar">
      <Badge className={className} data-testid="temporary-chat-badge" variant="default">
        <Incognito />
        {t.temporaryChat.badge}
      </Badge>
    </Tip>
  )
}
