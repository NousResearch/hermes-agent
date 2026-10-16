import type { TemporaryChatTranslations } from '@/i18n/types_temporary_chat'
import { confirm } from '@/store/confirm'
import { endTemporaryChat } from '@/store/temporary-chat'

/** End (and delete) a temporary chat once the user confirms — every End button goes through here. */
export async function confirmEndTemporaryChat(storedId: string, copy: TemporaryChatTranslations): Promise<void> {
  const ok = await confirm({
    confirmLabel: copy.endConfirm,
    description: copy.endDescription,
    destructive: true,
    title: copy.endTitle
  })

  if (ok) {
    endTemporaryChat(storedId)
  }
}
