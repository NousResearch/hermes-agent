import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { iconSize, Incognito } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $messagesEmpty, $selectedStoredSessionId } from '@/store/session'
import { $temporaryChats, $temporaryDraft, toggleTemporaryDraft } from '@/store/temporary-chat'

import { confirmEndTemporaryChat } from '../temporary/confirm-end'

import { ACTIVE_ICON_BTN, GHOST_ICON_BTN } from './control-classes'
import { useComposerScope } from './scope'

/**
 * The temporary-chat control on the main composer, visible for the whole life of the chat:
 * - a fresh, empty draft: a "Use / Exit temporary chat" toggle (the first send decides);
 * - a live temporary chat: a pressed "Temporary" pill — the persistent "you are private" mark —
 *   that ends (and deletes) the chat after a confirmation.
 */
export function TemporaryChatToggle({ disabled }: { disabled: boolean }) {
  const { t } = useI18n()
  const scope = useComposerScope()
  const draftTemporary = useStore($temporaryDraft)
  const chats = useStore($temporaryChats)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const messagesEmpty = useStore($messagesEmpty)

  if (scope.target !== 'main') {
    return null
  }

  if (selectedStoredSessionId && chats[selectedStoredSessionId]) {
    return <LiveTemporaryPill storedId={selectedStoredSessionId} />
  }

  if (selectedStoredSessionId || !messagesEmpty) {
    return null
  }

  const label = draftTemporary ? t.temporaryChat.exit : t.temporaryChat.use

  return (
    <Tip label={<TipKeybindLabel actionId="session.newTemporary" text={label} />} placement="control">
      <Button
        aria-label={label}
        aria-pressed={draftTemporary}
        className={cn(GHOST_ICON_BTN, 'p-0', draftTemporary && ACTIVE_ICON_BTN)}
        data-testid="temporary-chat-toggle"
        disabled={disabled}
        onClick={() => {
          triggerHaptic('selection')
          toggleTemporaryDraft()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        <Incognito className={iconSize.sm} />
      </Button>
    </Tip>
  )
}

function LiveTemporaryPill({ storedId }: { storedId: string }) {
  const { t } = useI18n()
  const c = t.temporaryChat

  return (
    <Tip label={`${c.endChat} — ${c.badgeHint}`} placement="control">
      <Button
        aria-label={c.endChat}
        aria-pressed
        className={cn(ACTIVE_ICON_BTN, 'h-(--composer-control-size) shrink-0 gap-1 rounded-full px-2 text-xs')}
        data-testid="temporary-chat-live"
        onClick={() => {
          triggerHaptic('selection')
          void confirmEndTemporaryChat(storedId, c)
        }}
        type="button"
        variant="ghost"
      >
        <Incognito className={iconSize.sm} />
        {c.badge}
      </Button>
    </Tip>
  )
}
