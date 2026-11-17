import { MessagePrimitive, useAuiState } from '@assistant-ui/react'
import type { FC } from 'react'

import { ClaimedDirective, useDirectiveClaimed } from '@/components/assistant-ui/transcript-directive'
import type { PluginCard } from '@/lib/chat-messages'

/** A plugin card (`ctx.post_card`) whose directive a Desktop plugin claims right now. */
export type ClaimedPluginCard = PluginCard & { directive: NonNullable<PluginCard['directive']> }

/** The claimed card on the current system row, or undefined when the row is no
 *  card, the card carries no directive, or no plugin claims it (the row then
 *  renders as plain text, which is the card's fallback). */
export function useClaimedPluginCard(): ClaimedPluginCard | undefined {
  const pluginCard = useAuiState(s => s.message.metadata.custom?.pluginCard as PluginCard | undefined)
  const directive = pluginCard?.directive
  const claimed = useDirectiveClaimed(directive?.name)

  return pluginCard && directive && claimed ? { ...pluginCard, directive } : undefined
}

/** A plugin's card rendered as that plugin's claimed directive component. */
export const PluginCardRow: FC<{ card: ClaimedPluginCard }> = ({ card }) => (
  <MessagePrimitive.Root
    className="w-full min-w-0 self-start py-1 pl-(--message-text-indent)"
    data-plugin-card={card.cardId}
    data-role="system"
    data-slot="aui_system-message-root"
  >
    <ClaimedDirective attrs={card.directive.attrs} name={card.directive.name} />
  </MessagePrimitive.Root>
)
