import { ContextMenuItem, host } from '@hermes/plugin-sdk'

import { ensureBotMetadata } from './canonical-chat'
import { saveBotMeta } from './data'
import { useBots } from './i18n'
import { displayName } from './labels'
import type { RosterRow } from './types'

interface PrivateToggleItemProps {
  bot: RosterRow
  isPrivate: boolean
}

/** Row-menu Make private / Make public. The label follows local meta; the write flips the
 *  hydrated backend value, so a bot whose state lives elsewhere is never flipped against a
 *  stale local copy. */
export function PrivateToggleItem({ bot, isPrivate }: PrivateToggleItemProps) {
  const b = useBots()

  return (
    <ContextMenuItem
      onSelect={() => {
        void ensureBotMetadata(bot)
          .then(current => {
            const wasPrivate = Boolean(current.private)
            void saveBotMeta(bot, { private: !wasPrivate })
            host.notify({
              kind: 'info',
              message: `${displayName(bot, current)} ${wasPrivate ? b.roster.madePublic : b.roster.madePrivate}`
            })
          })
          .catch(error => host.notifyError?.(error, b.bot.metadataLoadFailed))
      }}
    >
      {isPrivate ? b.roster.makePublic : b.roster.makePrivate}
    </ContextMenuItem>
  )
}
