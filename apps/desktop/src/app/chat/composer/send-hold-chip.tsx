import { memo } from 'react'

import { composerFloatingPill } from '@/components/chat/composer-dock'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

interface SendHoldChipProps {
  /** Localised "sending, press Esc to take it back" text. */
  label: string
  /** Take the send back. The draft never left the composer, so this is all it takes. */
  onCancel: () => void
}

/**
 * The visible half of the send grace window.
 *
 * A held send with no indicator is indistinguishable from the app hanging, so
 * the hold is only shippable alongside this chip: it names what is about to
 * happen, how much time there is to stop it, and is itself the mouse path to
 * the same cancel `Esc` performs.
 *
 * Rendered in the composer's floating strip rather than the status stack — the
 * hold belongs to THIS composer, not to the session, and it must not survive a
 * session swap the way a background-process row does.
 */
export const SendHoldChip = memo(function SendHoldChip({ label, onCancel }: SendHoldChipProps) {
  return (
    // `contents` keeps the strip's flex layout intact while giving the
    // announcement a real live region — the chip appears on its own, so there
    // is no other cue that the send is being withheld.
    <span className="contents" role="status">
      <button
        className={cn(composerFloatingPill, 'cursor-default')}
        onClick={onCancel}
        title={label}
        type="button"
      >
        <Codicon className="shrink-0 opacity-70" name="loading" size="0.75rem" spinning />
        <span className="truncate">{label}</span>
      </button>
    </span>
  )
})
