import { Codicon, relativeTime } from '@hermes/plugin-sdk'

import type { GroupReviewReceipt } from './types'

/** Read-only audit line, not a bot reply: no reply button, mentions or model input. */
export function GroupReviewRow({ receipt }: { receipt: GroupReviewReceipt }) {
  return (
    <div className="flex min-w-0 items-start gap-1.5 px-2 py-0.5 text-xs" data-review-receipt={receipt.id}>
      <Codicon className="shrink-0 text-(--tool-memory-legendary-icon)" name="lightbulb" />
      <span className="tool-memory-legendary-title shrink-0">{receipt.member}</span>
      <span
        className="tool-memory-legendary-meta min-w-0 whitespace-pre-wrap wrap-anywhere"
        data-selectable-text="true"
      >
        {receipt.text.replace(/^[^\p{L}\p{N}]+/u, '')}
      </span>
      <time
        className="shrink-0 text-(--ui-text-quaternary)"
        dateTime={new Date(receipt.at).toISOString()}
        title={new Date(receipt.at).toLocaleString()}
      >
        {relativeTime(receipt.at)}
      </time>
    </div>
  )
}
