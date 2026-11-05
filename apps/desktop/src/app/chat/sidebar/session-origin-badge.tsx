import type { SessionInfo } from '@/hermes'
import { sessionOriginName } from '@/lib/session-source'

/**
 * WHO the conversation is with.
 *
 * A messaging row's title is the conversation's topic, never the person on the
 * other end — several people's DMs read identically without it. The platform
 * records the sender in the row's origin payload, so show it and let the rows
 * be told apart. Nothing for a local session, or for a platform that records
 * no name.
 */
export function SessionOriginBadge({ session }: { session: SessionInfo }) {
  const name = sessionOriginName(session)

  if (!name) {
    return null
  }

  return (
    <span
      className="max-w-24 shrink-0 truncate text-[0.625rem] leading-none text-(--ui-text-tertiary)"
      data-origin-name=""
      title={name}
    >
      {name}
    </span>
  )
}
