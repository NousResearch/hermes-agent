import { SessionContributionSlot } from './session'
import { type SessionContributionIdentity, useSessionContributionContext } from './session-context'

/** Leaf subscription: status stacks and tab strips do not subscribe to other chats. */
export function SessionContributions({ area, ...identity }: SessionContributionIdentity & { area: string }) {
  const session = useSessionContributionContext(identity)

  return <SessionContributionSlot area={area} session={session} />
}
