import type { OwnerScope } from '@/hermes'

// Query-key and React-key form of an owner, so one owner's response never lands on another's view.
export const ownerKey = (owner: OwnerScope): string => `${owner.connectionId ?? ''}::${owner.profile ?? ''}`
