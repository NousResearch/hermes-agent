import { atom } from 'nanostores'

// Invalidate server-owned search snapshots, never retain a competing tags cache.
export const $sessionTagsRevision = atom(0)

export function invalidateSessionTags() {
  $sessionTagsRevision.set($sessionTagsRevision.get() + 1)
}
