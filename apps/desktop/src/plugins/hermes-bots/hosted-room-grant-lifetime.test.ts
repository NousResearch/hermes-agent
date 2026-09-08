import { describe, expect, it } from 'vitest'

import {
  hasRequestedRoomGrantLifetime,
  ROOM_GRANT_STATUS_TTL_SECONDS,
  ROOM_GRANT_TTL_SECONDS
} from './hosted-room-client'

describe('issued room-grant lifetime', () => {
  it('requires the bounded issuer-relative window, not client clock assumptions or a claimed capability', () => {
    const expires_at = 1788000000 + ROOM_GRANT_TTL_SECONDS
    const status_expires_at = 1788000000 + ROOM_GRANT_STATUS_TTL_SECONDS
    expect(hasRequestedRoomGrantLifetime({ expires_at, status_expires_at })).toBe(true)
    expect(
      hasRequestedRoomGrantLifetime({ expires_at: expires_at + 86400, status_expires_at: status_expires_at + 86400 })
    ).toBe(true)

    for (const candidate of [
      null,
      {},
      { status_expires_at },
      { expires_at },
      { expires_at, status_expires_at: expires_at },
      { expires_at, status_expires_at: Infinity },
      { expires_at: NaN, status_expires_at },
      { expires_at: String(expires_at), status_expires_at },
      { expires_at, status_expires_at: status_expires_at + 86400 }
    ]) {
      expect(hasRequestedRoomGrantLifetime(candidate)).toBe(false)
    }
  })
})
