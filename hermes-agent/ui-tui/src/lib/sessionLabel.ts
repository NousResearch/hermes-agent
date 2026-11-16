import { createHash } from 'node:crypto'

// The canonical owner names a session by its durable id: `local-` + 64 hex for sessions it created,
// the stored `YYYYMMDD_HHMMSS_xxxxxx` id for history adopted from main. Main's banner always showed an
// 8-hex runtime label; the full id stays in state for every RPC.
const CANONICAL_LOCAL_ID = /^local-([0-9a-f]{8})[0-9a-f]{56}$/
const STORED_ID = /^\d{8}_\d{6}_[0-9a-z]+$/i

export const sessionLabel = (sid: null | string | undefined): null | string | undefined => {
  if (!sid) {
    return sid
  }

  const local = CANONICAL_LOCAL_ID.exec(sid)?.[1]

  if (local) {
    return local
  }

  return STORED_ID.test(sid) ? createHash('sha256').update(sid).digest('hex').slice(0, 8) : sid
}
