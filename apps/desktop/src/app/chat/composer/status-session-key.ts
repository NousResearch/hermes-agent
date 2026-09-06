/**
 * Which session identity the composer's status stack polls under.
 *
 * Background rows come from `process.list`, and a conversation has two identities:
 * the RUNTIME session id (what gateway events and a live `process.list` speak) and
 * the DURABLE stored key (the lineage root the composer already uses to scope its
 * draft and queue). The runtime id wins whenever one is bound — it is the only one
 * that resolves a live session on the backend.
 *
 * The fallback matters for messaging: a stored Telegram conversation can be open in
 * Desktop with no runtime session at all, and passing `null` meant the stack armed
 * no poll, so a `terminal(background=true)` job running on the gateway side was
 * never discovered. The backend accepts the durable key only after proving it names
 * a real stored conversation, so handing it over widens nothing.
 */
export function statusStackSessionKey(
  runtimeSessionId: null | string | undefined,
  durableSessionKey: null | string | undefined
): null | string {
  return (runtimeSessionId ?? '').trim() || (durableSessionKey ?? '').trim() || null
}
