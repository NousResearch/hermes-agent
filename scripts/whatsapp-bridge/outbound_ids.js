/**
 * Bounded LRU set of outbound message IDs.
 *
 * Used by the WhatsApp bridge to distinguish "echo of our own /send" from
 * "owner-typed message on the linked device" when forwarding `fromMe`
 * inbound events back to the Python adapter.
 *
 * Heuristic limitation (intentional, documented for future debugging):
 * the set is in-memory only.  On bridge restart it is empty, so for the
 * brief window between restart and the first new outbound, any in-flight
 * delivery receipts of pre-restart sends would be classified as
 * owner-typed.  The TTL on owner-driven plugin actions (e.g. handover
 * sliding TTL) bounds blast radius; persisting would not be worth the
 * extra complexity / disk churn.
 */

export function createOutboundIdTracker(maxSize = 512) {
  const ids = new Set();

  function remember(id) {
    if (!id) return;
    ids.add(id);
    while (ids.size > maxSize) {
      // Set iteration order is insertion order, so values().next() is the
      // oldest entry — drop it to keep memory flat under sustained sending.
      ids.delete(ids.values().next().value);
    }
  }

  function has(id) {
    return Boolean(id) && ids.has(id);
  }

  function size() {
    return ids.size;
  }

  return { remember, has, size };
}
