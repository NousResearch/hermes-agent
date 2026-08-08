/**
 * WhatsApp bridge history store — importable helper.
 *
 * Encapsulates the bounded in-memory message/contact stores used by the
 * History API endpoints in bridge.js.  Extracted so the unit tests exercise
 * the SAME code the bridge runs in production (previously the tests copied
 * the functions, so production/endpoint regressions could slip through).
 *
 * Design:
 *  - Per-chat `Map<messageId, event>` for O(1) dedup and lookup.
 *  - A parallel insertion-order `Map` per chat so "newest" is tracked
 *    without O(n) array scans (dedup moves a key to the end via
 *    delete + re-set, which JS Maps preserve).
 *  - FIFO eviction: when a chat exceeds maxMessagesPerChat, the oldest
 *    insertion is dropped.
 *  - Contacts are a single bounded Map (maxContacts entries, oldest
 *    insertion evicted).
 *
 * When `enabled` is false the stores are `null` and every mutator is a
 * no-op — matching bridge.js's opt-in behavior without an extra flag check
 * at each call site.
 */

export function createHistoryStore({
  enabled = true,
  maxMessagesPerChat = 200,
  maxContacts = 5000,
} = {}) {
  const chatMessageStore = enabled ? new Map() : null;
  const chatOrderQueues = enabled ? new Map() : null;
  const contactStore = enabled ? new Map() : null;

  // Normalise a Baileys timestamp (number | protobuf Long | undefined) to
  // a plain Unix-second integer.  In Baileys `messaging-history.set` the
  // timestamp arrives as a protobuf Long object (e.g. `{ low, high }`),
  // not a plain number; storing it raw breaks sort comparisons.
  function toNumberSafe(raw) {
    if (raw == null) return Math.floor(Date.now() / 1000);
    if (typeof raw === 'object' && typeof raw.toNumber === 'function') return raw.toNumber();
    const n = Number(raw);
    return Number.isFinite(n) ? n : Math.floor(Date.now() / 1000);
  }

  // O(1) store + dedup by messageId using a Map per chat.
  function storeMessage(chatId, event) {
    if (!chatId || !event || !chatMessageStore || !chatOrderQueues) return;
    let byMsgId = chatMessageStore.get(chatId);
    let order = chatOrderQueues.get(chatId);
    if (!byMsgId) {
      byMsgId = new Map();
      chatMessageStore.set(chatId, byMsgId);
      order = new Map();
      chatOrderQueues.set(chatId, order);
    }
    const id = event.messageId;
    if (!id) return;
    // Update data in O(1)
    byMsgId.set(id, event);
    // Touch order: delete + re-set moves this key to the newest position
    // (JS Map preserves insertion order)
    order.delete(id);
    order.set(id, true);
    // Evict oldest insertion if over cap
    while (order.size > maxMessagesPerChat) {
      const oldestId = order.keys().next().value;
      if (oldestId) {
        order.delete(oldestId);
        byMsgId.delete(oldestId);
      }
    }
  }

  // Store a contact with bounded size.
  function storeContact(jid, info) {
    if (!jid || !contactStore) return;
    contactStore.set(jid, info);
    if (contactStore.size > maxContacts) {
      const oldest = contactStore.keys().next().value;
      if (oldest) contactStore.delete(oldest);
    }
  }

  // Get messages for a chat as an ordered array (newest first).
  function getMessages(chatId, limit) {
    if (!chatMessageStore || !chatOrderQueues) return null;
    const byMsgId = chatMessageStore.get(chatId);
    const order = chatOrderQueues.get(chatId);
    if (!byMsgId || !order || order.size === 0) return null;
    const keys = [...order.keys()];
    const slice = keys.slice(-limit).reverse();
    return slice.map(id => byMsgId.get(id)).filter(Boolean);
  }

  // Message count for a chat (0 when unknown/disabled).
  function count(chatId) {
    if (!chatOrderQueues) return 0;
    const order = chatOrderQueues.get(chatId);
    return order ? order.size : 0;
  }

  return {
    chatMessageStore,
    chatOrderQueues,
    contactStore,
    toNumberSafe,
    storeMessage,
    storeContact,
    getMessages,
    count,
  };
}

// Shared truthy parser for the opt-in feature flags.  The adapter passes
// "true"/"false" explicitly, but we accept the same set of truthy strings
// the bridge historically understood so configs written for older versions
// keep working.
export function parseTruthyEnv(value) {
  return typeof value === 'string' && ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}
