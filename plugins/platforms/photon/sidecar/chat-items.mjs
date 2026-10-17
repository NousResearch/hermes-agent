// Inbound chat-item filter for the Photon sidecar.
//
// iMessage stores some conversation events as chat items even though nobody
// wrote them: Name & Photo sharing and other chat actions, group renames, and
// people being added or removed. spectrum-ts turns any such item without text
// into custom {imessage_type: "unsupported-message"}, which reached the agent
// as "[Photon content type not handled: custom]" and drew an "I couldn't read
// that" reply. These items are settled like any delivered event and never
// become a turn.
//
// This lives in its own module (rather than inline in index.mjs) so tests can
// execute the real decision logic under node — see
// tests/plugins/platforms/photon/test_chat_items.py.

const NON_MESSAGE_ITEM_TYPES = new Set([
  "chatAction",
  "groupNameChange",
  "participantChange",
]);

/**
 * The item type of an inbound conversation event that is not a message, or
 * null when the item should be forwarded as usual. Only content-less items
 * (spectrum's `custom` fallback) are skipped, so a rename or member change
 * that does carry readable content still reaches the agent.
 *
 * @param {object} message spectrum-ts inbound Message
 * @returns {string|null}
 */
export function nonMessageItemType(message) {
  const itemType = typeof message?.itemType === "string" ? message.itemType : "";
  if (!NON_MESSAGE_ITEM_TYPES.has(itemType)) return null;
  return message?.content?.type === "custom" ? itemType : null;
}
