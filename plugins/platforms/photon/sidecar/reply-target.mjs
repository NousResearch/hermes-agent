// Shared native-reply dispatch for Photon sidecar send routes.
//
// Recent inbound Spectrum Message objects are cached by id. When the gateway
// supplies a matching reply anchor, send through Message.reply(); otherwise
// preserve the ordinary Space.send() path for proactive sends and stale ids.

/**
 * Send a content builder as a native reply when its target is cached.
 *
 * @param {object} space Spectrum Space used for the fallback send
 * @param {object} builder spectrum-ts content builder
 * @param {string|null|undefined} replyTo inbound message id from the gateway
 * @param {Map<string, object>} knownMessages recent inbound Message objects
 * @returns {Promise<object>}
 */
export async function sendWithReply(space, builder, replyTo, knownMessages) {
  const target = replyTo ? knownMessages.get(replyTo) : null;
  return target ? target.reply(builder) : space.send(builder);
}
