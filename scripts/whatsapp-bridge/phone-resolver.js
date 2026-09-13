/**
 * Resolve a user-supplied phone number through WhatsApp.
 *
 * The returned JID is canonical for the connected account and must be used
 * for allowlists and delivery targets instead of inferring national prefixes.
 */
export async function resolveWhatsAppNumber(sock, value) {
  const number = String(value || '').replace(/\D/g, '');
  if (!number) {
    throw new TypeError('number must contain digits');
  }

  const results = await sock.onWhatsApp(number);
  const matches = (results || []).filter(item => item?.exists && item?.jid);
  return { number, exists: matches.length > 0, matches };
}
