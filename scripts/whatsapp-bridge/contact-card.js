/**
 * WhatsApp contact card (vCard) parser.
 *
 * Extracts structured contact information from Baileys contactMessage
 * and contactsArrayMessage payloads.  Returns a human-readable text
 * block that the agent can process.
 *
 * vCard 3.0/4.0 fields handled:
 *   FN   — formatted name
 *   TEL  — phone numbers (all instances, with type labels)
 *   EMAIL — email addresses
 *   ORG  — organization / company name
 *
 * @module contact-card
 */

/**
 * Parse all TEL fields from a vCard string.
 * Handles formats like:
 *   TEL;type=CELL:+521234567890
 *   TEL;TYPE=WORK,VOICE:+521234567890
 *   TEL:+521234567890
 *   item1.TEL:+521234567890
 *
 * @param {string} vcard - Raw vCard string
 * @returns {{ number: string, type: string }[]} Parsed phone entries
 */
export function parseVcardPhones(vcard) {
  if (!vcard || typeof vcard !== 'string') return [];

  const phones = [];
  const lines = vcard.split(/\r?\n/);

  for (const line of lines) {
    // Match TEL lines, optionally prefixed with itemN. or grouped property
    const match = line.match(/^(?:item\d+\.)?TEL(?:[;:])(.*)/i);
    if (!match) continue;

    const rest = match[1];

    // Extract type hint (CELL, WORK, HOME, etc.)
    const typeMatch = rest.match(/type=([^:;,]+)/i);
    const type = typeMatch ? typeMatch[1].toLowerCase() : 'phone';

    // Extract the actual number (everything after the last colon)
    const colonIdx = rest.lastIndexOf(':');
    const number = colonIdx >= 0
      ? rest.slice(colonIdx + 1).trim()
      : rest.replace(/^[^+\d]*/, '').trim();

    if (number) {
      phones.push({ number, type });
    }
  }

  return phones;
}

/**
 * Extract a single vCard field value by field name.
 *
 * @param {string} vcard - Raw vCard string
 * @param {string} field - Field name (e.g. 'FN', 'EMAIL', 'ORG')
 * @returns {string|null} First matching value, or null
 */
export function parseVcardField(vcard, field) {
  if (!vcard || typeof vcard !== 'string') return null;

  const re = new RegExp(`^(?:item\\d+\\.)?${field}[;:](.*)`, 'im');
  const match = vcard.match(re);
  if (!match) return null;

  const rest = match[1];
  const colonIdx = rest.lastIndexOf(':');
  const value = colonIdx >= 0 ? rest.slice(colonIdx + 1).trim() : rest.trim();
  return value || null;
}

/**
 * Format a single contact object from Baileys into readable text lines.
 *
 * @param {{ displayName?: string, vcard?: string, vcardFormattedName?: string }} contact
 * @returns {string[]} Lines of formatted contact info
 */
function formatSingleContact(contact) {
  if (!contact || typeof contact !== 'object') return [];

  const lines = [];

  // Name: prefer displayName, fall back to vCard FN field
  const name = contact.displayName
    || (contact.vcard && parseVcardField(contact.vcard, 'FN'))
    || contact.vcardFormattedName
    || null;
  if (name) lines.push(`Name: ${name}`);

  if (contact.vcard) {
    // Organization
    const org = parseVcardField(contact.vcard, 'ORG');
    if (org) lines.push(`Org: ${org}`);

    // All phone numbers with type labels
    const phones = parseVcardPhones(contact.vcard);
    for (const { number, type } of phones) {
      const label = type === 'phone' ? 'Phone' : `Phone (${type})`;
      lines.push(`${label}: ${number}`);
    }

    // Email
    const email = parseVcardField(contact.vcard, 'EMAIL');
    if (email) lines.push(`Email: ${email}`);
  }

  return lines;
}

/**
 * Format a Baileys contact message payload into agent-readable text.
 *
 * Handles both message types:
 *   - contactMessage: single contact with displayName + vcard
 *   - contactsArrayMessage: wrapper with displayName + contacts[]
 *
 * @param {object} messageContent - Baileys contactMessage or contactsArrayMessage
 * @returns {string} Formatted text, e.g. "[contact card]\nName: ...\nPhone: ..."
 */
export function formatContactCard(messageContent) {
  if (!messageContent || typeof messageContent !== 'object') return '';

  const allLines = [];

  // Single contact (contactMessage has displayName + vcard at top level)
  if (messageContent.vcard) {
    allLines.push(...formatSingleContact(messageContent));
  }

  // Multiple contacts (contactsArrayMessage has contacts[])
  if (Array.isArray(messageContent.contacts)) {
    for (const contact of messageContent.contacts) {
      const contactLines = formatSingleContact(contact);
      if (contactLines.length > 0) {
        if (allLines.length > 0) allLines.push('---');
        allLines.push(...contactLines);
      }
    }
  }

  if (allLines.length > 0) {
    return `[contact card]\n${allLines.join('\n')}`;
  }

  return '[contact card received]';
}
