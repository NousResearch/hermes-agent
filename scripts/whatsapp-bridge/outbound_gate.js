/**
 * Outbound recipient gate for the WhatsApp bridge HTTP send endpoints.
 *
 * Inbound traffic is filtered by WHATSAPP_ALLOWED_USERS, but /send,
 * /send-media, /send-poll, /send-location and /edit historically trusted
 * whatever chatId the caller passed.  Any process able to reach the loopback
 * bridge (a stray curl, a debug script working around the Python gateway)
 * could therefore message arbitrary contacts of the *linked* WhatsApp
 * account.  This module mirrors the inbound allowlist on egress: the linked
 * account itself, WHATSAPP_ALLOWED_USERS, WHATSAPP_HOME_CHANNEL,
 * WHATSAPP_GROUP_ALLOWED_USERS and explicit WHATSAPP_OUTBOUND_ALLOWED entries
 * may receive; anything else is denied (fail closed).  Set
 * WHATSAPP_OUTBOUND_ALLOW_ALL=1 to restore the old open behaviour.
 *
 * Kept dependency-free (pure functions) so it can be unit-tested without
 * starting the server or a Baileys socket — same convention as
 * bridge_helpers.js.
 */

import path from 'node:path';
import { existsSync, readFileSync } from 'node:fs';

function normalize(value) {
  return String(value || '')
    .trim()
    .replace(/:.*@/, '@')
    .replace(/@.*/, '')
    .replace(/^\+/, '');
}

function parseList(rawValue) {
  return new Set(
    String(rawValue || '')
      .split(',')
      .map(normalize)
      .filter(Boolean),
  );
}

function readLidMapping(sessionDir, identifier, suffix = '') {
  const filePath = path.join(sessionDir, `lid-mapping-${identifier}${suffix}.json`);
  if (!existsSync(filePath)) return null;
  try {
    return normalize(JSON.parse(readFileSync(filePath, 'utf8'))) || null;
  } catch {
    return null;
  }
}

// Resolve a bare identifier (digits or full JID) to every alias the allowlist
// might carry it under: phone ↔ LID twins recorded in the Baileys session dir.
export function expandOutboundAliases(identifier, sessionDir) {
  const bare = normalize(identifier);
  const aliases = new Set([bare]);
  if (!bare || !sessionDir) return aliases;
  const direct = readLidMapping(sessionDir, bare);
  if (direct) aliases.add(direct);
  const reverse = readLidMapping(sessionDir, bare, '_reverse');
  if (reverse) aliases.add(reverse);
  return aliases;
}

/**
 * @param {object} options
 * @param {Record<string,string|undefined>} options.env  process.env (injectable for tests)
 * @param {string} options.sessionDir  Baileys multi-file auth dir (LID mappings)
 * @param {() => ({id?: string, lid?: string}|null)} options.getAccount  linked-account identity
 * @returns {(chatId: unknown) => boolean}
 */
export function createOutboundGate({ env, sessionDir, getAccount }) {
  const allowAll = ['1', 'true', 'yes', 'on'].includes(
    String(env.WHATSAPP_OUTBOUND_ALLOW_ALL || '').toLowerCase());
  const allowed = new Set([
    ...parseList(env.WHATSAPP_ALLOWED_USERS),
    ...parseList(env.WHATSAPP_OUTBOUND_ALLOWED),
    ...parseList(env.WHATSAPP_HOME_CHANNEL),
    ...parseList(env.WHATSAPP_GROUP_ALLOWED_USERS),
  ]);

  return function isOutboundChatAllowed(chatId) {
    if (allowAll) return true;
    const bare = normalize(chatId);
    if (!bare) return false;
    const account = getAccount?.() || {};
    const selfNumber = normalize(account.id || '');
    const selfLid = normalize(account.lid || '');
    if ((selfNumber && bare === selfNumber) || (selfLid && bare === selfLid)) return true;
    if (allowed.has('*')) return true;
    for (const alias of expandOutboundAliases(bare, sessionDir)) {
      if (allowed.has(alias)) return true;
    }
    return false;
  };
}
