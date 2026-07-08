import path from 'path';
import { mkdirSync, writeFileSync } from 'fs';
import { randomBytes } from 'crypto';

export const MIME_MAP = {
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
  webp: 'image/webp', gif: 'image/gif',
  mp4: 'video/mp4', mov: 'video/quicktime', avi: 'video/x-msvideo',
  mkv: 'video/x-matroska', '3gp': 'video/3gpp',
  pdf: 'application/pdf',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
};

export function normalizeWhatsAppId(value) {
  if (!value) return '';
  return String(value).replace(':', '@');
}

export function getMessageContent(msg) {
  const content = msg?.message || {};
  if (content.ephemeralMessage?.message) return content.ephemeralMessage.message;
  if (content.viewOnceMessage?.message) return content.viewOnceMessage.message;
  if (content.viewOnceMessageV2?.message) return content.viewOnceMessageV2.message;
  if (content.documentWithCaptionMessage?.message) return content.documentWithCaptionMessage.message;
  if (content.templateMessage?.hydratedTemplate) return content.templateMessage.hydratedTemplate;
  if (content.buttonsMessage) return content.buttonsMessage;
  if (content.listMessage) return content.listMessage;
  return content;
}

export function getContextInfo(messageContent) {
  if (!messageContent || typeof messageContent !== 'object') return {};
  for (const value of Object.values(messageContent)) {
    if (value && typeof value === 'object' && value.contextInfo) {
      return value.contextInfo;
    }
  }
  return {};
}

export function createBoundedMessageStore(limit = 512) {
  const byId = new Map();

  function remember(msg) {
    const id = msg?.key?.id;
    if (!id) return;
    byId.delete(id);
    byId.set(id, msg);
    while (byId.size > limit) {
      const oldest = byId.keys().next().value;
      byId.delete(oldest);
    }
  }

  function get(id) {
    if (!id || !byId.has(id)) return null;
    const msg = byId.get(id);
    byId.delete(id);
    byId.set(id, msg);
    return msg;
  }

  return { remember, get };
}

export function pollCreationMessageSecret(pollCreation) {
  return pollCreation?.message?.messageContextInfo?.messageSecret
    || pollCreation?.messageContextInfo?.messageSecret
    || null;
}

function uniqueStrings(values) {
  const seen = new Set();
  const out = [];
  for (const value of values || []) {
    const text = String(value || '').trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

export function pollUpdateForAggregation({
  pollUpdateMessage,
  pollUpdateMessageKey,
  pollCreation,
  decryptPollVote,
  getKeyAuthor,
  meId = 'me',
  pollCreatorJids = [],
  voterJids = [],
}) {
  if (!pollUpdateMessage) return null;
  const updateKey = pollUpdateMessage.pollUpdateMessageKey
    || pollUpdateMessageKey
    || pollUpdateMessage.key;
  if (!updateKey) return null;

  if (pollUpdateMessage.vote?.selectedOptions) {
    return {
      pollUpdateMessageKey: updateKey,
      vote: pollUpdateMessage.vote,
      senderTimestampMs: pollUpdateMessage.senderTimestampMs,
    };
  }

  const creationKey = pollUpdateMessage.pollCreationMessageKey;
  const secret = pollCreationMessageSecret(pollCreation);
  if (
    !creationKey?.id
    || !secret
    || !pollUpdateMessage.vote?.encPayload
    || !pollUpdateMessage.vote?.encIv
    || typeof decryptPollVote !== 'function'
    || typeof getKeyAuthor !== 'function'
  ) {
    return null;
  }

  // Baileys poll decryption keys include both creator and voter JIDs.  On
  // WhatsApp LID chats, the poll creator can be the linked-device LID even
  // when sock.user.id is the classic @s.whatsapp.net JID.  Try the exact
  // candidates the live bridge knows before falling back to the generic helper.
  const creatorCandidates = uniqueStrings([
    ...pollCreatorJids,
    getKeyAuthor(creationKey, meId),
  ]);
  const voterCandidates = uniqueStrings([
    ...voterJids,
    getKeyAuthor(updateKey, meId),
  ]);

  let lastError = null;
  for (const pollCreatorJid of creatorCandidates) {
    for (const voterJid of voterCandidates) {
      try {
        const vote = decryptPollVote(pollUpdateMessage.vote, {
          pollCreatorJid,
          pollMsgId: creationKey.id,
          pollEncKey: secret,
          voterJid,
        });
        return {
          pollUpdateMessageKey: updateKey,
          vote,
          senderTimestampMs: pollUpdateMessage.senderTimestampMs,
        };
      } catch (err) {
        lastError = err;
      }
    }
  }
  if (lastError) throw lastError;
  return null;
}

export function buildTextSendPayload(text, { replyTo, messageStore, mentions } = {}) {
  const content = { text };
  const options = {};
  // Auto-extract @<digits> mentions from the text if not explicitly passed
  if (!Array.isArray(mentions) || mentions.length === 0) {
    const matches = [...text.matchAll(/(?<![\w@])@(\d{8,})\b/g)];
    if (matches.length > 0) {
      mentions = [...new Set(matches.map((m) => m[1]))];
    }
  }
  if (Array.isArray(mentions) && mentions.length > 0) {
    content.mentions = mentions.map((m) => {
      const raw = String(m).trim();
      if (!raw) return m;
      if (raw.endsWith('@lid') || raw.endsWith('@s.whatsapp.net')) return raw;
      const digits = raw.replace(/\D/g, '');
      if (!digits) return m;
      return `${digits}@s.whatsapp.net`;
    });
  }
  const quoted = messageStore?.get(replyTo);
  if (quoted?.key && quoted?.message) {
    // Baileys expects quoted messages as sendMessage options, not inside the
    // message content payload. Keeping this split avoids silently sending a
    // literal/ignored `quoted` field instead of a native WhatsApp reply.
    options.quoted = quoted;
  }
  return { content, options };
}

export function buildLocationPayload({ latitude, longitude, name, address } = {}) {
  const lat = Number(latitude);
  const lon = Number(longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new Error('latitude and longitude must be numbers');
  }
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    throw new Error('latitude/longitude out of range');
  }

  const location = {
    degreesLatitude: lat,
    degreesLongitude: lon,
  };
  if (name) location.name = String(name);
  if (address) location.address = String(address);
  return { location };
}

function textFromQuotedMessage(quotedMessage) {
  if (!quotedMessage) return '';
  if (quotedMessage.conversation) return quotedMessage.conversation;
  if (quotedMessage.extendedTextMessage?.text) return quotedMessage.extendedTextMessage.text;
  if (quotedMessage.imageMessage?.caption) return quotedMessage.imageMessage.caption;
  if (quotedMessage.videoMessage?.caption) return quotedMessage.videoMessage.caption;
  if (quotedMessage.documentMessage?.caption) return quotedMessage.documentMessage.caption;
  if (quotedMessage.documentMessage?.fileName) return `[Document: ${quotedMessage.documentMessage.fileName}]`;
  if (quotedMessage.locationMessage) return formatLocationText(quotedMessage.locationMessage, false);
  if (quotedMessage.contactMessage) return formatContactText(quotedMessage.contactMessage);
  if (quotedMessage.pollCreationMessage) return formatPollText(quotedMessage.pollCreationMessage);
  return '';
}

function mediaExtForMime(mime, fallback) {
  const normalized = String(mime || '').split(';', 1)[0].toLowerCase();
  const extMap = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'video/mp4': '.mp4',
    'video/quicktime': '.mov',
    'video/x-matroska': '.mkv',
    'audio/ogg': '.ogg',
    'audio/mp4': '.m4a',
    'audio/mpeg': '.mp3',
    'application/pdf': '.pdf',
  };
  return extMap[normalized] || fallback;
}

function defaultWriteMediaFile({ buffer, dir, prefix, ext, fileName }) {
  mkdirSync(dir, { recursive: true });
  let safeName = fileName ? `_${path.basename(fileName).replace(/[^a-zA-Z0-9._-]/g, '_')}` : '';
  if (safeName && ext && !path.extname(safeName)) {
    safeName = `${safeName}${ext}`;
  }
  const filePath = path.join(dir, `${prefix}_${randomBytes(6).toString('hex')}${safeName || ext}`);
  writeFileSync(filePath, buffer);
  return filePath;
}

function formatLocationText(location, isLive) {
  const name = location.name || location.address || '';
  const lat = location.degreesLatitude ?? location.latitude;
  const lng = location.degreesLongitude ?? location.longitude;
  const kind = isLive ? 'Live location' : 'Location';
  const coords = lat !== undefined && lng !== undefined ? `${lat},${lng}` : '';
  return `[${kind}: ${[name, coords].filter(Boolean).join(' ')}]`;
}

function locationMetadata(location, isLive) {
  return {
    name: location.name || '',
    address: location.address || '',
    latitude: location.degreesLatitude ?? location.latitude ?? null,
    longitude: location.degreesLongitude ?? location.longitude ?? null,
    isLive,
  };
}

function formatContactText(contact) {
  const name = contact.displayName || contact.vcard?.match(/FN:(.+)/)?.[1] || 'unknown';
  const phone = contact.vcard?.match(/TEL[^:]*:(.+)/)?.[1] || '';
  return `[Contact: ${[name, phone].filter(Boolean).join(' ')}]`;
}

function formatContactsText(contacts) {
  const names = contacts.map(c => c.displayName).filter(Boolean);
  return `[Contacts: ${names.join(', ') || contacts.length}]`;
}

function formatReactionText(reaction) {
  const emoji = reaction.text || '';
  const target = reaction.key?.id || '';
  return `[Reaction: ${emoji}${target ? ` to ${target}` : ''}]`;
}

function pollOptions(poll) {
  return (poll.options || [])
    .map(option => option.optionName || option.name)
    .filter(Boolean);
}

function formatPollText(poll) {
  const question = poll.name || poll.title || 'poll';
  const options = pollOptions(poll);
  return `[Poll: ${question}${options.length ? ` Options: ${options.join(', ')}` : ''}]`;
}

function formatPollUpdateText(update) {
  const target = update.pollCreationMessageKey?.id || update.key?.id || '';
  return `[Poll update${target ? `: ${target}` : ''}]`;
}

/**
 * Append a visible note for media that failed to download, so the agent knows
 * something was sent rather than silently losing the attachment. Returns
 * `content` unchanged when nothing failed. (Port of nanoclaw#2895.)
 */
export function appendMediaFailureNote(content, failures) {
  if (!failures || failures.length === 0) return content;
  const note = failures.map((t) => `[${t} could not be downloaded]`).join(' ');
  return content ? `${content}\n${note}` : note;
}

export async function extractBridgeEvent({
  msg,
  chatId,
  senderId,
  senderNumber,
  botIds = [],
  isGroup = false,
  downloadMedia,
  writeMediaFile,
  cacheDirs = {},
}) {
  const messageContent = getMessageContent(msg);
  const contextInfo = getContextInfo(messageContent);
  const mentionedIds = Array.from(new Set((contextInfo?.mentionedJid || []).map(normalizeWhatsAppId).filter(Boolean)));
  const quotedMessageId = contextInfo?.stanzaId || null;
  const quotedParticipant = normalizeWhatsAppId(contextInfo?.participant || '') || null;
  const quotedRemoteJid = normalizeWhatsAppId(contextInfo?.remoteJid || '') || null;
  const hasQuotedMessage = !!contextInfo?.quotedMessage;
  const quotedText = textFromQuotedMessage(contextInfo?.quotedMessage);

  let body = '';
  let hasMedia = false;
  let mediaType = '';
  let mime = '';
  let fileName = '';
  let nativeType = '';
  const mediaUrls = [];
  const nativeMetadata = {};

  const mediaFailures = [];

  const saveMedia = async ({ mediaMessage, dir, prefix, fallbackExt, fileName: name, type }) => {
    if (!downloadMedia) return;
    try {
      const buf = await downloadMedia(msg);
      const ext = mediaExtForMime(mediaMessage?.mimetype, fallbackExt);
      const writer = writeMediaFile || defaultWriteMediaFile;
      const saved = await writer({ buffer: buf, dir, prefix, ext, fileName: name });
      if (saved) mediaUrls.push(saved);
    } catch (err) {
      // A failed CDN fetch (expired media URL, transient network error) must
      // never reject out of extractBridgeEvent — that would drop this message
      // AND every remaining message in the same upsert batch. Record the
      // failure so the agent is told media was sent instead of losing it
      // silently. (Port of nanoclaw#2895's never-silently-drop guarantee; the
      // reuploadRequest recovery half is already wired in bridge.js.)
      mediaFailures.push(type || 'media');
      try {
        console.warn(`[bridge] failed to download inbound ${type || 'media'}:`, err?.message || err);
      } catch {}
    }
  };

  if (messageContent.conversation) {
    body = messageContent.conversation;
    nativeType = 'conversation';
  } else if (messageContent.extendedTextMessage?.text) {
    body = messageContent.extendedTextMessage.text;
    nativeType = 'extendedTextMessage';
  } else if (messageContent.imageMessage) {
    const item = messageContent.imageMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'image';
    nativeType = 'imageMessage';
    mime = item.mimetype || 'image/jpeg';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.image, prefix: 'img', fallbackExt: '.jpg', type: 'image' });
  } else if (messageContent.videoMessage) {
    const item = messageContent.videoMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = item.gifPlayback ? 'gif' : 'video';
    nativeType = 'videoMessage';
    mime = item.mimetype || 'video/mp4';
    nativeMetadata.video = { gifPlayback: !!item.gifPlayback };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'vid', fallbackExt: '.mp4', type: mediaType });
  } else if (messageContent.audioMessage || messageContent.pttMessage) {
    const item = messageContent.pttMessage || messageContent.audioMessage;
    hasMedia = true;
    mediaType = item.ptt || messageContent.pttMessage ? 'ptt' : 'audio';
    nativeType = messageContent.pttMessage ? 'pttMessage' : 'audioMessage';
    mime = item.mimetype || 'audio/ogg';
    nativeMetadata.audio = { ptt: mediaType === 'ptt' };
    await saveMedia({ mediaMessage: item, dir: cacheDirs.audio, prefix: 'aud', fallbackExt: '.ogg', type: 'audio' });
  } else if (messageContent.documentMessage) {
    const item = messageContent.documentMessage;
    body = item.caption || '';
    hasMedia = true;
    mediaType = 'document';
    nativeType = 'documentMessage';
    mime = item.mimetype || 'application/octet-stream';
    fileName = item.fileName || 'document';
    await saveMedia({ mediaMessage: item, dir: cacheDirs.document, prefix: 'doc', fallbackExt: '.bin', fileName, type: 'document' });
  } else if (messageContent.stickerMessage) {
    hasMedia = true;
    mediaType = 'sticker';
    nativeType = 'stickerMessage';
    mime = messageContent.stickerMessage.mimetype || 'image/webp';
    body = '[Sticker]';
    nativeMetadata.sticker = {
      animated: !!messageContent.stickerMessage.isAnimated,
      mimetype: mime,
    };
    await saveMedia({ mediaMessage: messageContent.stickerMessage, dir: cacheDirs.image, prefix: 'sticker', fallbackExt: '.webp', type: 'sticker' });
  } else if (messageContent.locationMessage || messageContent.liveLocationMessage) {
    const isLive = !!messageContent.liveLocationMessage;
    const item = messageContent.liveLocationMessage || messageContent.locationMessage;
    mediaType = isLive ? 'live_location' : 'location';
    nativeType = isLive ? 'liveLocationMessage' : 'locationMessage';
    body = formatLocationText(item, isLive);
    nativeMetadata.location = locationMetadata(item, isLive);
  } else if (messageContent.contactMessage) {
    mediaType = 'contact';
    nativeType = 'contactMessage';
    body = formatContactText(messageContent.contactMessage);
    nativeMetadata.contact = {
      displayName: messageContent.contactMessage.displayName || '',
      vcard: messageContent.contactMessage.vcard || '',
    };
  } else if (messageContent.contactsArrayMessage) {
    const contacts = messageContent.contactsArrayMessage.contacts || [];
    mediaType = 'contacts';
    nativeType = 'contactsArrayMessage';
    body = formatContactsText(contacts);
    nativeMetadata.contacts = contacts.map(contact => ({
      displayName: contact.displayName || '',
      vcard: contact.vcard || '',
    }));
  } else if (messageContent.reactionMessage) {
    mediaType = 'reaction';
    nativeType = 'reactionMessage';
    body = formatReactionText(messageContent.reactionMessage);
    nativeMetadata.reaction = {
      text: messageContent.reactionMessage.text || '',
      messageId: messageContent.reactionMessage.key?.id || '',
      remoteJid: normalizeWhatsAppId(messageContent.reactionMessage.key?.remoteJid || ''),
      participant: normalizeWhatsAppId(messageContent.reactionMessage.key?.participant || ''),
    };
  } else if (messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3) {
    const item = messageContent.pollCreationMessage || messageContent.pollCreationMessageV2 || messageContent.pollCreationMessageV3;
    mediaType = 'poll';
    nativeType = messageContent.pollCreationMessage ? 'pollCreationMessage' : messageContent.pollCreationMessageV2 ? 'pollCreationMessageV2' : 'pollCreationMessageV3';
    body = formatPollText(item);
    nativeMetadata.poll = {
      question: item.name || item.title || '',
      options: pollOptions(item),
      selectableCount: item.selectableOptionsCount || item.selectableCount || 1,
    };
  } else if (messageContent.pollUpdateMessage) {
    mediaType = 'poll_update';
    nativeType = 'pollUpdateMessage';
    body = formatPollUpdateText(messageContent.pollUpdateMessage);
    nativeMetadata.pollUpdate = messageContent.pollUpdateMessage;
  }

  // Surface failed downloads to the agent instead of silently losing the
  // attachment. Applied before the generic "[<type> received]" fallback so an
  // uncaptioned message whose download failed reads "[image could not be
  // downloaded]" rather than claiming the media arrived.
  body = appendMediaFailureNote(body, mediaFailures);

  if (hasMedia && !body) {
    body = `[${mediaType} received]`;
  }

  return {
    messageId: msg.key.id,
    chatId,
    senderId,
    senderName: msg.pushName || senderNumber,
    chatName: isGroup ? (chatId.split('@')[0]) : (msg.pushName || senderNumber),
    isGroup,
    body,
    hasMedia,
    mediaType,
    mime,
    fileName,
    nativeType,
    nativeMetadata,
    mediaUrls,
    mentionedIds,
    quotedMessageId,
    quotedParticipant,
    quotedRemoteJid,
    quotedText,
    hasQuotedMessage,
    botIds,
    timestamp: msg.messageTimestamp,
  };
}

export function inferMediaType(ext) {
  if (['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(ext)) return 'image';
  if (['mp4', 'mov', 'avi', 'mkv', '3gp'].includes(ext)) return 'video';
  if (['ogg', 'opus', 'mp3', 'wav', 'm4a'].includes(ext)) return 'audio';
  return 'document';
}

export function mediaPayloadForFile({ buffer, filePath, mediaType, caption, fileName }) {
  const ext = filePath.toLowerCase().split('.').pop();
  const type = mediaType || inferMediaType(ext);
  if (type === 'image' && ext === 'gif') {
    // Pure helper fallback: do not lie and label raw GIF bytes as mp4.
    // The live bridge tries ffmpeg conversion to WhatsApp gifPlayback video
    // before it falls back to this regular image payload.
    return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/gif' };
  }
  switch (type) {
    case 'image':
      return { image: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'image/jpeg' };
    case 'video':
      return { video: buffer, caption: caption || undefined, mimetype: MIME_MAP[ext] || 'video/mp4' };
    case 'document':
      return {
        document: buffer,
        fileName: fileName || path.basename(filePath),
        caption: caption || undefined,
        mimetype: MIME_MAP[ext] || 'application/octet-stream',
      };
    default:
      return null;
  }
}

export function buildPollPayload({ question, options, selectableCount = 1 }) {
  const cleanQuestion = String(question || '').trim();
  const cleanOptions = (options || []).map(option => String(option || '').trim()).filter(Boolean);
  if (!cleanQuestion) throw new Error('question is required');
  if (cleanOptions.length < 2) throw new Error('at least two poll options are required');
  if (cleanOptions.length > 12) throw new Error('at most 12 poll options are supported');
  const count = Math.max(1, Math.min(Number(selectableCount) || 1, cleanOptions.length));
  return {
    poll: {
      name: cleanQuestion,
      values: cleanOptions,
      selectableCount: count,
      messageSecret: randomBytes(32),
    },
  };
}

export function pollCreationMessageFromPayload(payload) {
  const poll = payload?.poll;
  if (!poll) return null;
  const values = Array.isArray(poll.values) ? poll.values : [];
  const options = values.map(value => String(value || '').trim()).filter(Boolean);
  if (!poll.name || options.length < 2) return null;
  const selectableOptionsCount = Math.max(1, Math.min(Number(poll.selectableCount) || 1, options.length));
  const message = {};
  if (poll.messageSecret) {
    message.messageContextInfo = { messageSecret: poll.messageSecret };
  }
  message[selectableOptionsCount === 1 ? 'pollCreationMessageV3' : 'pollCreationMessage'] = {
    name: String(poll.name),
    options: options.map(optionName => ({ optionName })),
    selectableOptionsCount,
  };
  return message;
}
