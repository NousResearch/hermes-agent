// RFC 8621 §4.4: charset MUST be omitted when partId is given; blob-backed text/*
// parts should include charset (type strips Content-Type parameters).
const DEFAULT_TEXT_CHARSET = "utf-8";
function baseMediaType(type) {
    const semi = type.indexOf(";");
    return (semi === -1 ? type : type.slice(0, semi)).trim().toLowerCase();
}
function isTextStarType(type) {
    return baseMediaType(type).startsWith("text/");
}
/** RFC 8621 EmailBodyPart: never add charset when partId is set. */
function ensureCharsetOnBodyPart(part) {
    if (!part || typeof part !== "object" || Array.isArray(part))
        return;
    const o = part;
    if (typeof o.partId === "string" && o.partId.length > 0)
        return;
    if (typeof o.blobId !== "string" || o.blobId.length === 0)
        return;
    const t = o.type;
    if (typeof t !== "string" || !isTextStarType(t))
        return;
    if (o.charset != null && o.charset !== "")
        return;
    o.charset = DEFAULT_TEXT_CHARSET;
}
function normalizeBodyPartArray(arr) {
    if (!Array.isArray(arr))
        return;
    for (const item of arr)
        ensureCharsetOnBodyPart(item);
}
function normalizeEmailSetArg(arg) {
    if (!arg || typeof arg !== "object" || Array.isArray(arg))
        return;
    const create = arg["create"];
    if (!create || typeof create !== "object" || Array.isArray(create))
        return;
    for (const email of Object.values(create)) {
        if (!email || typeof email !== "object" || Array.isArray(email))
            continue;
        const e = email;
        normalizeBodyPartArray(e.attachments);
        normalizeBodyPartArray(e.textBody);
        normalizeBodyPartArray(e.htmlBody);
    }
}
/**
 * Ensures blob-backed `text/*` body parts in `Email/set` `create` include
 * `charset` when omitted, for strict JMAP servers. Skips parts with `partId`
 * (RFC 8621 forbids charset there).
 */
export function ensureTextCharsetOnEmailSetBlobParts(envelope) {
    for (const call of envelope.methodCalls) {
        if (!Array.isArray(call) || call.length < 2)
            continue;
        if (call[0] !== "Email/set")
            continue;
        normalizeEmailSetArg(call[1]);
    }
}
