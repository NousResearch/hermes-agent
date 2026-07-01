// JMAP envelope parsing, preset paths, $VAR substitution, and HTTP helpers.
import { readFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve as resolvePath } from "node:path";
import { cwd } from "node:process";
import { fileURLToPath } from "node:url";
import { tryReadSharedJson } from "../../core/shared-assets.js";
import { readCredentials } from "../session/agent-credentials-store.js";
import { inboxIdToMailboxEmail } from "../session/inbox-id-to-mailbox-email.js";
import { assertBlobUploadEnvelopeWithinLimits, } from "./agent-jmap-blob-limits.js";
import { buildVarsFromAttachmentFiles, } from "./agent-jmap-blob-upload.js";
import { ensureTextCharsetOnEmailSetBlobParts } from "./agent-jmap-email-charset.js";
import { substituteVars } from "./agent-vars.js";
export const DEFAULT_JMAP_USING = [
    "urn:ietf:params:jmap:core",
    "urn:ietf:params:jmap:mail",
];
/** Presets shipped with MCP / skill npm packages (for error hints). */
const sharedManifest = tryReadSharedJson("manifest.json");
const sharedHints = tryReadSharedJson("messages/hints.json");
export const BUNDLED_OPS_PRESET_NAMES = [
    "list_inbox.json",
    "reply.json",
    "send_mail.json",
    "send_mail_attachment.json",
    "send_mail_blob_attachment.json",
];
export const JMAP_MAIL_URN = "urn:ietf:params:jmap:mail";
/** RFC 9404 blob extension URN (Blob/upload, Blob/get, Blob/lookup). */
export const JMAP_BLOB_URN = "urn:ietf:params:jmap:blob";
export function parseJmapEnvelope(raw, defaultUsing, source) {
    let value;
    try {
        value = JSON.parse(raw);
    }
    catch (err) {
        throw new Error(`${source} is not valid JSON: ${err.message}`);
    }
    if (Array.isArray(value)) {
        return { using: [...defaultUsing], methodCalls: value };
    }
    if (value !== null &&
        typeof value === "object" &&
        Array.isArray(value.methodCalls)) {
        const obj = value;
        const using = Array.isArray(obj.using)
            ? obj.using.filter((u) => typeof u === "string")
            : [...defaultUsing];
        return { using, methodCalls: obj.methodCalls };
    }
    throw new Error(`${source} must be a methodCalls array, e.g. ` +
        '[["Mailbox/get",{...},"m0"]], or an object with a methodCalls array.');
}
export function resolveOpsFilePath(credentialDir, opsFile) {
    return isAbsolute(opsFile) ? opsFile : resolvePath(credentialDir, opsFile);
}
export async function readOpsFile(credentialDir, opsFile) {
    const filePath = resolveOpsFilePath(credentialDir, opsFile);
    try {
        return await readFile(filePath, "utf-8");
    }
    catch (err) {
        if (!(err instanceof Error) || !isFileNotFound(err) || isAbsolute(opsFile)) {
            throw err;
        }
    }
    const bundledPath = await resolveBundledPresetPath(opsFile);
    if (!bundledPath) {
        throw new Error(`ops_file '${opsFile}' not found under credential directory (${filePath}) ` +
            "and not among bundled presets: " +
            `${BUNDLED_OPS_PRESET_NAMES.join(", ")}.`);
    }
    return await readFile(bundledPath, "utf-8");
}
function isFileNotFound(err) {
    const code = err.code;
    return code === "ENOENT" || code === "ENOTDIR";
}
async function resolveBundledPresetPath(opsFile) {
    const moduleDir = dirname(fileURLToPath(globalThis[Symbol.for("import-meta-ponyfill-esmodule")](import.meta).url));
    let currentDir = moduleDir;
    for (let depth = 0; depth < 8; depth++) {
        const candidates = [
            resolvePath(currentDir, "shared", sharedManifest?.presets_dir ?? "presets", opsFile),
            resolvePath(currentDir, "presets", opsFile),
            resolvePath(currentDir, "agent", "jmap", "presets", opsFile),
            resolvePath(currentDir, "lib", "src", "agent", "jmap", "presets", opsFile),
        ];
        for (const candidate of candidates) {
            try {
                await readFile(candidate, "utf-8");
                return candidate;
            }
            catch (err) {
                if (!(err instanceof Error) || !isFileNotFound(err)) {
                    throw err;
                }
            }
        }
        const parent = resolvePath(currentDir, "..");
        if (parent === currentDir)
            break;
        currentDir = parent;
    }
    return undefined;
}
export function extractPrimaryMailAccountId(session) {
    const primary = session["primaryAccounts"];
    if (!primary || typeof primary !== "object") {
        throw new Error("JMAP session missing primaryAccounts.");
    }
    const id = primary[JMAP_MAIL_URN];
    if (typeof id !== "string" || id.length === 0) {
        throw new Error(`JMAP session missing primaryAccounts['${JMAP_MAIL_URN}'].`);
    }
    return id;
}
export function extractBlobEndpoints(session) {
    const uploadUrl = session["uploadUrl"];
    const downloadUrl = session["downloadUrl"];
    if (typeof uploadUrl !== "string" || uploadUrl.length === 0) {
        throw new Error("JMAP session missing uploadUrl.");
    }
    if (typeof downloadUrl !== "string" || downloadUrl.length === 0) {
        throw new Error("JMAP session missing downloadUrl.");
    }
    return { uploadUrl, downloadUrl };
}
/** RFC 8620 §2 / §3.1: POST target for JMAP API calls from the Session object. */
export function extractJmapApiUrl(session) {
    const u = session["apiUrl"];
    if (typeof u !== "string" || u.length === 0) {
        throw new Error("JMAP session missing apiUrl.");
    }
    return u;
}
function asNonNegativeInt(v) {
    if (typeof v !== "number" || !Number.isFinite(v))
        return undefined;
    if (!Number.isInteger(v) || v < 0 || v > Number.MAX_SAFE_INTEGER) {
        return undefined;
    }
    return v;
}
/**
 * RFC 9404 §3.1 blob limits for one account from GET /.well-known/jmap JSON.
 * Returns null when the account does not advertise `urn:ietf:params:jmap:blob`.
 */
export function extractBlobUploadLimits(session, accountId) {
    const accounts = session["accounts"];
    if (!accounts || typeof accounts !== "object")
        return null;
    const acc = accounts[accountId];
    if (!acc || typeof acc !== "object")
        return null;
    const caps = acc["accountCapabilities"];
    if (!caps || typeof caps !== "object")
        return null;
    const blob = caps[JMAP_BLOB_URN];
    if (!blob || typeof blob !== "object")
        return null;
    const b = blob;
    let maxSizeBlobSet = null;
    const rawMax = b["maxSizeBlobSet"];
    if (rawMax === null) {
        maxSizeBlobSet = null;
    }
    else {
        const n = asNonNegativeInt(rawMax);
        maxSizeBlobSet = n === undefined ? null : n;
    }
    const maxDs = asNonNegativeInt(b["maxDataSources"]);
    const out = { maxSizeBlobSet };
    if (maxDs !== undefined) {
        out.maxDataSources = maxDs;
    }
    return out;
}
export async function fetchJmapWellKnown(apiUrl, capabilityJwt) {
    const base = apiUrl.replace(/\/+$/, "");
    const res = await fetch(`${base}/.well-known/jmap`, {
        headers: { Authorization: `Bearer ${capabilityJwt}` },
    });
    const text = await res.text();
    if (!res.ok) {
        throw new Error(`JMAP session fetch failed (HTTP ${res.status}): ${text}`);
    }
    try {
        return JSON.parse(text);
    }
    catch {
        throw new Error("JMAP session response is not valid JSON.");
    }
}
/**
 * Parse ops JSON, substitute `$VAR_NAME` tokens (session + caller vars), POST to JMAP.
 */
export async function runJmapRequest(input) {
    if (input.dryRun && input.attachments && input.attachments.length > 0) {
        throw new Error("dryRun cannot be used with attachments: RFC 8620 upload runs first and would create blobs.");
    }
    let mergedVars = input.vars ?? {};
    if (input.attachments && input.attachments.length > 0) {
        const pathBase = input.attachmentPathBase ?? cwd();
        const injected = await buildVarsFromAttachmentFiles(input.session, input.attachments, pathBase);
        mergedVars = { ...injected, ...mergedVars };
    }
    const { text: raw } = await substituteVars({
        raw: input.opsJson,
        vars: mergedVars,
        autoResolvers: {
            ACCOUNT_ID: () => input.session.getPrimaryMailAccountId(),
            INBOX: async () => {
                const raw = input.session.currentInboxId ??
                    (input.session.files
                        ? (await readCredentials(input.session.files.credentialsFile))
                            .inboxId
                        : undefined);
                if (!raw) {
                    throw new Error("No inbox in session; run register first.");
                }
                return inboxIdToMailboxEmail(raw);
            },
            INBOX_MAILBOX_ID: () => fetchInboxMailboxId(input.session),
            UPLOAD_URL: async () => {
                if (input.session.currentUploadUrl) {
                    return input.session.currentUploadUrl;
                }
                if (input.session.files) {
                    return (await readCredentials(input.session.files.credentialsFile))
                        .uploadUrl;
                }
                throw new Error("JMAP session missing uploadUrl.");
            },
            DOWNLOAD_URL: async () => {
                if (input.session.currentDownloadUrl) {
                    return input.session.currentDownloadUrl;
                }
                if (input.session.files) {
                    return (await readCredentials(input.session.files.credentialsFile))
                        .downloadUrl;
                }
                throw new Error("JMAP session missing downloadUrl.");
            },
        },
    });
    const envelope = parseJmapEnvelope(raw, input.defaultUsing, input.sourceLabel);
    ensureTextCharsetOnEmailSetBlobParts(envelope);
    await enforceJmapBlobUploadLimitsIfApplicable(input.session, envelope);
    const jmapPostUrl = await input.session.getJmapPostUrl();
    if (input.dryRun) {
        return {
            ok: true,
            status: 200,
            bodyText: JSON.stringify({
                dryRun: true,
                url: jmapPostUrl,
                envelope,
            }, null, 2),
        };
    }
    const capabilityJwt = await input.session.getCapabilityToken();
    const { ok, status, bodyText } = await postJmap(jmapPostUrl, capabilityJwt, envelope);
    if (!ok) {
        return { ok, status, bodyText };
    }
    return { ok, status, bodyText: attachJmapNextHints(bodyText) };
}
/**
 * Resolves the JMAP `Mailbox` id for the account inbox (`role: "inbox"`).
 * Used for `$INBOX_MAILBOX_ID` substitution (distinct from `$INBOX`, which is
 * the mailbox *email address* — see `inboxIdToMailboxEmail` for normalization).
 */
export async function fetchInboxMailboxId(port) {
    const accountId = await port.getPrimaryMailAccountId();
    const capabilityJwt = await port.getCapabilityToken();
    const envelope = {
        using: [
            "urn:ietf:params:jmap:core",
            "urn:ietf:params:jmap:mail",
        ],
        methodCalls: [
            [
                "Mailbox/query",
                { accountId, filter: { role: "inbox" } },
                "mq0",
            ],
        ],
    };
    const jmapPostUrl = await port.getJmapPostUrl();
    const { ok, status, bodyText } = await postJmap(jmapPostUrl, capabilityJwt, envelope);
    if (!ok) {
        throw new Error(`Mailbox/query failed (HTTP ${status}): ${bodyText}`);
    }
    let parsed;
    try {
        parsed = JSON.parse(bodyText);
    }
    catch {
        throw new Error("Mailbox/query response is not valid JSON.");
    }
    const responses = parsed
        .methodResponses;
    const first = responses?.[0];
    if (!Array.isArray(first) || first[0] === "error") {
        throw new Error(`Mailbox/query failed: ${bodyText}`);
    }
    if (first[0] !== "Mailbox/query") {
        throw new Error(`Mailbox/query failed: ${bodyText}`);
    }
    const payload = first[1];
    const id = payload.ids?.[0];
    if (typeof id !== "string" || id.length === 0) {
        throw new Error("Mailbox/query returned no inbox mailbox id.");
    }
    return id;
}
function collectBlobUploadAccountIds(envelope) {
    const ids = new Set();
    for (const call of envelope.methodCalls) {
        if (!Array.isArray(call) || call[0] !== "Blob/upload")
            continue;
        const arg = call[1];
        if (!arg || typeof arg !== "object")
            continue;
        const aid = arg["accountId"];
        if (typeof aid === "string" && aid.length > 0)
            ids.add(aid);
    }
    return [...ids];
}
async function enforceJmapBlobUploadLimitsIfApplicable(session, envelope) {
    if (!envelope.using.includes(JMAP_BLOB_URN))
        return;
    const hasUpload = envelope.methodCalls.some((c) => Array.isArray(c) && c[0] === "Blob/upload");
    if (!hasUpload)
        return;
    const accountIds = collectBlobUploadAccountIds(envelope);
    const limitsByAccount = new Map();
    for (const id of accountIds) {
        limitsByAccount.set(id, await session.getBlobUploadLimitsForAccount(id));
    }
    assertBlobUploadEnvelopeWithinLimits(envelope, limitsByAccount);
}
export async function postJmap(jmapPostUrl, capabilityJwt, envelope) {
    const res = await fetch(jmapPostUrl, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${capabilityJwt}`,
        },
        body: JSON.stringify(envelope),
    });
    const bodyText = await res.text();
    return { ok: res.ok, status: res.status, bodyText };
}
const JMAP_NEXT_HINTS = sharedHints?.jmap_next_hints ?? [
    "Use jmap_request with Mailbox/get or Email/query to work with mail data.",
    "Use presets with $VAR placeholders — $ACCOUNT_ID, $INBOX, and $INBOX_MAILBOX_ID come from the session; pass others via vars / --vars.",
    "Call help for the JMAP cheatsheet and troubleshooting.",
];
/** Attach _next hints to a successful JMAP JSON object when parseable. */
export function attachJmapNextHints(bodyText) {
    try {
        const obj = JSON.parse(bodyText);
        if (obj && typeof obj === "object" && !Array.isArray(obj)) {
            return JSON.stringify({ ...obj, _next: [...JMAP_NEXT_HINTS] }, null, 2);
        }
    }
    catch {
        // not JSON — return raw
    }
    return bodyText;
}
