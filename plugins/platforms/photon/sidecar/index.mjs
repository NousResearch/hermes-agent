// Hermes Agent — Photon Spectrum private sidecar
//
// This process is launched by plugins/platforms/photon/adapter.py. It is a
// private stdio bridge to the TypeScript-only Spectrum SDK; it does not listen
// on HTTP ports and it does not expose a user-facing service.
//
// Protocol:
//   stdout JSON lines:
//     {type:"ready", pid, startedAt, protocolVersion}
//     {type:"event", event:{...normalized Spectrum event...}}
//     {type:"response", requestId, ok, data}
//     {type:"response", requestId, ok:false, error:{code,message,retryable}}
//     {type:"fatal"|"stream_error", error:{code,message,retryable}}
//   stdin JSON lines:
//     {requestId, type:"send", spaceId, text, replyTo?}
//     {requestId, type:"typing", spaceId}
//     {requestId, type:"shutdown"}
//
// Management mode:
//   node index.mjs --management
//   stdin JSON lines:
//     {requestId, type:"phones_list"}
//     {requestId, type:"phones_add", phone}
//     {requestId, type:"phones_remove", phone}
//
// Send-once mode:
//   node index.mjs --send-once
//   stdin JSON lines:
//     {requestId, type:"send", spaceId, text}
//   The sidecar initializes Spectrum for outbound delivery only and does not
//   consume app.messages.

import readline from "node:readline";

const projectId = process.env.PHOTON_PROJECT_ID;
const projectSecret = process.env.PHOTON_PROJECT_SECRET;
const managementMode =
  process.argv.includes("--management") ||
  process.env.PHOTON_SIDECAR_MODE === "management";
const sendOnceMode =
  process.argv.includes("--send-once") ||
  process.env.PHOTON_SIDECAR_MODE === "send-once";
const E164_RE = /^\+[1-9]\d{6,14}$/;
const SENT_ECHO_TTL_MS = 5 * 60 * 1000;

function emit(payload) {
  process.stdout.write(JSON.stringify(payload) + "\n");
}

function errorPayload(error) {
  const message =
    error && error.stack
      ? String(error.stack)
      : String(error && error.message ? error.message : error);
  const code = String(
    (error && (error.code || error.name || error.status || error.statusCode)) ||
      "SDK_ERROR",
  );
  return {
    code,
    message,
    retryable: isRetryableError(error, message, code),
  };
}

function isRetryableError(error, message, code) {
  const text = `${code} ${message || ""}`.toLowerCase();
  if (
    text.includes("unavailable") ||
    text.includes("deadline") ||
    text.includes("timeout") ||
    text.includes("temporar") ||
    text.includes("connection") ||
    text.includes("network") ||
    text.includes("econnreset") ||
    text.includes("econnrefused") ||
    text.includes("session")
  ) {
    return true;
  }
  const status = error && (error.status || error.statusCode || error.code);
  return (
    status === 408 ||
    status === 409 ||
    status === 429 ||
    (typeof status === "number" && status >= 500)
  );
}

function plain(value, depth = 0, seen = new Set()) {
  if (
    value == null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (typeof value === "function" || typeof value === "symbol") {
    return undefined;
  }
  if (seen.has(value)) {
    return undefined;
  }
  if (depth >= 4) {
    return String(value);
  }
  seen.add(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => plain(item, depth + 1, seen))
      .filter((item) => item !== undefined);
  }
  const out = {};
  for (const [key, item] of Object.entries(value)) {
    const normalized = plain(item, depth + 1, seen);
    if (normalized !== undefined) {
      out[key] = normalized;
    }
  }
  seen.delete(value);
  return out;
}

function firstString(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value;
    }
  }
  return "";
}

function normalizePhoneUser(user) {
  const profile = user?.profile && typeof user.profile === "object" ? user.profile : {};
  const raw = user?.raw && typeof user.raw === "object" ? user.raw : {};
  const phone = firstString(
    user?.phoneNumber,
    user?.phone_number,
    user?.phone,
    user?.submittedPhoneNumber,
    user?.submitted_phone_number,
    profile.phoneNumber,
    profile.phone_number,
    profile.submittedPhoneNumber,
    profile.submitted_phone_number,
    raw.phoneNumber,
    raw.phone_number,
    raw.submittedPhoneNumber,
    raw.submitted_phone_number,
  );
  const assignedPhoneNumber = firstString(
    user?.assignedPhoneNumber,
    user?.assigned_phone_number,
    user?.assignedNumber,
    user?.assigned_number,
    user?.imessageNumber,
    user?.iMessageNumber,
    user?.imessage_number,
    user?.photonNumber,
    user?.photon_number,
    profile.assignedPhoneNumber,
    profile.assigned_phone_number,
    profile.imessageNumber,
    profile.iMessageNumber,
    profile.imessage_number,
    profile.photonNumber,
    profile.photon_number,
    raw.assignedPhoneNumber,
    raw.assigned_phone_number,
    raw.imessageNumber,
    raw.iMessageNumber,
    raw.imessage_number,
    raw.photonNumber,
    raw.photon_number,
  );
  const phoneNumbers = [...new Set([phone, assignedPhoneNumber].filter((value) => E164_RE.test(value)))];
  return {
    phone,
    assigned_phone_number: assignedPhoneNumber,
    user_id: firstString(user?.id, user?.userId, user?.user_id, profile.id),
    phone_numbers: phoneNumbers,
    raw: plain(user),
  };
}

function phoneMatches(user, phone) {
  const normalized = normalizePhoneUser(user);
  return (
    normalized.phone === phone ||
    normalized.assigned_phone_number === phone ||
    normalized.phone_numbers.includes(phone)
  );
}

function basicAuthHeader() {
  return `Basic ${Buffer.from(`${projectId}:${projectSecret}`, "utf8").toString("base64")}`;
}

function spectrumHost() {
  return (
    process.env.PHOTON_API_HOST ||
    process.env.SPECTRUM_CLOUD_URL ||
    "https://spectrum.photon.codes"
  ).replace(/\/+$/, "");
}

class SpectrumManagementError extends Error {
  constructor(message, { code = "MANAGEMENT_ERROR", status, detail, retryable = false } = {}) {
    super(message);
    this.name = "SpectrumManagementError";
    this.code = code;
    this.status = status;
    this.detail = detail || "";
    this.retryable = retryable;
  }
}

async function spectrumRequest(path, init = {}) {
  const headers = {
    Authorization: basicAuthHeader(),
    ...(init.headers || {}),
  };
  const response = await fetch(`${spectrumHost()}${path}`, {
    ...init,
    headers,
  });
  const textBody = await response.text();
  let body = {};
  if (textBody) {
    try {
      body = JSON.parse(textBody);
    } catch {
      body = {};
    }
  }
  if (!response.ok) {
    const message = firstString(
      body?.message,
      body?.error,
      body?.detail,
      textBody,
      response.statusText,
    );
    throw new SpectrumManagementError(message || "Spectrum management request failed", {
      code: firstString(body?.code, body?.errorCode) || `HTTP_${response.status}`,
      status: response.status,
      detail: textBody,
      retryable: response.status === 408 || response.status === 429 || response.status >= 500,
    });
  }
  if (body && body.succeed === false) {
    throw new SpectrumManagementError(
      firstString(body.message, body.error, body.detail) || "Spectrum returned succeed=false",
      {
        code: firstString(body.code, body.errorCode) || "SPECTRUM_REJECTED",
        status: response.status,
        detail: textBody,
        retryable: false,
      },
    );
  }
  return body?.data ?? body ?? {};
}

function userItems(data) {
  if (Array.isArray(data)) {
    return data.filter((item) => item && typeof item === "object");
  }
  if (data?.users && Array.isArray(data.users)) {
    return data.users.filter((item) => item && typeof item === "object");
  }
  if (data?.items && Array.isArray(data.items)) {
    return data.items.filter((item) => item && typeof item === "object");
  }
  if (data?.results && Array.isArray(data.results)) {
    return data.results.filter((item) => item && typeof item === "object");
  }
  return [];
}

async function listPhoneUsers() {
  const data = await spectrumRequest(`/projects/${encodeURIComponent(projectId)}/users/`);
  return userItems(data).map((user) => normalizePhoneUser(user));
}

function validateManagementPhone(phone) {
  if (typeof phone !== "string" || !E164_RE.test(phone.trim())) {
    throw new SpectrumManagementError(
      "phone must be E.164 (format +<country-code><number>)",
      { code: "BAD_PHONE", retryable: false },
    );
  }
  return phone.trim();
}

async function handleManagementCommand(command) {
  const requestId = command.requestId;
  try {
    if (command.type === "phones_list") {
      const users = await listPhoneUsers();
      emit({
        type: "response",
        requestId,
        ok: true,
        data: { project_id: projectId, users },
      });
      return;
    }

    if (command.type === "phones_add") {
      const phone = validateManagementPhone(command.phone);
      const existing = (await listPhoneUsers()).find((user) => phoneMatches(user, phone));
      if (existing) {
        throw new SpectrumManagementError(
          `phone ${phone} already exists on this Photon project`,
          { code: "PHONE_EXISTS", retryable: false },
        );
      }
      const data = await spectrumRequest(`/projects/${encodeURIComponent(projectId)}/users/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "shared", phoneNumber: phone }),
      });
      const user = normalizePhoneUser(data);
      emit({
        type: "response",
        requestId,
        ok: true,
        data: { project_id: projectId, user },
      });
      return;
    }

    if (command.type === "phones_remove") {
      const phone = validateManagementPhone(command.phone);
      const existing = (await listPhoneUsers()).find((user) => phoneMatches(user, phone));
      if (!existing) {
        throw new SpectrumManagementError(
          `phone ${phone} does not exist on this Photon project`,
          { code: "PHONE_NOT_FOUND", retryable: false },
        );
      }
      if (!existing.user_id) {
        throw new SpectrumManagementError(
          `phone ${phone} exists but Photon did not return a user id`,
          { code: "MISSING_USER_ID", retryable: false },
        );
      }
      await spectrumRequest(
        `/projects/${encodeURIComponent(projectId)}/users/${encodeURIComponent(existing.user_id)}/`,
        { method: "DELETE" },
      );
      emit({
        type: "response",
        requestId,
        ok: true,
        data: { project_id: projectId, user: existing },
      });
      return;
    }

    if (command.type === "shutdown") {
      emit({ type: "response", requestId, ok: true, data: { stopping: true } });
      return;
    }

    throw new SpectrumManagementError(`unknown command ${command.type}`, {
      code: "BAD_PAYLOAD",
      retryable: false,
    });
  } catch (error) {
    emit({
      type: "response",
      requestId,
      ok: false,
      error: managementErrorPayload(error),
    });
  }
}

function managementErrorPayload(error) {
  const code = String(error?.code || error?.name || "MANAGEMENT_ERROR");
  const message = String(error?.message || error || "Photon management error");
  return {
    code,
    message,
    detail: String(error?.detail || ""),
    status: typeof error?.status === "number" ? error.status : undefined,
    retryable: Boolean(error?.retryable ?? isRetryableError(error, message, code)),
  };
}

async function runManagementSidecar() {
  emit({
    type: "ready",
    pid: process.pid,
    startedAt: new Date().toISOString(),
    protocolVersion: 1,
    mode: "management",
  });
  const managementRl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of managementRl) {
    if (!line.trim()) {
      continue;
    }
    let command;
    try {
      command = JSON.parse(line);
    } catch {
      emit({
        type: "error",
        error: {
          code: "BAD_JSON",
          message: "invalid sidecar command JSON",
          retryable: false,
        },
      });
      continue;
    }
    await handleManagementCommand(command);
  }
}

function normalizeContent(message) {
  const directText = firstString(message?.text, message?.body, message?.message);
  const content = message?.content;
  if (directText) {
    return { type: "text", text: directText };
  }
  if (typeof content === "string") {
    return { type: "text", text: content };
  }
  if (content && typeof content === "object") {
    const type = firstString(content.type, content.kind) || "unknown";
    if (type === "text") {
      return {
        type: "text",
        text: firstString(content.text, content.body, content.value),
      };
    }
    return {
      type,
      text: firstString(
        content.text,
        content.body,
        content.name,
        content.filename,
      ),
      name: firstString(content.name, content.filename),
      mimeType: firstString(content.mimeType, content.mime, content.contentType),
      raw: plain(content),
    };
  }
  const attachments = message?.attachments;
  if (Array.isArray(attachments) && attachments.length > 0) {
    const first = attachments[0] || {};
    return {
      type: "attachment",
      name: firstString(first.name, first.filename),
      mimeType: firstString(first.mimeType, first.mime, first.contentType),
      raw: plain(first),
    };
  }
  return { type: "unknown" };
}

function normalizeSpace(space, message) {
  const rawSpace = message?.space || {};
  const id = firstString(
    space?.id,
    rawSpace.id,
    message?.spaceId,
    message?.chatId,
    message?.conversationId,
  );
  return {
    id,
    name: firstString(space?.name, rawSpace.name, rawSpace.displayName, id),
    type: firstString(space?.type, rawSpace.type),
    raw: plain(rawSpace),
  };
}

function normalizeSender(message) {
  const sender = message?.sender || message?.from || message?.author || {};
  const id = firstString(
    sender.id,
    sender.userId,
    sender.phone,
    sender.address,
    message?.senderId,
    message?.fromId,
    message?.from,
  );
  return {
    id,
    name: firstString(sender.name, sender.displayName, sender.handle, id),
    raw: plain(sender),
  };
}

function normalizeEvent(space, message) {
  return {
    id: firstString(message?.id, message?.messageId, message?.uuid),
    platform: firstString(message?.platform, "imessage"),
    timestamp: firstString(
      message?.timestamp,
      message?.createdAt,
      message?.created_at,
    ),
    space: normalizeSpace(space, message),
    sender: normalizeSender(message),
    content: normalizeContent(message),
    raw: plain(message),
  };
}

const recentlySentMessageIds = new Map();

function rememberSentMessageId(messageId) {
  if (typeof messageId === "string" && messageId.trim()) {
    recentlySentMessageIds.set(messageId.trim(), Date.now());
  }
}

function wasRecentlySentMessageId(messageId) {
  if (typeof messageId !== "string" || !messageId.trim()) {
    return false;
  }
  const now = Date.now();
  const cutoff = now - SENT_ECHO_TTL_MS;
  for (const [id, seenAt] of recentlySentMessageIds) {
    if (seenAt < cutoff) {
      recentlySentMessageIds.delete(id);
    }
  }
  return recentlySentMessageIds.has(messageId.trim());
}

function messageIsFromSelf(message) {
  return message?.isFromMe === true;
}

function shouldSuppressSelfEcho(message, event) {
  return messageIsFromSelf(message) || wasRecentlySentMessageId(event?.id);
}

function dmAddressFromSpaceId(spaceId) {
  if (typeof spaceId !== "string") {
    return "";
  }
  if (spaceId.startsWith("any;-;")) {
    return spaceId.slice("any;-;".length);
  }
  if (spaceId.startsWith("+")) {
    return spaceId;
  }
  return "";
}

if (!projectId || !projectSecret) {
  emit({
    type: "fatal",
    error: {
      code: "MISSING_CREDENTIALS",
      message: "PHOTON_PROJECT_ID and PHOTON_PROJECT_SECRET are required",
      retryable: false,
    },
  });
  process.exit(2);
}

if (managementMode) {
  await runManagementSidecar();
  process.exit(0);
}

let Spectrum;
let imessage;
let spectrumText;
try {
  ({ Spectrum, text: spectrumText } = await import("spectrum-ts"));
  ({ imessage } = await import("spectrum-ts/providers/imessage"));
} catch (error) {
  emit({
    type: "fatal",
    error: {
      code: "MISSING_SPECTRUM_SDK",
      message:
        "spectrum-ts is not installed. Run npm install inside plugins/platforms/photon/sidecar. " +
        (error && error.message ? error.message : String(error)),
      retryable: false,
    },
  });
  process.exit(3);
}

let app;
try {
  app = await Spectrum({
    projectId,
    projectSecret,
    providers: [imessage.config()],
  });
} catch (error) {
  emit({ type: "fatal", error: errorPayload(error) });
  process.exit(4);
}

const cachedSpaces = new Map();

function cacheSpace(space) {
  if (
    space &&
    typeof space.id === "string" &&
    typeof space.send === "function"
  ) {
    cachedSpaces.set(space.id, space);
  }
}

async function resolveSpace(spaceId) {
  const cached = cachedSpaces.get(spaceId);
  if (cached) {
    return cached;
  }
  const provider = imessage(app);
  const address = dmAddressFromSpaceId(spaceId);
  if (address && typeof provider.space === "function") {
    const space = await provider.space(address);
    cacheSpace(space);
    return space;
  }
  throw Object.assign(new Error(`unable to resolve Spectrum space ${spaceId}`), {
    code: "SPACE_NOT_FOUND",
  });
}

let shuttingDown = false;
async function shutdown(reason) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  try {
    await Promise.race([
      app.stop(),
      new Promise((resolve) => setTimeout(resolve, 3000)),
    ]);
  } catch (error) {
    emit({
      type: "log",
      level: "debug",
      message: `app.stop failed during ${reason}: ${String(error)}`,
    });
  }
  process.exit(0);
}

process.on("SIGINT", () => void shutdown("SIGINT"));
process.on("SIGTERM", () => void shutdown("SIGTERM"));

async function handleCommand(command) {
  const requestId = command.requestId;
  const type = command.type;
  try {
    if (type === "shutdown") {
      emit({ type: "response", requestId, ok: true, data: { stopping: true } });
      await shutdown("command");
      return;
    }

    if (type === "send") {
      const spaceId = command.spaceId;
      const messageText = command.text;
      if (!spaceId || typeof messageText !== "string") {
        throw Object.assign(new Error("spaceId and text are required"), {
          code: "BAD_PAYLOAD",
          retryable: false,
        });
      }
      const space = await resolveSpace(spaceId);
      if (command.replyTo) {
        emit({
          type: "log",
          level: "debug",
          message:
            "replyTo is not supported by spectrum-ts send yet; sending a plain message",
        });
      }
      const result = await space.send(spectrumText(messageText));
      const messageId = result?.id || result?.messageId || null;
      rememberSentMessageId(messageId);
      emit({
        type: "response",
        requestId,
        ok: true,
        data: {
          messageId,
          raw: plain(result),
        },
      });
      return;
    }

    if (type === "typing") {
      if (!command.spaceId) {
        throw Object.assign(new Error("spaceId is required"), {
          code: "BAD_PAYLOAD",
          retryable: false,
        });
      }
      const space = await resolveSpace(command.spaceId);
      if (typeof space.typing === "function") {
        await space.typing();
      } else if (typeof space.setTyping === "function") {
        await space.setTyping(true);
      }
      emit({ type: "response", requestId, ok: true, data: {} });
      return;
    }

    throw Object.assign(new Error(`unknown command ${type}`), {
      code: "BAD_PAYLOAD",
      retryable: false,
    });
  } catch (error) {
    const payload = errorPayload(error);
    if (error && Object.prototype.hasOwnProperty.call(error, "retryable")) {
      payload.retryable = Boolean(error.retryable);
    }
    emit({ type: "response", requestId, ok: false, error: payload });
  }
}

async function runSendOnceSidecar() {
  emit({
    type: "ready",
    pid: process.pid,
    startedAt: new Date().toISOString(),
    protocolVersion: 1,
    mode: "send-once",
  });

  const sendOnceRl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });
  for await (const line of sendOnceRl) {
    if (!line.trim()) {
      continue;
    }
    let command;
    try {
      command = JSON.parse(line);
    } catch {
      emit({
        type: "error",
        error: {
          code: "BAD_JSON",
          message: "invalid sidecar command JSON",
          retryable: false,
        },
      });
      break;
    }
    await handleCommand(command);
    break;
  }
  await shutdown("send-once-complete");
}

if (sendOnceMode) {
  await runSendOnceSidecar();
  process.exit(0);
}

(async () => {
  try {
    for await (const [space, message] of app.messages) {
      cacheSpace(space);
      const event = normalizeEvent(space, message);
      if (shouldSuppressSelfEcho(message, event)) {
        emit({
          type: "log",
          level: "debug",
          message: `suppressed self echo${event.id ? ` (${event.id})` : ""}`,
        });
        continue;
      }
      emit({ type: "event", event });
    }
    emit({
      type: "stream_error",
      error: {
        code: "STREAM_ENDED",
        message: "Spectrum inbound stream ended",
        retryable: true,
      },
    });
    process.exit(12);
  } catch (error) {
    emit({ type: "stream_error", error: errorPayload(error) });
    process.exit(12);
  }
})();

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  let command;
  try {
    command = JSON.parse(line);
  } catch {
    emit({
      type: "error",
      error: {
        code: "BAD_JSON",
        message: "invalid sidecar command JSON",
        retryable: false,
      },
    });
    return;
  }
  void handleCommand(command);
});

emit({
  type: "ready",
  pid: process.pid,
  startedAt: new Date().toISOString(),
  protocolVersion: 1,
});
