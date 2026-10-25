import fs from "node:fs";

export const MAX_ENVELOPE_BYTES = 4096;
export const MAX_QUESTION_CHARS = 1024;
export const MAX_ANSWER_CHARS = 2048;
export const MAX_CORRELATION_CHARS = 128;
export const MAX_TTL_MS = 30_000;
export const SDK_INTEGRITY =
  "sha512-SWsVtH/0vULSQStoZ6xxH2K09Li8fXYTMC64ZeogCdpZ3XSi1eCPOByV1OpuJxuI3OcmHhosDWYyii07tpPsLw==";

const ROUTE_ENV = Object.freeze({
  ticket_ref: "ESN0407_TICKET_REF",
  assignment_id: "ESN0407_ASSIGNMENT_ID",
  attempt_id: "ESN0407_ATTEMPT_ID",
  owner_identity: "ESN0407_OWNER_IDENTITY",
  ticket_revision: "ESN0407_TICKET_REVISION",
  authority_event_sha256: "ESN0407_AUTHORITY_EVENT_SHA256",
  correlation_id: "ESN0407_CORRELATION_ID",
  route_nonce: "ESN0407_ROUTE_NONCE",
});

export const FORBIDDEN_PROVIDER_ENV = Object.freeze([
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_AUTH_TOKEN",
  "ANTHROPIC_BASE_URL",
  "CLAUDE_CODE_USE_BEDROCK",
  "CLAUDE_CODE_USE_VERTEX",
  "CLAUDE_CODE_USE_FOUNDRY",
  "AWS_ACCESS_KEY_ID",
  "AWS_SECRET_ACCESS_KEY",
  "AWS_SESSION_TOKEN",
  "AWS_PROFILE",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "ANTHROPIC_VERTEX_PROJECT_ID",
  "AZURE_CLIENT_ID",
  "AZURE_CLIENT_SECRET",
  "AZURE_TENANT_ID",
]);

function requiredValue(value, name) {
  if (typeof value !== "string" || !value || /^__.*__$/.test(value)) {
    throw new Error(`${name} must be a resolved non-placeholder value`);
  }
  return value;
}

export function readExpectedRoute(env) {
  const route = {};
  for (const [field, name] of Object.entries(ROUTE_ENV)) {
    route[field] = requiredValue(env[name], name);
  }
  route.ticket_revision = Number(route.ticket_revision);
  if (!Number.isInteger(route.ticket_revision) || route.ticket_revision < 1) {
    throw new Error("ESN0407_TICKET_REVISION must be a positive integer");
  }
  if (!/^ESN-[0-9]{3,}$/.test(route.ticket_ref)) {
    throw new Error("ESN0407_TICKET_REF must be a canonical ESN ticket");
  }
  if (!/^[0-9a-f]{64}$/.test(route.authority_event_sha256)) {
    throw new Error("ESN0407_AUTHORITY_EVENT_SHA256 must be a lowercase SHA-256 digest");
  }
  if (!/^[0-9a-f]{64}$/.test(route.route_nonce)) {
    throw new Error("ESN0407_ROUTE_NONCE must be a 256-bit lowercase hex nonce");
  }
  if (route.correlation_id.length > MAX_CORRELATION_CHARS) {
    throw new Error("ESN0407_CORRELATION_ID is too long");
  }
  return Object.freeze(route);
}

export function validateQuestion(value, expectedRoute) {
  const required = [
    "correlation_id",
    "ticket_ref",
    "assignment_id",
    "attempt_id",
    "owner_identity",
    "authority_event_sha256",
    "route_nonce",
    "question",
  ];
  if (value?.type !== "question" || value?.authority_scope !== "answer-only-no-changes") {
    throw new Error("unbounded question envelope");
  }
  for (const key of required) requiredValue(value[key], key);
  if (value.correlation_id.length > MAX_CORRELATION_CHARS) {
    throw new Error("correlation_id is too long");
  }
  if (!Number.isInteger(value.ticket_revision) || value.ticket_revision < 1) {
    throw new Error("invalid ticket_revision");
  }
  if (!/^[0-9a-f]{64}$/.test(value.authority_event_sha256)) {
    throw new Error("invalid authority digest");
  }
  if (value.question.length > MAX_QUESTION_CHARS) throw new Error("question too large");
  for (const [key, expected] of Object.entries(expectedRoute)) {
    if (value[key] !== expected) throw new Error(`route mismatch: ${key}`);
  }
  const ttlMs = Number(value.ttl_ms);
  if (!Number.isInteger(ttlMs) || ttlMs < 1 || ttlMs > MAX_TTL_MS) {
    throw new Error("invalid ttl_ms");
  }
  return ttlMs;
}

export function validateReply(argumentsValue, expectedRoute) {
  const correlationId = argumentsValue?.correlation_id;
  const answer = argumentsValue?.answer;
  if (typeof correlationId !== "string" || typeof answer !== "string") {
    throw new Error("invalid reply envelope");
  }
  if (correlationId.length > MAX_CORRELATION_CHARS) {
    throw new Error("reply correlation_id is too long");
  }
  if (correlationId !== expectedRoute.correlation_id) {
    throw new Error("reply correlation_id does not match the authorized route");
  }
  if (answer.length > MAX_ANSWER_CHARS) throw new Error("answer too large");
  return { correlationId, answer };
}

export function assertSubscriptionAuth(status, env) {
  const present = FORBIDDEN_PROVIDER_ENV.filter((name) => Boolean(env[name]));
  if (present.length) throw new Error(`provider credential environment is not scrubbed: ${present.join(",")}`);
  if (
    status?.loggedIn !== true ||
    status?.authMethod !== "claude.ai" ||
    status?.apiProvider !== "firstParty" ||
    typeof status?.subscriptionType !== "string" ||
    !status.subscriptionType
  ) {
    throw new Error("Claude subscription authentication is required; API-key fallback is refused");
  }
}

export function assertLockedDependencies(lock) {
  const root = lock?.packages?.[""];
  const sdk = lock?.packages?.["node_modules/@modelcontextprotocol/sdk"];
  if (root?.dependencies?.["@modelcontextprotocol/sdk"] !== "1.32.2") {
    throw new Error("lockfile root does not pin MCP SDK 1.32.2");
  }
  if (sdk?.version !== "1.32.2" || sdk?.integrity !== SDK_INTEGRITY) {
    throw new Error("lockfile MCP SDK version or integrity mismatch");
  }
  for (const [name, item] of Object.entries(lock.packages ?? {})) {
    if (!name || item.link) continue;
    if (typeof item.version !== "string" || typeof item.integrity !== "string") {
      throw new Error(`unlocked transitive dependency: ${name}`);
    }
  }
}

export function unlinkOwnedSocket(socketPath) {
  try {
    const stat = fs.lstatSync(socketPath);
    if (stat.isSocket()) fs.unlinkSync(socketPath);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

export async function closeChannelResources({
  localServer,
  activeSockets,
  pending,
  socketPath,
}) {
  for (const waiting of pending.values()) clearTimeout(waiting.timer);
  pending.clear();
  for (const socket of activeSockets) socket.destroy();
  await new Promise((resolve) => {
    if (localServer.listening) localServer.close(resolve);
    else resolve();
  });
  unlinkOwnedSocket(socketPath);
}
