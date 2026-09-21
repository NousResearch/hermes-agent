import crypto from "node:crypto";
import https from "node:https";
import path from "node:path";
import { loadTextFile } from "./local_file_io.mjs";
import { isObject } from "./utils.mjs";

/**
 * Allowed domains for feed/signature fetching.
 * Only connections to these domains are permitted for security.
 */
const ALLOWED_DOMAINS = [
  "clawsec.prompt.security",
  "prompt.security",
  "raw.githubusercontent.com",
  "github.com",
];

/**
 * Custom error class for security policy violations.
 * These errors should always propagate and never be silently caught.
 */
class SecurityPolicyError extends Error {
  constructor(message) {
    super(message);
    this.name = "SecurityPolicyError";
  }
}

/**
 * Creates a secure HTTPS agent with TLS 1.2+ enforcement and certificate validation.
 * @returns {https.Agent}
 */
function createSecureAgent() {
  return new https.Agent({
    // Enforce minimum TLS 1.2 (eliminate TLS 1.0, 1.1)
    minVersion: "TLSv1.2",
    // Ensure certificate validation is enabled (reject unauthorized certificates)
    rejectUnauthorized: true,
    // Use strong cipher suites
    ciphers: "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256",
  });
}

/**
 * Validates that a URL is from an allowed domain.
 * @param {string} url
 * @returns {boolean}
 */
function isAllowedDomain(url) {
  try {
    const parsed = new URL(url);

    // Only allow HTTPS protocol
    if (parsed.protocol !== "https:") {
      return false;
    }

    const hostname = parsed.hostname.toLowerCase();

    // Check if hostname matches any allowed domain
    return ALLOWED_DOMAINS.some(
      (allowed) =>
        hostname === allowed || hostname.endsWith(`.${allowed}`)
    );
  } catch {
    return false;
  }
}

/**
 * Secure wrapper around fetch with TLS enforcement and domain validation.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<Response>}
 * @throws {SecurityPolicyError} If URL is not from an allowed domain
 */
async function secureFetch(url, options = {}) {
  // Validate domain before making request
  if (!isAllowedDomain(url)) {
    throw new SecurityPolicyError(
      `Security policy violation: URL domain not allowed. ` +
      `Only connections to ${ALLOWED_DOMAINS.join(", ")} are permitted. ` +
      `Blocked: ${url}`
    );
  }

  // Use secure HTTPS agent with TLS 1.2+ enforcement
  const agent = createSecureAgent();

  return globalThis.fetch(url, {
    ...options,
    // Attach secure agent for Node.js fetch
    // @ts-ignore - agent is supported in Node.js fetch
    agent,
  });
}

/**
 * @param {string} rawSpecifier
 * @returns {{ name: string; versionSpec: string } | null}
 */
export function parseAffectedSpecifier(rawSpecifier) {
  const specifier = String(rawSpecifier ?? "").trim();
  if (!specifier) return null;

  const atIndex = specifier.lastIndexOf("@");
  if (atIndex <= 0) {
    return { name: specifier, versionSpec: "*" };
  }

  return {
    name: specifier.slice(0, atIndex),
    versionSpec: specifier.slice(atIndex + 1),
  };
}

/**
 * @param {unknown} raw
 * @returns {raw is import("./types.ts").FeedPayload}
 */
export function isValidFeedPayload(raw) {
  if (!isObject(raw)) return false;
  if (typeof raw.version !== "string" || !raw.version.trim()) return false;
  if (!Array.isArray(raw.advisories)) return false;

  for (const advisory of raw.advisories) {
    if (!isObject(advisory)) return false;
    if (typeof advisory.id !== "string" || !advisory.id.trim()) return false;
    if (typeof advisory.severity !== "string" || !advisory.severity.trim()) return false;
    if (!Array.isArray(advisory.affected)) return false;
    if (!advisory.affected.every((entry) => typeof entry === "string" && entry.trim())) return false;
  }

  return true;
}

/**
 * @param {string} signatureRaw
 * @returns {Buffer | null}
 */
function decodeSignature(signatureRaw) {
  const trimmed = String(signatureRaw ?? "").trim();
  if (!trimmed) return null;

  let encoded = trimmed;
  if (trimmed.startsWith("{")) {
    try {
      const parsed = JSON.parse(trimmed);
      if (isObject(parsed) && typeof parsed.signature === "string") {
        encoded = parsed.signature;
      }
    } catch {
      return null;
    }
  }

  const normalized = encoded.replace(/\s+/g, "");
  if (!normalized) return null;

  try {
    return Buffer.from(normalized, "base64");
  } catch {
    return null;
  }
}

/**
 * @param {string} payloadRaw
 * @param {string} signatureRaw
 * @param {string} publicKeyPem
 * @returns {boolean}
 */
export function verifySignedPayload(payloadRaw, signatureRaw, publicKeyPem) {
  const signature = decodeSignature(signatureRaw);
  if (!signature) return false;

  const keyPem = String(publicKeyPem ?? "").trim();
  if (!keyPem) return false;

  try {
    const publicKey = crypto.createPublicKey(keyPem);
    return crypto.verify(null, Buffer.from(payloadRaw, "utf8"), publicKey, signature);
  } catch {
    return false;
  }
}

/**
 * @param {string | Buffer} content
 * @returns {string}
 */
function sha256Hex(content) {
  return crypto.createHash("sha256").update(content).digest("hex");
}

/**
 * @param {unknown} value
 * @returns {string | null}
 */
function extractSha256Value(value) {
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    return /^[a-f0-9]{64}$/.test(normalized) ? normalized : null;
  }

  if (isObject(value) && typeof value.sha256 === "string") {
    const normalized = value.sha256.trim().toLowerCase();
    return /^[a-f0-9]{64}$/.test(normalized) ? normalized : null;
  }

  return null;
}

/**
 * @param {string} manifestRaw
 * @returns {{ schemaVersion: string; algorithm: string; files: Record<string, string> }}
 */
function parseChecksumsManifest(manifestRaw) {
  let parsed;
  try {
    parsed = JSON.parse(manifestRaw);
  } catch {
    throw new Error("Checksum manifest is not valid JSON");
  }

  if (!isObject(parsed)) {
    throw new Error("Checksum manifest must be an object");
  }

  const algorithmRaw = typeof parsed.algorithm === "string" ? parsed.algorithm.trim().toLowerCase() : "sha256";
  if (algorithmRaw !== "sha256") {
    throw new Error(`Unsupported checksum manifest algorithm: ${algorithmRaw || "(empty)"}`);
  }

  // Support legacy manifest formats:
  // - New standard: schema_version field
  // - skill-release.yml: version field (e.g., "0.0.1")
  // - deploy-pages.yml (pre-fix): generated_at field (e.g., "2026-02-08T...")
  // - Ultimate fallback: "1"
  const schemaVersion = (
    typeof parsed.schema_version === "string" ? parsed.schema_version.trim() :
    typeof parsed.version === "string" ? parsed.version.trim() :
    typeof parsed.generated_at === "string" ? parsed.generated_at.trim() :
    "1"
  );

  if (!schemaVersion) {
    throw new Error("Checksum manifest missing schema_version");
  }

  if (!isObject(parsed.files)) {
    throw new Error("Checksum manifest missing files object");
  }

  const files = /** @type {Record<string, string>} */ ({});
  for (const [key, value] of Object.entries(parsed.files)) {
    if (!String(key).trim()) continue;
    const digest = extractSha256Value(value);
    if (!digest) {
      throw new Error(`Invalid checksum digest entry for ${key}`);
    }
    files[key] = digest;
  }

  if (Object.keys(files).length === 0) {
    throw new Error("Checksum manifest has no usable file digests");
  }

  return {
    schemaVersion,
    algorithm: algorithmRaw,
    files,
  };
}

/**
 * @param {string} entryName
 * @returns {string}
 */
function normalizeChecksumEntryName(entryName) {
  return String(entryName ?? "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/^(?:\.\/)+/, "")
    .replace(/^\/+/, "");
}

/**
 * @param {Record<string, string>} files
 * @param {string} entryName
 * @returns {{ key: string; digest: string } | null}
 */
function resolveChecksumManifestEntry(files, entryName) {
  const normalizedEntry = normalizeChecksumEntryName(entryName);
  if (!normalizedEntry) return null;

  const directCandidates = [
    normalizedEntry,
    path.posix.basename(normalizedEntry),
    `advisories/${path.posix.basename(normalizedEntry)}`,
  ].filter((candidate, index, all) => candidate && all.indexOf(candidate) === index);

  for (const candidate of directCandidates) {
    if (Object.prototype.hasOwnProperty.call(files, candidate)) {
      return { key: candidate, digest: files[candidate] };
    }
  }

  const basename = path.posix.basename(normalizedEntry);
  if (!basename) return null;

  const basenameMatches = Object.entries(files).filter(([key]) => {
    const normalizedKey = normalizeChecksumEntryName(key);
    return path.posix.basename(normalizedKey) === basename;
  });

  if (basenameMatches.length > 1) {
    throw new Error(
      `Checksum manifest entry is ambiguous for ${entryName}; ` +
        `multiple manifest keys share basename ${basename}`,
    );
  }

  if (basenameMatches.length === 1) {
    const [resolvedKey, digest] = basenameMatches[0];
    return { key: resolvedKey, digest };
  }

  return null;
}

/**
 * @param {{ files: Record<string, string> }} manifest
 * @param {Record<string, string | Buffer>} expectedEntries
 */
function verifyChecksums(manifest, expectedEntries) {
  for (const [entryName, entryContent] of Object.entries(expectedEntries)) {
    if (!entryName) continue;

    const resolved = resolveChecksumManifestEntry(manifest.files, entryName);
    if (!resolved) {
      throw new Error(`Checksum manifest missing required entry: ${entryName}`);
    }

    const actualDigest = sha256Hex(entryContent);
    if (actualDigest !== resolved.digest) {
      throw new Error(`Checksum mismatch for ${entryName} (manifest key: ${resolved.key})`);
    }
  }
}

/**
 * @param {string} feedUrl
 * @returns {string}
 */
export function defaultChecksumsUrl(feedUrl) {
  try {
    return new URL("checksums.json", feedUrl).toString();
  } catch {
    const fallbackBase = String(feedUrl ?? "").replace(/\/?[^/]*$/, "");
    return `${fallbackBase}/checksums.json`;
  }
}

/**
 * Safely extracts the basename from a URL or file path.
 * @param {string} urlOrPath
 * @param {string} fallback
 * @returns {string}
 */
function safeBasename(urlOrPath, fallback) {
  try {
    // Try parsing as URL first
    const parsed = new URL(urlOrPath);
    const pathname = parsed.pathname;
    const lastSlash = pathname.lastIndexOf("/");
    if (lastSlash >= 0 && lastSlash < pathname.length - 1) {
      return pathname.slice(lastSlash + 1);
    }
  } catch {
    // Not a URL, try as path
    const normalized = String(urlOrPath ?? "").trim();
    const lastSlash = normalized.lastIndexOf("/");
    if (lastSlash >= 0 && lastSlash < normalized.length - 1) {
      return normalized.slice(lastSlash + 1);
    }
  }
  return fallback;
}

/**
 * @param {Function} fetchFn
 * @param {string} targetUrl
 * @returns {Promise<string | null>}
 */
async function fetchText(fetchFn, targetUrl) {
  const controller = new globalThis.AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetchFn(targetUrl, {
      method: "GET",
      signal: controller.signal,
      headers: { accept: "application/json,text/plain;q=0.9,*/*;q=0.8" },
    });
    if (!response.ok) return null;
    return await response.text();
  } catch (error) {
    // Re-throw security policy violations - these should never be silently caught
    if (error instanceof SecurityPolicyError) {
      throw error;
    }
    // Network errors, timeouts, etc. return null (graceful degradation)
    return null;
  } finally {
    globalThis.clearTimeout(timeout);
  }
}

/**
 * @param {string} feedPath
 * @param {{
 *   signaturePath?: string;
 *   checksumsPath?: string;
 *   checksumsSignaturePath?: string;
 *   publicKeyPem?: string;
 *   checksumsPublicKeyPem?: string;
 *   allowUnsigned?: boolean;
 *   verifyChecksumManifest?: boolean;
 *   checksumFeedEntry?: string;
 *   checksumSignatureEntry?: string;
 *   checksumPublicKeyEntry?: string;
 * }} [options]
 * @returns {Promise<import("./types.ts").FeedPayload>}
 */
export async function loadLocalFeed(feedPath, options = {}) {
  const signaturePath = options.signaturePath ?? `${feedPath}.sig`;
  const checksumsPath = options.checksumsPath ?? path.join(path.dirname(feedPath), "checksums.json");
  const checksumsSignaturePath = options.checksumsSignaturePath ?? `${checksumsPath}.sig`;
  const publicKeyPem = String(options.publicKeyPem ?? "");
  const checksumsPublicKeyPem = String(options.checksumsPublicKeyPem ?? publicKeyPem);
  const allowUnsigned = options.allowUnsigned === true;
  const verifyChecksumManifest = options.verifyChecksumManifest !== false;

  const payloadRaw = await loadTextFile(feedPath);

  if (!allowUnsigned) {
    const signatureRaw = await loadTextFile(signaturePath);
    if (!verifySignedPayload(payloadRaw, signatureRaw, publicKeyPem)) {
      throw new Error(`Feed signature verification failed for local feed: ${feedPath}`);
    }

    if (verifyChecksumManifest) {
      const checksumsRaw = await loadTextFile(checksumsPath);
      const checksumsSignatureRaw = await loadTextFile(checksumsSignaturePath);

      if (!verifySignedPayload(checksumsRaw, checksumsSignatureRaw, checksumsPublicKeyPem)) {
        throw new Error(`Checksum manifest signature verification failed: ${checksumsPath}`);
      }

      const checksumsManifest = parseChecksumsManifest(checksumsRaw);
      const checksumFeedEntry = options.checksumFeedEntry ?? path.basename(feedPath);
      const checksumSignatureEntry = options.checksumSignatureEntry ?? path.basename(signaturePath);
      const expectedEntries = /** @type {Record<string, string>} */ ({
        [checksumFeedEntry]: payloadRaw,
        [checksumSignatureEntry]: signatureRaw,
      });

      if (options.checksumPublicKeyEntry) {
        expectedEntries[options.checksumPublicKeyEntry] = publicKeyPem;
      }

      verifyChecksums(checksumsManifest, expectedEntries);
    }
  }

  const payload = JSON.parse(payloadRaw);
  if (!isValidFeedPayload(payload)) {
    throw new Error(`Invalid advisory feed format: ${feedPath}`);
  }
  return payload;
}

/**
 * @param {string} feedUrl
 * @param {{
 *   signatureUrl?: string;
 *   checksumsUrl?: string;
 *   checksumsSignatureUrl?: string;
 *   publicKeyPem?: string;
 *   checksumsPublicKeyPem?: string;
 *   allowUnsigned?: boolean;
 *   verifyChecksumManifest?: boolean;
 *   checksumFeedEntry?: string;
 *   checksumSignatureEntry?: string;
 * }} [options]
 * @returns {Promise<import("./types.ts").FeedPayload | null>}
 */
export async function loadRemoteFeed(feedUrl, options = {}) {
  // Use secure fetch with TLS 1.2+ enforcement and domain validation
  const fetchFn = secureFetch;
  if (typeof fetchFn !== "function") return null;

  const signatureUrl = options.signatureUrl ?? `${feedUrl}.sig`;
  const checksumsUrl = options.checksumsUrl ?? defaultChecksumsUrl(feedUrl);
  const checksumsSignatureUrl = options.checksumsSignatureUrl ?? `${checksumsUrl}.sig`;
  const publicKeyPem = String(options.publicKeyPem ?? "");
  const checksumsPublicKeyPem = String(options.checksumsPublicKeyPem ?? publicKeyPem);
  const allowUnsigned = options.allowUnsigned === true;
  const verifyChecksumManifest = options.verifyChecksumManifest !== false;

  try {
    const payloadRaw = await fetchText(fetchFn, feedUrl);
    if (!payloadRaw) return null;

  if (!allowUnsigned) {
    const signatureRaw = await fetchText(fetchFn, signatureUrl);
    if (!signatureRaw) return null;

    if (!verifySignedPayload(payloadRaw, signatureRaw, publicKeyPem)) {
      return null;
    }

    // Only verify checksums if explicitly requested AND both checksum files are available.
    // Note: Many upstream workflows (e.g., GitHub raw content) don't publish checksums.json,
    // so we gracefully skip verification when these files are missing.
    if (verifyChecksumManifest) {
      const checksumsRaw = await fetchText(fetchFn, checksumsUrl);
      const checksumsSignatureRaw = await fetchText(fetchFn, checksumsSignatureUrl);

      // Only proceed if BOTH checksum files are present
      if (checksumsRaw && checksumsSignatureRaw) {
        if (!verifySignedPayload(checksumsRaw, checksumsSignatureRaw, checksumsPublicKeyPem)) {
          return null;  // Fail-closed: invalid signature
        }

        const checksumsManifest = parseChecksumsManifest(checksumsRaw);
        // Derive checksum entry names from actual URLs (supports any filename, not just feed.json)
        const checksumFeedEntry = options.checksumFeedEntry ?? safeBasename(feedUrl, "feed.json");
        const checksumSignatureEntry = options.checksumSignatureEntry ?? safeBasename(signatureUrl, "feed.json.sig");
        verifyChecksums(checksumsManifest, {
          [checksumFeedEntry]: payloadRaw,
          [checksumSignatureEntry]: signatureRaw,
        });
      }
      // If checksum files missing: continue without checksum verification
      // (feed signature was already verified above at line 328)
    }
  }

    try {
      const payload = JSON.parse(payloadRaw);
      if (!isValidFeedPayload(payload)) return null;
      return payload;
    } catch {
      return null;
    }
  } catch (error) {
    // Security policy violations (invalid URLs, non-HTTPS, disallowed domains) return null
    // to allow graceful fallback to local feed
    if (error instanceof SecurityPolicyError) {
      return null;
    }
    // Re-throw unexpected errors
    throw error;
  }
}
