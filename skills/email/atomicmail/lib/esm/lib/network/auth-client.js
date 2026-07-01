// Thin HTTP client for auth-service (PoW challenge → session → capability).
//
// Encapsulates the full PoW challenge → session → capability flow so callers
// (integration tests, the future agent skill, etc.) don't have to reimplement scrypt grinding.
//
// The PoW digest is scrypt-based and uses the SAME salt the auth-service
// uses on the verify path (see services/auth-service/src/crypto.ts). The
// client must therefore be configured with that salt — there is no public
// hash function here, the salt is part of the protocol.
import { scrypt } from "node:crypto";
import { DEFAULT_POW_SCRYPT_SALT_HEX } from "../core/consts.js";
// Mirror services/auth-service/src/crypto.ts exactly. Changing any of these
// constants on either side breaks PoW interop.
const SCRYPT_PARAMS = { N: 16384, r: 8, p: 1 };
const POW_HASH_BYTES = 64;
// Node's undici fetch() has no default read/idle timeout; bound every request.
const AUTH_TIMEOUT_MS = 15000;
/** Thrown for any non-2xx HTTP response or malformed payload. */
export class AuthClientError extends Error {
    status;
    bodyText;
    constructor(status, bodyText, message) {
        super(message);
        this.name = "AuthClientError";
        this.status = status;
        this.bodyText = bodyText;
    }
}
export class AuthClient {
    baseUrl;
    scryptSaltHex;
    constructor(options) {
        this.baseUrl = options.baseUrl.replace(/\/+$/, "");
        this.scryptSaltHex = options.scryptSaltHex ?? DEFAULT_POW_SCRYPT_SALT_HEX;
    }
    /**
     * Register a new inbox under `username`. Returns the freshly minted API key
     * (the server only ever returns it once — the caller MUST persist it) and
     * a session JWT.
     */
    async signup(username) {
        const { challengeJWT, challenge, difficulty } = await this.fetchChallenge();
        const { powHex, nonce } = await this.solvePoW(challenge, difficulty);
        const { sessionJWT, data } = await this.postSession(challengeJWT, {
            powHex,
            nonce: nonce.toString(),
            username,
        });
        if (typeof data.apiKey !== "string") {
            throw new AuthClientError(200, JSON.stringify(data), "Signup response missing apiKey.");
        }
        return { apiKey: data.apiKey, sessionJWT };
    }
    /** Exchange an existing API key for a fresh session JWT. */
    async login(apiKey) {
        const { challengeJWT, challenge, difficulty } = await this.fetchChallenge();
        const { powHex, nonce } = await this.solvePoW(challenge, difficulty);
        const { sessionJWT } = await this.postSession(challengeJWT, {
            powHex,
            nonce: nonce.toString(),
            apiKey,
        });
        return { sessionJWT };
    }
    /**
     * Exchange a session JWT for a short-lived capability JWT (audience:
     * api-service).
     */
    async renew(sessionJWT) {
        const res = await fetch(`${this.baseUrl}/api/v1/capability`, {
            method: "POST",
            headers: { Authorization: `Bearer ${sessionJWT}` },
            signal: AbortSignal.timeout(AUTH_TIMEOUT_MS),
        });
        const text = await res.text();
        if (!res.ok) {
            throw new AuthClientError(res.status, text, `auth-service capability returned ${res.status}: ${text}`);
        }
        const capabilityJWT = readBearerToken(res.headers.get("Authorization"), "Capability response missing Authorization bearer token.");
        return { capabilityJWT };
    }
    async fetchChallenge() {
        const res = await fetch(`${this.baseUrl}/api/v1/challenge`, {
            method: "POST",
            signal: AbortSignal.timeout(AUTH_TIMEOUT_MS),
        });
        const text = await res.text();
        if (!res.ok) {
            throw new AuthClientError(res.status, text, `auth-service challenge returned ${res.status}: ${text}`);
        }
        const challengeJWT = readBearerToken(res.headers.get("Authorization"), "Challenge response missing Authorization bearer token.");
        const payload = decodeJwtPayload(challengeJWT);
        if (typeof payload.jti !== "string" ||
            typeof payload.difficulty !== "number") {
            // Do not embed the raw challenge JWT (a bearer token) in the error.
            throw new AuthClientError(res.status, `${challengeJWT.slice(0, 12)}…(challenge JWT redacted)`, "Challenge JWT payload is malformed (missing jti or difficulty).");
        }
        return {
            challengeJWT,
            challenge: payload.jti,
            difficulty: payload.difficulty,
        };
    }
    async postSession(challengeJWT, body) {
        const res = await fetch(`${this.baseUrl}/api/v1/session`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${challengeJWT}`,
            },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(AUTH_TIMEOUT_MS),
        });
        const text = await res.text();
        if (!res.ok) {
            throw new AuthClientError(res.status, text, `auth-service session returned ${res.status}: ${text}`);
        }
        const sessionJWT = readBearerToken(res.headers.get("Authorization"), "Session response missing Authorization bearer token.");
        let data = {};
        if (text.trim().length > 0) {
            try {
                data = JSON.parse(text);
            }
            catch {
                throw new AuthClientError(res.status, text, "auth-service session returned non-JSON body.");
            }
        }
        return { sessionJWT, data };
    }
    async parseJsonOrThrow(res, endpoint) {
        const text = await res.text();
        if (!res.ok) {
            throw new AuthClientError(res.status, text, `auth-service ${endpoint} returned ${res.status}: ${text}`);
        }
        try {
            return JSON.parse(text);
        }
        catch {
            throw new AuthClientError(res.status, text, `auth-service ${endpoint} returned non-JSON body.`);
        }
    }
    /**
     * Brute-force a PoW nonce. Mirrors `generatePow` in
     * services/auth-service/src/crypto.ts: scrypt(`${challenge}:${nonce}`, salt,
     * 64) until `difficulty` leading bits of the digest are zero.
     *
     * Expected work at the server's POW_DIFFICULTY=6 is ~2^6 = 64 attempts; well
     * within the challenge JWT's 3-minute TTL.
     */
    async solvePoW(challenge, difficulty) {
        assertSolvableDifficulty(difficulty);
        let nonce = 0n;
        while (nonce < MAX_POW_ITERATIONS) {
            const digest = await scryptHash(`${challenge}:${nonce}`, this.scryptSaltHex);
            if (hasLeadingZeroBits(digest, difficulty)) {
                return { powHex: bytesToHex(digest), nonce };
            }
            nonce++;
        }
        throw new AuthClientError(0, "", `PoW exceeded ${MAX_POW_ITERATIONS} iterations without a solution ` +
            `at difficulty ${difficulty}; aborting to avoid an unbounded hang.`);
    }
}
// A difficulty above the digest bit-length (POW_HASH_BYTES * 8 = 512) can never
// be satisfied and would spin forever; cap iterations as a hard safety net for a
// misconfigured or tampered challenge (nominal server difficulty is ~6).
const MAX_POW_DIFFICULTY = POW_HASH_BYTES * 8;
const MAX_POW_ITERATIONS = 100_000_000n;
function assertSolvableDifficulty(difficulty) {
    if (typeof difficulty !== "number" ||
        !Number.isInteger(difficulty) ||
        difficulty < 0 ||
        difficulty > MAX_POW_DIFFICULTY) {
        throw new AuthClientError(0, "", `PoW difficulty ${difficulty} is out of range (expected an integer ` +
            `0-${MAX_POW_DIFFICULTY}); refusing to grind — the challenge is likely ` +
            "misconfigured or tampered.");
    }
}
function scryptHash(data, salt) {
    const bytes = new TextEncoder().encode(data);
    return new Promise((resolve, reject) => {
        scrypt(bytes, salt, POW_HASH_BYTES, SCRYPT_PARAMS, (err, derived) => {
            if (err)
                return reject(err);
            resolve(new Uint8Array(derived));
        });
    });
}
function hasLeadingZeroBits(hash, bits) {
    if (bits > hash.length * 8)
        return false;
    const fullBytes = Math.floor(bits / 8);
    const remainingBits = bits % 8;
    for (let i = 0; i < fullBytes; i++) {
        if (hash[i] !== 0)
            return false;
    }
    if (remainingBits > 0) {
        const mask = (0xff << (8 - remainingBits)) & 0xff;
        if ((hash[fullBytes] & mask) !== 0)
            return false;
    }
    return true;
}
function bytesToHex(bytes) {
    let hex = "";
    for (let i = 0; i < bytes.length; i++) {
        hex += bytes[i].toString(16).padStart(2, "0");
    }
    return hex;
}
function decodeJwtPayload(jwt) {
    const parts = jwt.split(".");
    if (parts.length < 2) {
        throw new Error("Malformed JWT: expected at least 2 dot-separated segments.");
    }
    const payloadB64Url = parts[1];
    const padLen = (4 - (payloadB64Url.length % 4)) % 4;
    const base64 = payloadB64Url
        .replace(/-/g, "+")
        .replace(/_/g, "/")
        .padEnd(payloadB64Url.length + padLen, "=");
    return JSON.parse(atob(base64));
}
function readBearerToken(headerValue, missingError) {
    if (!headerValue) {
        throw new Error(missingError);
    }
    const match = /^\s*Bearer\s+(.+?)\s*$/i.exec(headerValue);
    if (!match || !match[1]) {
        throw new Error("Authorization header must use Bearer scheme.");
    }
    return match[1];
}
