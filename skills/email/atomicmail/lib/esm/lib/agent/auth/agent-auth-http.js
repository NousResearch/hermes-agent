// auth-service HTTP: challenge → session → capability.
import { decodeJwtPayload } from "./agent-jwt.js";
import { solvePow } from "./agent-pow.js";
export async function fetchChallenge(authUrl) {
    const res = await fetch(`${authUrl}/api/v1/challenge`, {
        method: "POST",
    });
    const text = await res.text();
    if (!res.ok) {
        throw new Error(`auth-service /api/v1/challenge returned ${res.status}: ${text}`);
    }
    const challengeJWT = readBearerToken(res.headers.get("Authorization"), "Challenge response missing Authorization bearer token.");
    const payload = decodeJwtPayload(challengeJWT);
    if (typeof payload.jti !== "string" ||
        typeof payload.difficulty !== "number") {
        throw new Error("Challenge JWT payload malformed (missing jti or difficulty).");
    }
    return {
        challengeJWT,
        challenge: payload.jti,
        difficulty: payload.difficulty,
    };
}
export async function exchangeSession(authUrl, body) {
    const { challengeJWT, ...payload } = body;
    const res = await fetch(`${authUrl}/api/v1/session`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${challengeJWT}`,
        },
        body: JSON.stringify(payload),
    });
    const text = await res.text();
    if (!res.ok) {
        throw new Error(`auth-service /api/v1/session returned ${res.status}: ${text}`);
    }
    const sessionJWT = readBearerToken(res.headers.get("Authorization"), "Session response missing Authorization bearer token.");
    let data = {};
    if (text.trim().length > 0) {
        try {
            data = JSON.parse(text);
        }
        catch {
            throw new Error("auth-service /api/v1/session returned non-JSON body.");
        }
    }
    return {
        sessionJWT,
        apiKey: typeof data.apiKey === "string" ? data.apiKey : undefined,
    };
}
export async function fetchCapability(authUrl, sessionJWT) {
    const res = await fetch(`${authUrl}/api/v1/capability`, {
        method: "POST",
        headers: { Authorization: `Bearer ${sessionJWT}` },
    });
    const text = await res.text();
    if (!res.ok) {
        throw new Error(`auth-service /api/v1/capability returned ${res.status}: ${text}`);
    }
    return readBearerToken(res.headers.get("Authorization"), "Capability response missing Authorization bearer token.");
}
export async function performPoWAndSession(input) {
    const { authUrl, scryptSalt } = input;
    const { challengeJWT, challenge, difficulty } = await fetchChallenge(authUrl);
    const { powHex, nonce } = await solvePow(challenge, difficulty, scryptSalt, input.onPowProgress);
    return exchangeSession(authUrl, {
        challengeJWT,
        powHex,
        nonce,
        apiKey: input.apiKey,
        username: input.username,
    });
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
