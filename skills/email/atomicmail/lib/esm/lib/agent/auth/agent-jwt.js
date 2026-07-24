// JWT helpers for capability/session expiry checks.
export const SESSION_SAFETY_MARGIN_MS = 60_000;
export const CAPABILITY_SAFETY_MARGIN_MS = 20_000;
export function decodeJwtPayload(jwt) {
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
export function isJwtExpired(jwt, marginMs) {
    try {
        const { exp } = decodeJwtPayload(jwt);
        if (typeof exp !== "number")
            return true;
        return Date.now() >= exp * 1000 - marginMs;
    }
    catch {
        return true;
    }
}
