/** Parse JMAP responses for integration tests and dev CLIs. */
/** Throws if `EmailSubmission/set` did not create a submission. */
export function assertJmapSubmissionCreated(bodyText) {
    let parsed;
    try {
        parsed = JSON.parse(bodyText);
    }
    catch {
        throw new Error("JMAP response was not JSON; cannot verify submission.");
    }
    const responses = parsed.methodResponses ?? [];
    for (const r of responses) {
        if (!Array.isArray(r) || r[0] !== "EmailSubmission/set")
            continue;
        const payload = r[1];
        if (payload?.created && Object.keys(payload.created).length > 0) {
            return;
        }
    }
    throw new Error("EmailSubmission/set did not create a submission. Response:\n" + bodyText);
}
/**
 * Throws if the first `Blob/upload` response reports `size: 0` for a non-empty
 * payload (common server misconfiguration).
 */
export function assertBlobUploadSizesNonZero(bodyText, expectBytes) {
    if (expectBytes <= 0)
        return;
    let parsed;
    try {
        parsed = JSON.parse(bodyText);
    }
    catch {
        return;
    }
    const first = parsed.methodResponses?.[0];
    if (!Array.isArray(first) || first[0] !== "Blob/upload")
        return;
    const payload = first[1];
    const created = payload?.created;
    if (!created || typeof created !== "object")
        return;
    for (const v of Object.values(created)) {
        const s = v?.size;
        if (typeof s === "number" && s === 0) {
            throw new Error("Blob/upload returned size 0 — this host is not persisting blob bytes. " +
                "Fix the server, or use RFC 8620 binary upload when POST to uploadUrl works.");
        }
    }
}
