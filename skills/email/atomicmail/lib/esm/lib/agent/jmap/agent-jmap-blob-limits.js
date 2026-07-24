// RFC 9404 §3.1 client-side checks before Blob/upload POST (maxSizeBlobSet, maxDataSources).
const utf8Encoder = new TextEncoder();
export function utf8ByteLength(s) {
    return utf8Encoder.encode(s).length;
}
/** Decoded byte length; throws if the string is not valid standard base64. */
export function decodedBase64ByteLength(b64) {
    const t = b64.replace(/\s+/g, "");
    if (t.length === 0)
        return 0;
    if (t.length % 4 === 1) {
        throw new Error("Invalid base64 in Blob/upload data:asBase64 (RFC 9404 §4.1): bad length.");
    }
    try {
        const bin = atob(t.replace(/-/g, "+").replace(/_/g, "/"));
        return bin.length;
    }
    catch {
        throw new Error("Invalid base64 in Blob/upload data:asBase64 (RFC 9404 §4.1).");
    }
}
function sliceOctetCount(sourceLen, offset, length) {
    let off = 0;
    if (typeof offset === "number" && Number.isInteger(offset) && offset >= 0) {
        off = offset;
    }
    const rest = Math.max(0, sourceLen - off);
    if (length === null || length === undefined) {
        return rest;
    }
    if (typeof length === "number" && Number.isInteger(length) && length >= 0) {
        return Math.min(rest, length);
    }
    return rest;
}
/**
 * Returns total octets for one UploadObject `data` array, or `null` if a literal
 * (non-`#`) blobId slice is present or a `#ref` is not yet in `knownSizes`.
 */
export function tryComputeUploadDataOctets(dataUnknown, knownSizes) {
    if (!Array.isArray(dataUnknown))
        return 0;
    let total = 0;
    for (const part of dataUnknown) {
        if (!part || typeof part !== "object")
            continue;
        const o = part;
        const tText = o["data:asText"];
        const tB64 = o["data:asBase64"];
        const tBlob = o["blobId"];
        const hasText = tText !== undefined && tText !== null;
        const hasB64 = tB64 !== undefined && tB64 !== null;
        const hasBlob = tBlob !== undefined && tBlob !== null;
        const n = [hasText, hasB64, hasBlob].filter(Boolean).length;
        if (n === 0)
            continue;
        if (n !== 1) {
            throw new Error("Each Blob/upload DataSourceObject must use exactly one of " +
                "data:asText, data:asBase64, or blobId (RFC 9404 §4.1).");
        }
        if (hasText) {
            if (typeof tText !== "string") {
                throw new Error("Blob/upload data:asText must be a string.");
            }
            total += utf8ByteLength(tText);
            continue;
        }
        if (hasB64) {
            if (typeof tB64 !== "string") {
                throw new Error("Blob/upload data:asBase64 must be a string.");
            }
            total += decodedBase64ByteLength(tB64);
            continue;
        }
        if (typeof tBlob !== "string" || tBlob.length === 0) {
            throw new Error("Blob/upload blobId must be a non-empty string.");
        }
        if (!tBlob.startsWith("#")) {
            return null;
        }
        const refId = tBlob.slice(1);
        const sourceLen = knownSizes.get(refId);
        if (sourceLen === undefined) {
            return null;
        }
        total += sliceOctetCount(sourceLen, o["offset"], o["length"]);
    }
    return total;
}
function resolveCreateSizesForOneBlobUpload(create, limits, priorSizes) {
    const merged = new Map(priorSizes);
    const pending = new Set(Object.keys(create));
    while (pending.size > 0) {
        let progressed = false;
        for (const id of [...pending]) {
            const upload = create[id];
            if (!upload || typeof upload !== "object") {
                pending.delete(id);
                progressed = true;
                continue;
            }
            const data = upload["data"];
            const dsCount = Array.isArray(data) ? data.length : 0;
            if (limits.maxDataSources !== undefined &&
                dsCount > limits.maxDataSources) {
                throw new Error(`Blob/upload create "${id}" uses ${dsCount} DataSourceObject entries; ` +
                    `account maxDataSources is ${limits.maxDataSources} (RFC 9404 §3.1).`);
            }
            const computed = tryComputeUploadDataOctets(data, merged);
            if (computed === null) {
                continue;
            }
            if (limits.maxSizeBlobSet !== null &&
                computed > limits.maxSizeBlobSet) {
                throw new Error(`Blob/upload create "${id}" would be ${computed} octets; account ` +
                    `maxSizeBlobSet is ${limits.maxSizeBlobSet} (RFC 9404 §3.1). ` +
                    "Use a smaller payload, split data, or POST the file to the session " +
                    "uploadUrl (RFC 8620) / MCP attachments for large binaries.");
            }
            merged.set(id, computed);
            pending.delete(id);
            progressed = true;
        }
        if (!progressed) {
            break;
        }
    }
    return merged;
}
/**
 * Walks `methodCalls` in order and enforces limits for each `Blob/upload` whose
 * `accountId` has an entry in `limitsByAccount`.
 */
export function assertBlobUploadEnvelopeWithinLimits(envelope, limitsByAccount) {
    let globalSizes = new Map();
    for (const call of envelope.methodCalls) {
        if (!Array.isArray(call) || call[0] !== "Blob/upload")
            continue;
        const arg = call[1];
        if (!arg || typeof arg !== "object")
            continue;
        const rec = arg;
        const accountId = rec["accountId"];
        if (typeof accountId !== "string" || accountId.length === 0)
            continue;
        const limits = limitsByAccount.get(accountId);
        if (!limits)
            continue;
        const create = rec["create"];
        if (!create || typeof create !== "object")
            continue;
        globalSizes = resolveCreateSizesForOneBlobUpload(create, limits, globalSizes);
    }
}
export function assertAttachmentBytesWithinBlobLimit(items, limits) {
    if (!limits || limits.maxSizeBlobSet === null)
        return;
    const max = limits.maxSizeBlobSet;
    for (const it of items) {
        if (it.byteLength > max) {
            throw new Error(`${it.label} is ${it.byteLength} octets but account maxSizeBlobSet is ` +
                `${max} (RFC 9404 §3.1). Use a smaller file or refresh the session if ` +
                "limits changed.");
        }
    }
}
