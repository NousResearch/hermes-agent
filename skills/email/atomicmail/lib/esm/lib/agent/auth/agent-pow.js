// PoW scrypt — mirrors services/auth-service/src/crypto.ts.
import { scrypt } from "node:crypto";
const SCRYPT_PARAMS = { N: 16384, r: 8, p: 1 };
const POW_HASH_BYTES = 64;
function bytesToHex(bytes) {
    let hex = "";
    for (let i = 0; i < bytes.length; i++) {
        hex += bytes[i].toString(16).padStart(2, "0");
    }
    return hex;
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
export async function solvePow(challenge, difficulty, salt, onProgress) {
    let nonce = 0n;
    while (true) {
        const digest = await scryptHash(`${challenge}:${nonce}`, salt);
        if (hasLeadingZeroBits(digest, difficulty)) {
            return { powHex: bytesToHex(digest), nonce: nonce.toString() };
        }
        nonce++;
        if (onProgress && nonce % 64n === 0n)
            onProgress(nonce);
    }
}
