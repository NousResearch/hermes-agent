// n8n credential persistence via host key-value / static-data storage.
import { KeyValueCredentialStore, } from "./key-value-credential-store.js";
const KEY_PREFIX = "atomicmail";
function scopedKey(accountId, suffix) {
    return `${KEY_PREFIX}:${accountId}:${suffix}`;
}
function normalizeBackend(backend) {
    return {
        async get(key) {
            const value = await backend.get(key);
            return value === undefined || value === null ? undefined : String(value);
        },
        async set(key, value) {
            await backend.set(key, value);
        },
        async delete(key) {
            await backend.delete(key);
        },
        ...(backend.has && {
            async has(key) {
                return Boolean(await backend.has(key));
            },
        }),
    };
}
function prefixingStore(backend, accountId) {
    return {
        get: (key) => backend.get(scopedKey(accountId, key)),
        set: (key, value) => backend.set(scopedKey(accountId, key), value),
        delete: (key) => backend.delete(scopedKey(accountId, key)),
        ...(backend.has && {
            has: (key) => backend.has(scopedKey(accountId, key)),
        }),
    };
}
/**
 * Wrap n8n host storage as a CredentialStore.
 * Keys: `atomicmail:{accountId}:account:{accountId}:credentials.json`, etc.
 */
export function createN8nCredentialStore(backend, accountId = "default") {
    return new KeyValueCredentialStore(prefixingStore(normalizeBackend(backend), accountId), accountId);
}
/** Alias for integration hosts that expect a generic factory name. */
export const createKeyValueStore = createN8nCredentialStore;
/** Adapter for n8n `getWorkflowStaticData()`-style object storage. */
export function n8nStaticDataBackend(data) {
    return {
        get(key) {
            const value = data[key];
            return typeof value === "string" ? value : undefined;
        },
        set(key, value) {
            data[key] = value;
        },
        delete(key) {
            delete data[key];
        },
        has(key) {
            return Object.prototype.hasOwnProperty.call(data, key);
        },
    };
}
