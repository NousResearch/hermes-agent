// Credential persistence backed by a host-provided key-value store (Dify, Activepieces, …).
import { parseCredentialsJson, serializeCredentials, } from "../agent/session/agent-credentials-store.js";
export class KeyValueCredentialStore {
    storage;
    accountId;
    constructor(storage, accountId = "default") {
        this.storage = storage;
        this.accountId = accountId;
    }
    key(suffix) {
        return `account:${this.accountId}:${suffix}`;
    }
    get credentialsKey() {
        return this.key("credentials.json");
    }
    get sessionKey() {
        return this.key("session.jwt");
    }
    get capabilityKey() {
        return this.key("capability.jwt");
    }
    async exists(key) {
        if (this.storage.has) {
            return this.storage.has(key);
        }
        const value = await this.storage.get(key);
        return value !== undefined;
    }
    async load() {
        let credentials;
        const rawCredentials = await this.storage.get(this.credentialsKey);
        if (rawCredentials) {
            try {
                credentials = parseCredentialsJson(rawCredentials, this.credentialsKey);
            }
            catch {
                credentials = undefined;
            }
        }
        const sessionJwt = await this.storage.get(this.sessionKey);
        const capabilityJwt = await this.storage.get(this.capabilityKey);
        return {
            credentials,
            sessionJwt,
            capabilityJwt,
        };
    }
    async save(artifacts) {
        if (artifacts.credentials !== undefined) {
            await this.storage.set(this.credentialsKey, serializeCredentials(artifacts.credentials));
        }
        if (artifacts.sessionJwt !== undefined) {
            await this.storage.set(this.sessionKey, artifacts.sessionJwt);
        }
        if (artifacts.capabilityJwt !== undefined) {
            await this.storage.set(this.capabilityKey, artifacts.capabilityJwt);
        }
    }
    async clear() {
        for (const key of [this.credentialsKey, this.sessionKey, this.capabilityKey]) {
            try {
                if (await this.exists(key)) {
                    await this.storage.delete(key);
                }
            }
            catch {
                // non-fatal
            }
        }
    }
}
