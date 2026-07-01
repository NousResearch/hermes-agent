// Create AgentSession for integration hosts (Activepieces, Dify-style runtimes).
import { DEFAULT_API_URL, DEFAULT_AUTH_URL, DEFAULT_POW_SCRYPT_SALT_HEX, } from "../core/consts.js";
import { AgentSession } from "../agent/session/agent-session.js";
import { KeyValueCredentialStore, } from "./key-value-credential-store.js";
function resolveIntegrationEnv(env) {
    return {
        authUrl: (env?.authUrl ?? DEFAULT_AUTH_URL).replace(/\/+$/, ""),
        apiUrl: (env?.apiUrl ?? DEFAULT_API_URL).replace(/\/+$/, ""),
        scryptSalt: env?.scryptSalt ?? DEFAULT_POW_SCRYPT_SALT_HEX,
    };
}
export async function createAgentSession(input) {
    const resolved = resolveIntegrationEnv(input.env);
    const loaded = await input.store.load();
    const creds = loaded.credentials;
    return AgentSession.create({
        authUrl: creds?.authUrl ?? resolved.authUrl,
        apiUrl: creds?.apiUrl ?? resolved.apiUrl,
        scryptSalt: creds?.scryptSalt ?? resolved.scryptSalt,
        apiKey: input.apiKey ?? creds?.apiKey,
        inboxId: creds?.inboxId,
        credentialDir: input.credentialDir ?? "integration://default",
        store: input.store,
    });
}
export async function createAgentSessionFromKeyValue(input) {
    const accountId = input.accountId ?? "default";
    const store = new KeyValueCredentialStore(input.storage, accountId);
    return createAgentSession({
        store,
        env: input.env,
        apiKey: input.apiKey,
        credentialDir: input.credentialDir ??
            `integration://account/${accountId}`,
    });
}
