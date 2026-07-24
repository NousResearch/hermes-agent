// Create AgentSession for a specific credential directory (MCP per-request / CLI parity).
import { AgentSession } from "./agent-session.js";
import { defaultFilesFromOutDir, FilesystemCredentialStore, tryReadCredentials, } from "./agent-credentials-store.js";
import { expandCredentialDirInput, } from "./agent-resolve-config.js";
export async function createAgentSessionForCredentialDir(credentialDir, envDefaults, options = {}) {
    const expandedDir = expandCredentialDirInput(credentialDir);
    const files = defaultFilesFromOutDir(expandedDir);
    const store = new FilesystemCredentialStore(files);
    const fileCreds = await tryReadCredentials(files.credentialsFile);
    if (!fileCreds) {
        if (options.requireCredentials) {
            throw new Error(`No credentials in '${expandedDir}'. Run register with ` +
                `credentials_dir pointing at that directory first.`);
        }
        return AgentSession.create({
            authUrl: envDefaults.authUrl,
            apiUrl: envDefaults.apiUrl,
            scryptSalt: envDefaults.scryptSalt,
            credentialDir: expandedDir,
            store,
        });
    }
    const creds = fileCreds;
    return AgentSession.create({
        authUrl: creds.authUrl,
        apiUrl: creds.apiUrl,
        scryptSalt: creds.scryptSalt,
        apiKey: creds.apiKey,
        inboxId: creds.inboxId,
        credentialDir: expandedDir,
        store,
    });
}
