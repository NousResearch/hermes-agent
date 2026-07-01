// Credential file I/O shared by MCP and AgentSkill.
// Three files: credentials.json, session.jwt, capability.jwt (mode 0600).
import { mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
export function defaultFilesFromOutDir(outDir) {
    const base = resolve(outDir);
    return {
        credentialsFile: join(base, "credentials.json"),
        sessionFile: join(base, "session.jwt"),
        capabilityFile: join(base, "capability.jwt"),
    };
}
async function ensureParent(path) {
    await mkdir(dirname(path), { recursive: true });
}
export function parseCredentialsJson(raw, pathForErrors = "credentials.json") {
    let obj;
    try {
        obj = JSON.parse(raw);
    }
    catch (err) {
        throw new Error(`Credentials file '${pathForErrors}' is not valid JSON: ${err.message}`);
    }
    const required = [
        "apiKey",
        "inboxId",
        "authUrl",
        "apiUrl",
        "scryptSalt",
        "uploadUrl",
        "downloadUrl",
    ];
    for (const k of required) {
        if (typeof obj[k] !== "string" || obj[k].length === 0) {
            throw new Error(`Credentials file '${pathForErrors}' missing required field: ${k}`);
        }
    }
    return obj;
}
export function serializeCredentials(creds) {
    return JSON.stringify(creds, null, 2) + "\n";
}
export async function writeCredentials(path, creds) {
    await ensureParent(path);
    await writeFile(path, serializeCredentials(creds), { mode: 0o600 });
}
export async function readCredentials(path) {
    let raw;
    try {
        raw = await readFile(path, "utf-8");
    }
    catch (err) {
        throw new Error(`Could not read credentials file '${path}': ${err.message}. ` +
            "Did you run register first?");
    }
    return parseCredentialsJson(raw, path);
}
export async function tryReadCredentials(path) {
    try {
        const raw = await readFile(path, "utf-8");
        return parseCredentialsJson(raw, path);
    }
    catch {
        return undefined;
    }
}
export async function writeJwtFile(path, jwt) {
    await ensureParent(path);
    await writeFile(path, jwt, { mode: 0o600 });
}
export async function tryReadJwtFile(path) {
    try {
        const raw = await readFile(path, "utf-8");
        return raw.trim();
    }
    catch {
        return undefined;
    }
}
export class FilesystemCredentialStore {
    files;
    constructor(files) {
        this.files = files;
    }
    async load() {
        return {
            credentials: await tryReadCredentials(this.files.credentialsFile),
            sessionJwt: await tryReadJwtFile(this.files.sessionFile),
            capabilityJwt: await tryReadJwtFile(this.files.capabilityFile),
        };
    }
    async save(artifacts) {
        if (artifacts.credentials !== undefined) {
            await writeCredentials(this.files.credentialsFile, artifacts.credentials);
        }
        if (artifacts.sessionJwt !== undefined) {
            await writeJwtFile(this.files.sessionFile, artifacts.sessionJwt);
        }
        if (artifacts.capabilityJwt !== undefined) {
            await writeJwtFile(this.files.capabilityFile, artifacts.capabilityJwt);
        }
    }
    async clear() {
        await unlinkCredentialArtifacts(this.files);
    }
}
/** Best-effort removal of credential artifacts (ignore missing files). */
export async function unlinkCredentialArtifacts(files) {
    for (const p of [
        files.credentialsFile,
        files.sessionFile,
        files.capabilityFile,
    ]) {
        try {
            await unlink(p);
        }
        catch {
            // ignore
        }
    }
}
