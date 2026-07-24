// Resolve MCP / process credential dir + URLs from env + credentials.json.
import { homedir } from "node:os";
import process from "node:process";
import { resolve } from "node:path";
import { DEFAULT_API_URL, DEFAULT_AUTH_URL, DEFAULT_POW_SCRYPT_SALT_HEX, } from "../../core/consts.js";
import { defaultFilesFromOutDir, tryReadCredentials, } from "./agent-credentials-store.js";
/**
 * Default credential directory:
 *   1. ATOMIC_MAIL_CREDENTIALS_DIR
 *   2. ~/.atomicmail/ or %USERPROFILE%/.atomicmail
 */
export function resolveCredentialDir() {
    const fromEnv = process.env.ATOMIC_MAIL_CREDENTIALS_DIR;
    if (fromEnv && fromEnv.length > 0)
        return fromEnv;
    const home = process.env.HOME || process.env.USERPROFILE;
    if (!home) {
        throw new Error("Cannot determine default credential directory: HOME and USERPROFILE " +
            "are both unset. Set ATOMIC_MAIL_CREDENTIALS_DIR explicitly.");
    }
    return `${home.replace(/[\\/]+$/, "")}/.atomicmail`;
}
/**
 * AgentSkill / CLI: resolve credential directory from `--credentials-dir` or
 * `ATOMIC_MAIL_CREDENTIALS_DIR`, with `~` expansion (MCP uses `resolveCredentialDir` instead).
 */
export function expandCredentialDirInput(dir) {
    const raw = dir ?? process.env.ATOMIC_MAIL_CREDENTIALS_DIR ?? "~/.atomicmail";
    if (raw === "~")
        return homedir();
    return resolve(raw.replace(/^~\//, `${homedir()}/`));
}
/**
 * Merge credentials.json with ATOMIC_MAIL_* env (env wins per field).
 * authUrl and apiUrl fall back to production defaults when unset.
 */
export async function resolveAgentConfigFromEnv() {
    const credentialDir = resolveCredentialDir();
    const files = defaultFilesFromOutDir(credentialDir);
    const fileCreds = await tryReadCredentials(files.credentialsFile);
    const env = process.env;
    const envAuthUrl = env.ATOMIC_MAIL_AUTH_URL;
    const envApiUrl = env.ATOMIC_MAIL_API_URL;
    const envSalt = env.ATOMIC_MAIL_SCRYPT_SALT;
    const envApiKey = env.ATOMIC_MAIL_API_KEY;
    const authUrl = envAuthUrl ?? fileCreds?.authUrl ?? DEFAULT_AUTH_URL;
    const apiUrl = envApiUrl ?? fileCreds?.apiUrl ?? DEFAULT_API_URL;
    const scryptSalt = envSalt ?? fileCreds?.scryptSalt ??
        DEFAULT_POW_SCRYPT_SALT_HEX;
    const apiKey = envApiKey ?? fileCreds?.apiKey;
    const inboxId = fileCreds?.inboxId;
    const usingFile = fileCreds !== undefined;
    const usingEnv = !!(envAuthUrl || envApiUrl || envSalt || envApiKey);
    const source = usingFile && usingEnv
        ? "mixed"
        : usingFile
            ? "credentials-file"
            : usingEnv
                ? "env"
                : "defaults";
    return {
        authUrl: authUrl.replace(/\/+$/, ""),
        apiUrl: apiUrl.replace(/\/+$/, ""),
        scryptSalt: scryptSalt,
        apiKey,
        inboxId,
        credentialDir,
        files,
        source,
    };
}
