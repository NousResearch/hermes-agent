// Locate and read README.md from an installed @atomicmail/* npm package.
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const MAX_DEPTH = 16;
const ATOMICMAIL_NPM_NAMES = new Set([
    "@atomicmail/mcp",
    "@atomicmail/mcp-github",
    "@atomicmail/mcp-gh-pages",
    "@atomicmail/mcp-modelcontextprotocol",
    "@atomicmail/mcp-clawhub",
    "@atomic-mail/mcp",
    "@atomic-mail/mcp-github",
    "@atomic-mail/mcp-gh-pages",
    "@atomic-mail/mcp-modelcontextprotocol",
    "@atomic-mail/mcp-clawhub",
    "@atomicmail/agent-skill",
    "@atomicmail/agent-skill-github",
    "@atomicmail/agent-skill-gh-pages",
    "@atomic-mail/agent-skill",
    "@atomic-mail/agent-skill-github",
    "@atomic-mail/agent-skill-gh-pages",
]);
function isEnoent(err) {
    if (!(err instanceof Error))
        return false;
    const code = err.code;
    return code === "ENOENT" || code === "ENOTDIR";
}
/**
 * Reads README.md from the npm package root (next to package.json).
 * Intended for published @atomicmail/mcp and @atomicmail/agent-skill layouts.
 */
export async function readNpmPackageReadme() {
    const moduleDir = dirname(fileURLToPath(globalThis[Symbol.for("import-meta-ponyfill-esmodule")](import.meta).url));
    let currentDir = moduleDir;
    for (let i = 0; i < MAX_DEPTH; i++) {
        const pkgPath = resolve(currentDir, "package.json");
        const readmePath = resolve(currentDir, "README.md");
        let pkgRaw;
        try {
            pkgRaw = await readFile(pkgPath, "utf-8");
        }
        catch (err) {
            if (!isEnoent(err))
                throw err;
            const parent = resolve(currentDir, "..");
            if (parent === currentDir)
                break;
            currentDir = parent;
            continue;
        }
        let name;
        try {
            name = JSON.parse(pkgRaw).name;
        }
        catch {
            name = undefined;
        }
        if (!name || !ATOMICMAIL_NPM_NAMES.has(name)) {
            const parent = resolve(currentDir, "..");
            if (parent === currentDir)
                break;
            currentDir = parent;
            continue;
        }
        try {
            return await readFile(readmePath, "utf-8");
        }
        catch (err) {
            if (!isEnoent(err))
                throw err;
        }
        const parent = resolve(currentDir, "..");
        if (parent === currentDir)
            break;
        currentDir = parent;
    }
    throw new Error("Could not find Atomic Mail package README.md — use a published npm install " +
        "(@atomicmail/mcp* channel package) for --topic readme.");
}
