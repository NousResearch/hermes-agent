import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve as resolvePath } from "node:path";
import { fileURLToPath } from "node:url";
const moduleDir = dirname(fileURLToPath(globalThis[Symbol.for("import-meta-ponyfill-esmodule")](import.meta).url));
function resolveSharedRoot() {
    let current = moduleDir;
    for (let depth = 0; depth < 12; depth++) {
        const candidate = resolvePath(current, "shared");
        if (existsSync(candidate))
            return candidate;
        const parent = resolvePath(current, "..");
        if (parent === current)
            break;
        current = parent;
    }
    throw new Error(`Shared asset directory was not found from module path: ${moduleDir}`);
}
let cachedSharedRoot;
export function getSharedRootPath() {
    if (!cachedSharedRoot)
        cachedSharedRoot = resolveSharedRoot();
    return cachedSharedRoot;
}
export function readSharedText(relativePath) {
    const fullPath = resolvePath(getSharedRootPath(), relativePath);
    return readFileSync(fullPath, "utf-8");
}
export function readSharedJson(relativePath) {
    return JSON.parse(readSharedText(relativePath));
}
export function tryReadSharedText(relativePath) {
    try {
        return readSharedText(relativePath);
    }
    catch {
        return undefined;
    }
}
export function tryReadSharedJson(relativePath) {
    try {
        return readSharedJson(relativePath);
    }
    catch {
        return undefined;
    }
}
