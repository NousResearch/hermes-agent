
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// returns true if the passsed file is being invoked from node,
// not imported. `node -e` / stdin runs have no argv[1]: nothing is main.
/** @param {string} importMetaUrl @returns {boolean} */
export function isMain(importMetaUrl) {
    const entry = process.argv[1];
    return entry !== undefined && importMetaUrl === pathToFileURL(path.resolve(entry)).href;
}
