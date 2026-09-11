
import { pathToFileURL } from 'node:url';

// returns true if the passsed file is being invoked from node,
// not imported.
export function isMain(importMetaUrl) {
    return   importMetaUrl === pathToFileURL(process.argv[1]).href;
}