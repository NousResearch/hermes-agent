// pen-embed asset keys → files on disk.
//
// The editor keys an asset by its absolute path resolved next to the connect message's
// fileURI, minus the leading slash: `Users/me/docs/images/photo.png` for
// `file:///Users/me/docs/untitled.pen` (pen-embed-demo README, "storage-*-asset"). The key
// carries the URI's percent-escapes, so it decodes as a file URL. Assets land beside the
// .pen; a key that resolves outside the canvas folder is refused.

import path from 'node:path'
import { fileURLToPath } from 'node:url'

export function resolvePenAssetPath(penFilePath: string, key: string): string | null {
  let resolved: string

  try {
    resolved = path.resolve(fileURLToPath(new URL(`file:///${key.replace(/^\/+/, '')}`)))
  } catch {
    return null
  }

  const dir = path.dirname(path.resolve(penFilePath))

  return resolved.startsWith(dir + path.sep) ? resolved : null
}
