// Electron 44 rearchitected the clipboard onto the W3C async API: readText /
// writeText return Promises, and image reads no longer have a readImage()
// shortcut — the bytes arrive through clipboard.read() (ClipboardItem[]) and
// ClipboardItem.getType('image/png') as a Blob. This helper keeps the old
// handler contract ("PNG bytes or null") behind the new API.

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const PNG_MIME = 'image/png'
const URI_LIST_MIME = 'text/uri-list'

// The composer's image types (its image picker's filter): a clipboard file URL
// naming anything else is never read.
const IMAGE_FILE_EXTENSIONS = new Set(['.bmp', '.gif', '.jpeg', '.jpg', '.png', '.tiff', '.webp'])

// Read the system clipboard's image as PNG bytes, or null when the clipboard
// carries no image. Depends on the injected clipboard so tests can drive it.
async function readClipboardImagePng(clipped) {
  let items

  try {
    items = await clipped.read()
  } catch {
    // A clipboard read can reject while a clipboard owner is mid-write; the
    // old sync readImage() returned an empty NativeImage instead.
    return null
  }

  for (const item of items) {
    if (!Array.isArray(item?.types) || !item.types.includes(PNG_MIME)) {
      continue
    }

    try {
      const blob = await item.getType(PNG_MIME)

      if (!blob || typeof blob.arrayBuffer !== 'function') {
        continue
      }

      const buffer = Buffer.from(await blob.arrayBuffer())

      // Guard against a degenerate empty payload: an empty image must behave
      // like "no image" so the WSL fallback still runs.
      if (buffer.length > 0) {
        return buffer
      }
    } catch {
      // getType rejects when the type vanished between read() and getType();
      // keep scanning the remaining items.
    }
  }

  return null
}

function clipboardFilePaths(uriList) {
  const paths = []

  for (const line of String(uriList).split(/\r?\n/)) {
    const uri = line.trim()

    // RFC 2483: `#` lines are comments; only local files can be read here.
    if (!uri.startsWith('file:')) {
      continue
    }

    try {
      paths.push(fileURLToPath(uri))
    } catch {
      // A malformed file URL must not abort the rest of the list.
    }
  }

  return paths
}

async function readImageFile(filePath, maxBytes) {
  try {
    const stat = await fs.promises.stat(filePath)

    if (!stat.isFile() || stat.size > maxBytes) {
      return null
    }

    const data = await fs.promises.readFile(filePath)

    return data.length > 0 && data.length <= maxBytes ? data : null
  } catch {
    return null
  }
}

// Read the first image FILE the clipboard names, or null. macOS Universal
// Clipboard (an image copied on an iPhone) puts a file URL on the pasteboard
// instead of image bytes, which Electron surfaces as text/uri-list with no
// image/png item (#159184). Reading the file is also what makes macOS finish
// fetching it from the phone.
async function readClipboardImageFile(clipped, maxBytes) {
  let items

  try {
    items = await clipped.read()
  } catch {
    return null
  }

  for (const item of items) {
    if (!Array.isArray(item?.types) || !item.types.includes(URI_LIST_MIME)) {
      continue
    }

    let uriList

    try {
      uriList = await (await item.getType(URI_LIST_MIME)).text()
    } catch {
      continue
    }

    for (const filePath of clipboardFilePaths(uriList)) {
      const ext = path.extname(filePath).toLowerCase()

      if (!IMAGE_FILE_EXTENSIONS.has(ext)) {
        continue
      }

      const data = await readImageFile(filePath, maxBytes)

      if (data) {
        return { data, ext, name: path.basename(filePath) }
      }
    }
  }

  return null
}

// The clipboard's image for the composer: its PNG bytes, else the first image
// file it names (capped at maxBytes), or null.
async function readClipboardImage(clipped, maxBytes) {
  const png = await readClipboardImagePng(clipped)

  return png ? { data: png, ext: '.png', name: '' } : readClipboardImageFile(clipped, maxBytes)
}

export { readClipboardImage, readClipboardImagePng }
