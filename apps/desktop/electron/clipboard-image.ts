// Electron 44 rearchitected the clipboard onto the W3C async API: readText /
// writeText return Promises, and image reads no longer have a readImage()
// shortcut — the bytes arrive through clipboard.read() (ClipboardItem[]) and
// ClipboardItem.getType('image/png') as a Blob. This helper keeps the old
// handler contract ("PNG bytes or null") behind the new API.

const PNG_MIME = 'image/png'

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

export { readClipboardImagePng }
