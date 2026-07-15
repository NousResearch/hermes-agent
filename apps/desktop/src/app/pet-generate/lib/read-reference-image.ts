const DEFAULT_MAX_INPUT_BYTES = 16 * 1024 * 1024

function loadImage(url: string): Promise<HTMLImageElement> {
  const img = new Image()

  return new Promise((resolve, reject) => {
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('unreadable image'))
    img.src = url
  })
}

// Read an image file as a downscaled PNG data URL. We decode from an object URL
// (not readAsDataURL) so large files don't inflate into giant base64 strings
// before we scale them down for generation.
export async function readReferenceImage(
  file: File,
  max = 1024,
  maxInputBytes = DEFAULT_MAX_INPUT_BYTES
): Promise<string> {
  if (file.size > maxInputBytes) {
    throw new Error('reference image too large')
  }

  const objectUrl = URL.createObjectURL(file)

  try {
    const img = await loadImage(objectUrl)
    const scale = Math.min(1, max / Math.max(img.width, img.height))
    const width = Math.max(1, Math.round(img.width * scale))
    const height = Math.max(1, Math.round(img.height * scale))

    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')

    if (!ctx) {
      throw new Error('could not create canvas context')
    }

    ctx.drawImage(img, 0, 0, width, height)

    return canvas.toDataURL('image/png')
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}
