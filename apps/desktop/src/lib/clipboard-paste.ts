const PNG_SIGNATURE = '\x89PNG\r\n\x1A\n'
const PNG_SIGNATURE_WITH_REPLACEMENT = '\uFFFDPNG\r\n\x1A\n'

export function shouldBlockPlainTextPaste(text: string): boolean {
  if (!text) {
    return false
  }

  if (text.startsWith(PNG_SIGNATURE) || text.startsWith(PNG_SIGNATURE_WITH_REPLACEMENT)) {
    return true
  }

  const sample = text.slice(0, 4096)
  let controlCount = 0

  for (let index = 0; index < sample.length; index += 1) {
    const code = sample.charCodeAt(index)
    const isAllowedWhitespace = code === 9 || code === 10 || code === 13

    if ((code < 32 && !isAllowedWhitespace) || code === 127 || code === 65533) {
      controlCount += 1
    }
  }

  return sample.length >= 16 && controlCount / sample.length > 0.15
}
