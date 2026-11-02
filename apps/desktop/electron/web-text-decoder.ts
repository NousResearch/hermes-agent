const META_SCAN_BYTES = 8192

function charsetFromContentType(contentType: string): string {
  const match = contentType.match(/(?:^|;)\s*charset\s*=\s*(?:"([^"]+)"|'([^']+)'|([^;\s]+))/i)

  return (match?.slice(1).find(Boolean) ?? '').trim().slice(0, 64)
}

function charsetFromHtml(bytes: Uint8Array): string {
  // Charset declarations are ASCII even when the document is not. Decode a
  // small prefix as a single-byte encoding so invalid UTF-8 cannot erase the
  // declaration before we know which decoder to use.
  const head = new TextDecoder('windows-1252').decode(bytes.subarray(0, META_SCAN_BYTES))

  for (const tag of head.match(/<meta\b[^>]*>/gi) ?? []) {
    const direct = tag.match(/\bcharset\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'/>;]+))/i)
    const content = tag.match(/\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i)

    const nested = (content?.slice(1).find(Boolean) ?? '').match(
      /(?:^|;)\s*charset\s*=\s*(?:"([^"]+)"|'([^']+)'|([^;\s]+))/i
    )

    const label = direct?.slice(1).find(Boolean) ?? nested?.slice(1).find(Boolean)

    if (label) {
      return label.trim().slice(0, 64)
    }
  }

  return ''
}

/** Decode fetched page/manifest bytes the same way a browser would choose a
 * character encoding: HTTP header, then an HTML meta declaration, then UTF-8. */
export function decodeWebText(bytes: Uint8Array, contentType = ''): string {
  const labels = [charsetFromContentType(contentType), charsetFromHtml(bytes), 'utf-8']

  for (const [index, label] of labels.entries()) {
    if (!label || labels.indexOf(label) !== index) {
      continue
    }

    try {
      return new TextDecoder(label).decode(bytes)
    } catch {
      // Unknown/malformed server labels fall through to the next source.
    }
  }

  return new TextDecoder().decode(bytes)
}
