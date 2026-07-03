export const HTML_PREVIEW_CSP = [
  "default-src 'none'",
  "img-src data: blob:",
  "style-src 'unsafe-inline'",
  'font-src data:',
  'media-src data: blob:',
  "script-src 'none'",
  "connect-src 'none'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'"
].join('; ')

const HTML_PREVIEW_CSP_META = `<meta http-equiv="Content-Security-Policy" content="${HTML_PREVIEW_CSP}">`

export function hardenHtmlForPreview(html: string): string {
  const source = html || ''

  if (/<head(?:\s[^>]*)?>/i.test(source)) {
    return source.replace(/<head(\s[^>]*)?>/i, match => `${match}${HTML_PREVIEW_CSP_META}`)
  }

  return `${HTML_PREVIEW_CSP_META}${source}`
}

function bytesFromBase64(encoded: string): Uint8Array {
  const raw = atob(encoded)
  const bytes = new Uint8Array(raw.length)

  for (let i = 0; i < raw.length; i += 1) {
    bytes[i] = raw.charCodeAt(i)
  }

  return bytes
}

export function textFromDataUrl(dataUrl: string): string {
  const match = /^data:([^,]*),(.*)$/s.exec(dataUrl || '')

  if (!match) {
    throw new Error('Invalid data URL')
  }

  const metadata = match[1] || ''
  const data = match[2] || ''

  if (metadata.toLowerCase().includes(';base64')) {
    return new TextDecoder().decode(bytesFromBase64(data))
  }

  return decodeURIComponent(data.replace(/\+/g, '%20'))
}

export function createHardenedHtmlBlobUrl(dataUrl: string): string {
  const html = hardenHtmlForPreview(textFromDataUrl(dataUrl))
  const blob = new Blob([html], { type: 'text/html' })

  return URL.createObjectURL(blob)
}
