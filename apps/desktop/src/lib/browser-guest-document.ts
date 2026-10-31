/** The markup of a `data:text/html` URL, or null for any other src.
 *
 *  The Webapp page CSP has no room for guest inline scripts, and a `data:`
 *  document inherits it, so the browser host re-serves this markup through
 *  the gateway's guest shell instead of navigating to the URL. Decoding
 *  follows the URL standard's data: processor: the fragment is not content,
 *  the body is percent-decoded bytes (or base64), and `charset=` picks the
 *  decoder. */
export function htmlFromDataUrl(src: string): null | string {
  let url: URL

  try {
    url = new URL(src)
  } catch {
    return null
  }

  if (url.protocol !== 'data:') {
    return null
  }

  url.hash = ''
  const raw = url.href.slice('data:'.length)
  const comma = raw.indexOf(',')

  if (comma < 0) {
    return null
  }

  const [type = '', ...params] = raw
    .slice(0, comma)
    .split(';')
    .map(part => part.trim().toLowerCase())

  if (type !== 'text/html') {
    return null
  }

  const bytes = params.includes('base64') ? base64Bytes(raw.slice(comma + 1)) : percentBytes(raw.slice(comma + 1))

  if (!bytes) {
    return null
  }

  const charset = params.find(param => param.startsWith('charset='))?.slice('charset='.length) || 'utf-8'

  try {
    return new TextDecoder(charset).decode(bytes)
  } catch {
    return new TextDecoder().decode(bytes)
  }
}

function base64Bytes(body: string): null | Uint8Array {
  try {
    return Uint8Array.from(atob(decodeURIComponent(body).replace(/\s/g, '')), char => char.charCodeAt(0))
  } catch {
    return null
  }
}

function percentBytes(body: string): Uint8Array {
  const input = new TextEncoder().encode(body)
  const output: number[] = []

  for (let index = 0; index < input.length; index += 1) {
    const hex = input[index] === 0x25 ? String.fromCharCode(input[index + 1] ?? 0, input[index + 2] ?? 0) : ''

    if (/^[\da-f]{2}$/i.test(hex)) {
      output.push(parseInt(hex, 16))
      index += 2
    } else {
      output.push(input[index])
    }
  }

  return Uint8Array.from(output)
}
