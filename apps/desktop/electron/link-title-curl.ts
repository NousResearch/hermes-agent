import type { TitleFetchTarget } from './link-title-guard'
import { decodeWebText } from './web-text-decoder'

const CONTENT_TYPE_MARK = 'hermes-content-type:'
const URL_EFFECTIVE_MARK = 'hermes-url-effective:'
const URL_REDIRECT_MARK = 'hermes-url-redirect:'
const HTTP_CODE_MARK = 'hermes-http-code:'

// The curl tier follows redirects manually (#126885) so every hop can be
// re-admitted by the destination guard before it is dialed; %{redirect_url}
// and %{http_code} are what decides whether a hop happened at all.
export const CURL_TITLE_WRITE_OUT = `\n${CONTENT_TYPE_MARK}%{content_type}\n${URL_EFFECTIVE_MARK}%{url_effective}\n${URL_REDIRECT_MARK}%{redirect_url}\n${HTTP_CODE_MARK}%{http_code}`

export interface CurlTitleResult {
  effectiveUrl: string
  html: string
  redirectUrl: string
  httpCode: number
}

export function parseCurlTitleResponse(bodyWithTrailer: Buffer, tail: Buffer): CurlTitleResult {
  const contentTypeMarker = Buffer.from(`\n${CONTENT_TYPE_MARK}`)
  const at = bodyWithTrailer.lastIndexOf(contentTypeMarker)
  const trailer = (at >= 0 ? bodyWithTrailer.subarray(at) : tail).toString('utf8')
  const field = (mark: string) => trailer.match(new RegExp(`(?:^|\\n)${mark}([^\\n]*)`))?.[1]?.trim() ?? ''
  const contentType = field(CONTENT_TYPE_MARK)
  const effectiveUrl = field(URL_EFFECTIVE_MARK)
  const redirectUrl = field(URL_REDIRECT_MARK)
  const httpCode = Number.parseInt(field(HTTP_CODE_MARK), 10)
  const body = at >= 0 ? bodyWithTrailer.subarray(0, at) : bodyWithTrailer

  return {
    effectiveUrl,
    html: decodeWebText(body, contentType),
    redirectUrl,
    httpCode: Number.isFinite(httpCode) ? httpCode : 0
  }
}

/**
 * curl arguments that make one hop dial exactly what admission vetted.
 *
 * Direct: `--noproxy '*'` (no env proxy the guard did not account for) and
 * `--resolve host:port:addr` pinned to a vetted DNS answer, so curl never
 * resolves the name a second time — a rebinding answer between the check and
 * the connect cannot reach a private address. Proxied: the same proxy the
 * guard saw, with `--noproxy ''` overriding NO_PROXY, so a host the guard
 * never resolved is not dialed directly either.
 */
export function curlTitleTargetArgs(target: TitleFetchTarget): string[] {
  if (target.proxy) {
    return ['--proxy', target.proxy, '--noproxy', '']
  }

  const [address] = target.addresses

  if (!address) {
    return ['--noproxy', '*']
  }

  const pinned = address.includes(':') ? `[${address}]` : address

  return ['--noproxy', '*', '--resolve', `${target.hostname}:${target.port}:${pinned}`]
}
