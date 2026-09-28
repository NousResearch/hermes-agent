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
