import { decodeWebText } from './web-text-decoder'

const CONTENT_TYPE_MARK = 'hermes-content-type:'
const URL_EFFECTIVE_MARK = 'hermes-url-effective:'

export const CURL_TITLE_WRITE_OUT = `\n${CONTENT_TYPE_MARK}%{content_type}\n${URL_EFFECTIVE_MARK}%{url_effective}`

export function parseCurlTitleResponse(bodyWithTrailer: Buffer, tail: Buffer): { effectiveUrl: string; html: string } {
  const contentTypeMarker = Buffer.from(`\n${CONTENT_TYPE_MARK}`)
  const at = bodyWithTrailer.lastIndexOf(contentTypeMarker)
  const trailer = (at >= 0 ? bodyWithTrailer.subarray(at) : tail).toString('utf8')
  const contentType = trailer.match(new RegExp(`(?:^|\\n)${CONTENT_TYPE_MARK}([^\\n]*)`))?.[1]?.trim() ?? ''
  const effectiveUrl = trailer.match(new RegExp(`(?:^|\\n)${URL_EFFECTIVE_MARK}([^\\n]*)`))?.[1]?.trim() ?? ''
  const body = at >= 0 ? bodyWithTrailer.subarray(0, at) : bodyWithTrailer

  return { effectiveUrl, html: decodeWebText(body, contentType) }
}
