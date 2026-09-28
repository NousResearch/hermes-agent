import { bareHost, type EmbedMatcher } from './types'

export const tiktok: EmbedMatcher = url => {
  if (bareHost(url.hostname) !== 'tiktok.com') {
    return null
  }

  const segments = url.pathname.split('/').filter(Boolean)
  const videoIndex = segments.indexOf('video')
  const id = videoIndex >= 0 ? segments[videoIndex + 1] : ''

  if (!/^\d+$/.test(id || '')) {
    return null
  }

  return {
    // The official player is a clean dark video iframe (no white blockquote
    // chrome), so it goes through the plain-iframe frame path, sized 9:16.
    aspectRatio: 9 / 16,
    embedUrl: `https://www.tiktok.com/player/v1/${id}`,
    id: `tiktok:${id}`,
    label: 'TikTok',
    maxWidth: 365,
    provider: 'tiktok',
    renderer: 'frame',
    sourceUrl: url.toString()
  }
}
