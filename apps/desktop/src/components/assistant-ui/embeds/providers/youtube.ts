import { bareHost, type EmbedMatcher } from './types'

const YOUTUBE_ID_RE = /^[A-Za-z0-9_-]{11}$/

// `t`/`start` accept either raw seconds ("90") or the "1m30s" form.
function startSeconds(value: string | null): number | undefined {
  if (!value) {
    return undefined
  }

  if (/^\d+$/.test(value)) {
    return Number(value)
  }

  const match = value.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$/)

  if (!match || !match[0]) {
    return undefined
  }

  const seconds = Number(match[1] || 0) * 3600 + Number(match[2] || 0) * 60 + Number(match[3] || 0)

  return seconds > 0 ? seconds : undefined
}

export const youtube: EmbedMatcher = url => {
  const host = bareHost(url.hostname)
  const segments = url.pathname.split('/').filter(Boolean)
  let id = ''

  if (host === 'youtu.be') {
    id = segments[0] || ''
  } else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
    if (segments[0] === 'watch') {
      id = url.searchParams.get('v') || ''
    } else if (['embed', 'shorts', 'live', 'v'].includes(segments[0] || '')) {
      id = segments[1] || ''
    }
  } else {
    return null
  }

  if (!YOUTUBE_ID_RE.test(id)) {
    return null
  }

  const params = new URLSearchParams({ modestbranding: '1', rel: '0' })

  const start = startSeconds(url.searchParams.get('t') || url.searchParams.get('start'))

  if (start) {
    params.set('start', String(start))
  }

  return {
    aspectRatio: 16 / 9,
    embedUrl: `https://www.youtube-nocookie.com/embed/${id}?${params.toString()}`,
    id: `youtube:${id}`,
    label: 'YouTube',
    maxWidth: 640,
    provider: 'youtube',
    renderer: 'frame',
    sourceUrl: url.toString()
  }
}
