import { bareHost, type EmbedMatcher } from './types'

// Spotify's embed has only two layouts: compact (≤152) and full (352). Any
// in-between height renders the compact player and pads the rest with grey, so
// we snap to the compact size for every type — tight, no dead space.
const COMPACT_HEIGHT = 152
const EMBED_TYPES = new Set(['album', 'artist', 'episode', 'playlist', 'show', 'track'])

export const spotify: EmbedMatcher = url => {
  if (bareHost(url.hostname) !== 'open.spotify.com') {
    return null
  }

  // Drop an optional locale prefix (`/intl-de/track/...`).
  const segments = url.pathname.split('/').filter(Boolean)
  const start = segments[0]?.startsWith('intl-') ? 1 : 0
  const type = segments[start] || ''
  const id = segments[start + 1] || ''

  if (!EMBED_TYPES.has(type) || !/^[A-Za-z0-9]+$/.test(id)) {
    return null
  }

  return {
    embedUrl: `https://open.spotify.com/embed/${type}/${id}`,
    height: COMPACT_HEIGHT,
    id: `spotify:${type}:${id}`,
    label: 'Spotify',
    maxWidth: 480,
    provider: 'spotify',
    renderer: 'frame',
    sourceUrl: url.toString()
  }
}
