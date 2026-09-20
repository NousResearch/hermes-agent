import { bareHost, type EmbedMatcher } from './types'

export const instagram: EmbedMatcher = url => {
  if (bareHost(url.hostname) !== 'instagram.com') {
    return null
  }

  const [typeRaw, code] = url.pathname.split('/').filter(Boolean)
  const type = typeRaw === 'reels' ? 'reel' : typeRaw

  if (!code || !['p', 'reel', 'tv'].includes(type || '') || !/^[A-Za-z0-9_-]+$/.test(code)) {
    return null
  }

  return {
    embedUrl: `https://www.instagram.com/${type}/${code}/embed`,
    // Placeholder height for content-visibility; embed.js self-sizes in-document.
    height: 450,
    id: `instagram:${code}`,
    label: 'Instagram',
    maxWidth: 400,
    provider: 'instagram',
    renderer: 'frame',
    sourceUrl: url.toString()
  }
}
