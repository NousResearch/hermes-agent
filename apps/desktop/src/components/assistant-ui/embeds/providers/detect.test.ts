import { describe, expect, it } from 'vitest'

import type { FrameEmbed, TweetEmbed } from './types'

import { detectEmbed, isEmbeddableUrl } from './index'

function frame(url: string): FrameEmbed {
  const descriptor = detectEmbed(url)

  if (!descriptor || descriptor.renderer !== 'frame') {
    throw new Error(`expected a frame embed for ${url}`)
  }

  return descriptor
}

describe('detectEmbed — YouTube', () => {
  it.each([
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'https://youtu.be/dQw4w9WgXcQ',
    'https://www.youtube.com/shorts/dQw4w9WgXcQ',
    'https://m.youtube.com/watch?v=dQw4w9WgXcQ',
    'https://www.youtube.com/embed/dQw4w9WgXcQ',
    'https://www.youtube.com/live/dQw4w9WgXcQ'
  ])('resolves %s to the privacy-enhanced embed of the same id', url => {
    const embed = frame(url)

    expect(embed.provider).toBe('youtube')
    expect(embed.id).toBe('youtube:dQw4w9WgXcQ')
    expect(embed.embedUrl).toContain('youtube-nocookie.com/embed/dQw4w9WgXcQ')
    expect(embed.previewUrl).toBe('https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg')
  })

  it('carries a start time from t/start through to the embed', () => {
    expect(frame('https://youtu.be/dQw4w9WgXcQ?t=90').embedUrl).toContain('start=90')
    expect(frame('https://youtu.be/dQw4w9WgXcQ?t=1m30s').embedUrl).toContain('start=90')
  })

  it('rejects ids that are not 11 chars', () => {
    expect(detectEmbed('https://www.youtube.com/watch?v=short')).toBeNull()
  })
})

describe('detectEmbed — other frame providers', () => {
  it('resolves Vimeo numeric ids across path shapes', () => {
    expect(frame('https://vimeo.com/76979871').embedUrl).toBe('https://player.vimeo.com/video/76979871')
    expect(frame('https://vimeo.com/channels/staffpicks/76979871').id).toBe('vimeo:76979871')
  })

  it('resolves Instagram posts and reels', () => {
    expect(frame('https://www.instagram.com/p/CabcDEF123/').embedUrl).toBe(
      'https://www.instagram.com/p/CabcDEF123/embed'
    )
    expect(frame('https://www.instagram.com/reel/CabcDEF123/').embedUrl).toContain('/reel/CabcDEF123/embed')
    expect(frame('https://www.instagram.com/reels/CabcDEF123/').embedUrl).toContain('/reel/CabcDEF123/embed')
  })

  it('resolves Pinterest pins across locale hosts', () => {
    expect(frame('https://www.pinterest.com/pin/1234567890/').embedUrl).toBe(
      'https://assets.pinterest.com/ext/embed.html?id=1234567890'
    )
    expect(frame('https://fr.pinterest.com/pin/1234567890/').provider).toBe('pinterest')
  })

  it('resolves TikTok videos to the official player', () => {
    expect(frame('https://www.tiktok.com/@user/video/7212345678901234567').embedUrl).toBe(
      'https://www.tiktok.com/player/v1/7212345678901234567'
    )
  })

  it('resolves Spotify tracks, collections, and locale-prefixed urls', () => {
    expect(frame('https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT').embedUrl).toBe(
      'https://open.spotify.com/embed/track/4cOdK2wGLETKBW3PvgPWqT'
    )
    expect(frame('https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M').provider).toBe('spotify')
    expect(frame('https://open.spotify.com/intl-de/album/1DFixLWuPkv3KT3TnV35m3').id).toBe(
      'spotify:album:1DFixLWuPkv3KT3TnV35m3'
    )
    expect(detectEmbed('https://open.spotify.com/track/')).toBeNull()
  })
})

describe('detectEmbed — maps', () => {
  it('resolves Google Maps coordinates with zoom', () => {
    const embed = frame('https://www.google.com/maps/@40.7128,-74.0060,12z')

    expect(embed.provider).toBe('googlemaps')
    expect(embed.embedUrl).toContain('output=embed')
    expect(embed.embedUrl).toContain('q=40.7128%2C-74.006')
    expect(embed.embedUrl).toContain('z=12')
  })

  it('resolves a Google Maps place name', () => {
    expect(frame('https://www.google.com/maps/place/Eiffel+Tower/').embedUrl).toContain('q=Eiffel+Tower')
  })

  it('resolves OpenStreetMap fragment state to a bbox embed', () => {
    const embed = frame('https://www.openstreetmap.org/#map=12/40.7128/-74.0060')

    expect(embed.provider).toBe('openstreetmap')
    expect(embed.embedUrl).toContain('export/embed.html')
    expect(embed.embedUrl).toContain('marker=40.7128%2C-74.006')
    expect(embed.embedUrl).toContain('bbox=')
  })
})

describe('detectEmbed — Twitter/X', () => {
  it('resolves twitter.com and x.com status urls to a tweet descriptor', () => {
    for (const url of ['https://twitter.com/jack/status/20', 'https://x.com/jack/status/20']) {
      const descriptor = detectEmbed(url)

      expect(descriptor?.renderer).toBe('tweet')
      expect((descriptor as TweetEmbed).tweetId).toBe('20')
    }
  })
})

describe('detectEmbed — non-matches', () => {
  it.each([
    'https://example.com/watch?v=dQw4w9WgXcQ',
    'https://github.com/NousResearch/hermes',
    'not-a-url',
    'ftp://youtube.com/watch?v=dQw4w9WgXcQ',
    'mailto:someone@youtube.com'
  ])('returns null for %s', url => {
    expect(detectEmbed(url)).toBeNull()
    expect(isEmbeddableUrl(url)).toBe(false)
  })

  it('handles empty input without throwing', () => {
    expect(detectEmbed(undefined)).toBeNull()
    expect(detectEmbed('')).toBeNull()
  })
})
