// Embed provider model. Detection is pure, synchronous, and dependency-free so
// it is safe to run during render and trivial to unit-test. Rendering lives in
// the lazy renderers (see ../registry.tsx) keyed off `renderer`.

export type EmbedProvider =
  | 'googlemaps'
  | 'instagram'
  | 'openstreetmap'
  | 'pinterest'
  | 'spotify'
  | 'tiktok'
  | 'twitter'
  | 'vimeo'
  | 'youtube'

/** Which lazy renderer materialises the descriptor. */
export type EmbedRenderer = 'frame' | 'tweet'

interface EmbedLayout {
  /** Frame aspect ratio (width / height). For video/maps. */
  aspectRatio?: number
  /** Fixed pixel height for non-ratio embeds (Instagram, Pinterest, Spotify). */
  height?: number
  /** Max rendered width in px; falls back to the conversation column. */
  maxWidth?: number
}

interface BaseEmbed extends EmbedLayout {
  /** Stable id for React keys / dedupe. */
  id: string
  /** Human-facing provider name (e.g. "YouTube"). */
  label: string
  /** Optional image shown before the user consents to load the full embed. */
  previewUrl?: string
  provider: EmbedProvider
  renderer: EmbedRenderer
  /** Canonical URL opened in the system browser from the card. */
  sourceUrl: string
}

/** A provider whose embed is a single iframe URL (video, post, map, ...). */
export interface FrameEmbed extends BaseEmbed {
  /** URL loaded inside the iframe. */
  embedUrl: string
  renderer: 'frame'
}

/** Twitter/X ships no iframe URL — only a widget script (see social-embed.tsx). */
export interface TweetEmbed extends BaseEmbed {
  renderer: 'tweet'
  tweetId: string
}

export type EmbedDescriptor = FrameEmbed | TweetEmbed

/** A provider matcher. Receives a parsed http(s) URL; returns null if unmatched. */
export type EmbedMatcher = (url: URL) => EmbedDescriptor | null

/** Strip a leading `www.`/`m.`/`mobile.` so host checks read cleanly. */
export function bareHost(host: string): string {
  return host.replace(/^(?:www|m|mobile)\./i, '').toLowerCase()
}
