// Shared height cap for inline embeds. Ratio embeds cap their width off this in
// UrlEmbed so height follows the aspect ratio; fenced renderers (mermaid, svg)
// reuse it directly. Pure CSS — no measuring.
export const EMBED_MAX_H = '33dvh'

// Fallback height (px) for non-ratio embeds that don't declare one: the consent
// placeholder, the intrinsic-size hint, and social frames until they measure.
export const EMBED_DEFAULT_H = 320
