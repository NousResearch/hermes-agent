/**
 * Punctuation that ends a bare URL in prose even with no space before it, for
 * regex character classes. CJK text sets no spaces between a link and the
 * clause around it (`见https://a.com，还有`), so every bare-URL matcher needs
 * these as hard stops. Only clause and bracket punctuation is named: fullwidth
 * letters, digits, `～` and half-width katakana are valid internationalized
 * URL content and stay inside the link.
 *
 * Covers: ideographic space and `、。〃`, CJK brackets `〈〉《》「」『』【】〔〕〖〗〘〙〚〛`,
 * corner quotes `〝〞〟`, typographic quotes and ellipsis `‘’“”…`, fullwidth
 * `！（），．：；？［］｛｝` and half-width `｡｢｣､`.
 */
export const URL_PROSE_BOUNDARY =
  '\\u2018\\u2019\\u201C\\u201D\\u2026\\u3000-\\u3003\\u3008-\\u3011\\u3014-\\u301B\\u301D-\\u301F' +
  '\\uFF01\\uFF08\\uFF09\\uFF0C\\uFF0E\\uFF1A\\uFF1B\\uFF1F\\uFF3B\\uFF3D\\uFF5B\\uFF5D\\uFF61-\\uFF64'
