import type { CodeHighlighterPlugin, CodePluginOptions, HighlightOptions, HighlightResult } from '@streamdown/code'

/**
 * Mobile renderer shim for @streamdown/code.
 *
 * The desktop renderer uses @streamdown/code + Shiki for syntax-highlighted
 * tokens. The mobile bundle already aliases react-shiki to a plain-code
 * renderer, so importing @streamdown/code only pulls Shiki into the build and
 * currently breaks Vite/Rolldown on Shiki's generated package layout.
 *
 * Keep Streamdown's code-block parsing path available, but report no supported
 * languages so code blocks render through the local plain-code fallback instead
 * of requiring Shiki in the embedded mobile WebView bundle.
 */
export function createCodePlugin(_options: CodePluginOptions = {}): CodeHighlighterPlugin {
  return {
    name: 'shiki',
    type: 'code-highlighter',
    supportsLanguage: () => false,
    getSupportedLanguages: () => [],
    getThemes: () => ['github-light', 'github-dark'],
    highlight: (_options: HighlightOptions, _callback?: (result: HighlightResult) => void) => null
  }
}

export const code = createCodePlugin()
