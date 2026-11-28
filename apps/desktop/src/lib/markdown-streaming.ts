import { tailBoundedRemend } from '@assistant-ui/react-streamdown'

import { preprocessMarkdown } from './markdown-preprocess'

// Keep the renderer and its tests on one module-scope, stable callback.
export function preprocessWithTailRepair(text: string): string {
  try {
    // Preprocessing already escapes prose tildes while preserving URLs.
    // remend's duplicate single-tilde pass can insert backslashes into URLs;
    // leave its other repairs, including incomplete strikethrough, enabled.
    return tailBoundedRemend(preprocessMarkdown(text), { singleTilde: false })
  } catch {
    return text
  }
}
