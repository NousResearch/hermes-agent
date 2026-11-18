/**
 * What the design system page reads, as the build ships it: the token CSS and DESIGN.md's text. The
 * page renders from these, so it shows what the code defines rather than a restatement of it.
 */
import { markdownSection } from '@/lib/markdown-section'

import designDoc from '../../../../DESIGN.md?raw'

const tokenFiles = import.meta.glob<string>('../tokens/*.css', { eager: true, import: 'default', query: '?raw' })

/** Each token file's text by stem (`raw` → raw.css). */
export const TOKEN_SOURCES: Readonly<Record<string, string>> = Object.fromEntries(
  Object.entries(tokenFiles).map(([path, css]) => [path.replace(/^.*\/([\w-]+)\.css$/, '$1'), css])
)

export const STILL_TO_UNIFY = markdownSection(designDoc, 'Still to unify') ?? []
