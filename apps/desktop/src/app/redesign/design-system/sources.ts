/**
 * What the design system page reads, as the build ships it: the token CSS and DESIGN.md's text. The
 * page renders from these, so it shows what the code defines rather than a restatement of it.
 */
import { type MarkdownBlock, markdownSection } from '@/lib/markdown-section'

const tokenFiles = import.meta.glob<string>('../tokens/*.css', { eager: true, import: 'default', query: '?raw' })

/** Each token file's text by stem (`raw` → raw.css). */
export const TOKEN_SOURCES: Readonly<Record<string, string>> = Object.fromEntries(
  Object.entries(tokenFiles).map(([path, css]) => [path.replace(/^.*\/([\w-]+)\.css$/, '$1'), css])
)

let stillToUnifyPromise: Promise<MarkdownBlock[]> | null = null

/** Loaded on demand so DESIGN.md never ships inside the app's main bundle. */
export function loadStillToUnify(): Promise<MarkdownBlock[]> {
  stillToUnifyPromise ??= import('../../../../DESIGN.md?raw').then(
    mod => markdownSection(mod.default, 'Still to unify') ?? []
  )

  return stillToUnifyPromise
}
