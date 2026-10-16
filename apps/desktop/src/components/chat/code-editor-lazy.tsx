/**
 * LAZY CodeMirror editor — light entry-graph module.
 *
 * `@codemirror/*` + `@lezer/*` only matter in edit-mode surfaces (file
 * preview editing, skill/profile/config editors), none of which paint on
 * boot. Importing `code-editor.tsx` statically pulls all of CodeMirror into
 * the entry chunk, so call sites render through this boundary instead: the
 * editor chunk loads the first time an edit surface opens.
 */

import { type ComponentProps, lazy, type ReactNode, Suspense } from 'react'

import type { CodeEditor as CodeEditorComponent } from './code-editor'

const CodeEditor = lazy(() => import('./code-editor').then(module => ({ default: module.CodeEditor })))

export type LazyCodeEditorProps = ComponentProps<typeof CodeEditorComponent> & {
  fallback?: ReactNode
}

export function LazyCodeEditor({ fallback = null, ...props }: LazyCodeEditorProps) {
  return (
    <Suspense fallback={fallback}>
      <CodeEditor {...props} />
    </Suspense>
  )
}
