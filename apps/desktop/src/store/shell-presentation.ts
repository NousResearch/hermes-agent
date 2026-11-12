import { useStore } from '@nanostores/react'
import { atom, type ReadableAtom } from 'nanostores'

/**
 * Leaf seam for the shell presentation, read by low-level modules — the icon layer in
 * `@/lib/icons` and `Codicon` — that 200+ files import. Importing `shell-variant` from there would
 * pull the app stores into every importer and cycle back through i18n, so this module imports
 * nothing but nanostores. `shell-variant.ts` provides `$portalShell` here when it loads (at boot,
 * before the first render); the reference is stored, not subscribed, so nothing evaluates at import.
 */
let source: ReadableAtom<boolean> = atom(false)

export function providePortalPresentation(store: ReadableAtom<boolean>): void {
  source = store
}

export function usePortalPresentation(): boolean {
  return useStore(source)
}
