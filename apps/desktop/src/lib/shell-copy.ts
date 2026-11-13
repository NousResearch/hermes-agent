import { portalPresentation } from '@/store/shell-presentation'

/**
 * A hardcoded (non-catalog) message in each shell's wording. Classic is `main`'s text verbatim,
 * redesign is the redesign vocabulary (Machines, Agents, ...). Catalog strings get the same split
 * from the redesign overlay (`@/i18n/redesign`); these stay English-only in both shells, as on
 * `main`, so they live beside their call sites instead of in every locale.
 */
export interface ShellCopy<T = string> {
  classic: T
  redesign: T
}

/** THIS window's wording of `copy`, read per call (the HUD and popped-out Browser render classic).
 *  Depends only on the presentation leaf, so dependency-free modules can use it. */
export function shellCopy<T>(copy: ShellCopy<T>): T {
  return portalPresentation() ? copy.redesign : copy.classic
}
