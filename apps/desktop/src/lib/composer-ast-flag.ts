/**
 * MessageDocument composer flag. Default on; opt out via
 * `VITE_HERMES_COMPOSER_AST=false` or `localStorage['hermes:composer:ast'] = '0'`.
 */
export const COMPOSER_AST_STORAGE_KEY = 'hermes:composer:ast'

export function isComposerAstEnabled(): boolean {
  if (import.meta.env?.VITE_HERMES_COMPOSER_AST === 'false') {
    return false
  }

  if (import.meta.env?.VITE_HERMES_COMPOSER_AST === 'true') {
    return true
  }

  try {
    if (typeof localStorage !== 'undefined') {
      const stored = localStorage.getItem(COMPOSER_AST_STORAGE_KEY)

      if (stored === '0') {
        return false
      }

      if (stored === '1') {
        return true
      }
    }
  } catch {
    // localStorage can throw in locked-down contexts — treat as enabled.
  }

  return true
}
