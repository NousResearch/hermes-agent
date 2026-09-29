/**
 * #95089 — iPadOS Safari hardware-keyboard contact AutoFill suppression.
 *
 * These attributes are applied to the visible contentEditable editor in both
 * the main chat composer and the user-edit composer. They tell password
 * managers (1Password, LastPass) to stand down on these fields, and are
 * inert in Safari but keep the whole suppression contract in one place.
 *
 * Keeping them as a shared constant means both composers and their tests
 * import the same source of truth — a refactor that drops one of these
 * attributes will fail the tests here rather than resurfacing the iOS
 * AutoFill bar for users.
 */

/** Attributes applied to the visible contentEditable editor. */
export const EDITOR_SUPPRESSION_ATTRS = {
  'data-1p-ignore': '',
  'data-composer-rich-input': '',
  'data-lpignore': 'true',
} as const

/** Attributes applied to the form primitive. */
export const FORM_SUPPRESSION_ATTRS = {
  autoComplete: 'off',
} as const

/** Attributes applied to the hidden sr-only textarea. */
export const TEXTAREA_SUPPRESSION_ATTRS = {
  autoComplete: 'off',
  autoCapitalize: 'off',
  autoCorrect: 'off',
  spellCheck: false,
} as const

/** All required suppression attribute names for the editor. */
export const EDITOR_REQUIRED_ATTRS = Object.keys(EDITOR_SUPPRESSION_ATTRS) as Array<
  keyof typeof EDITOR_SUPPRESSION_ATTRS
>

