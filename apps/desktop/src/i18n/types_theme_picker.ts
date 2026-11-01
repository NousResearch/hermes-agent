/** The set_theme picker cards in the chat and their settled receipt. */
export interface ThemePickerCopy {
  accent: string
  layout: string
  mode: string
  customColor: string
  /** A settled accent that is the theme's own colour (no pick stored). */
  themeAccent: string
  basic: string
  basicDetail: string
  elite: string
  eliteDetail: string
  /** The card closed without a pick: dismissed, timed out, or answered in the composer. */
  noPick: string
}
