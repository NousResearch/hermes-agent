/**
 * Canonical Hermes skin — the theme SDK's cross-surface contract.
 *
 * A skin is authored once as YAML in `$HERMES_HOME/skins/<name>.yaml` (or a
 * built-in), resolved by the Python skin engine (`hermes_cli/skin_engine.py`),
 * and pushed to every surface over JSON-RPC (`gateway.ready`, `skin.changed`,
 * `config.get skin`). This is the ONE shape every TypeScript surface consumes;
 * each owns a resolver that normalizes it into its render model:
 *
 *   • TUI     → `fromSkin` → ansi-safe `Theme` (Ink)
 *   • Desktop → `skinToDesktopTheme` → CSS custom properties (Tailwind/shadcn)
 *   • CLI     → `hermes_cli/skin_engine` → prompt_toolkit / Rich styles (Python)
 *
 * Tokens are terminal-first (the CLI is the oldest surface); GUIs derive their
 * fuller palettes from the load-bearing few. Every field is optional — a resolver
 * falls back to its own default for anything a skin omits.
 */

/** Canonical semantic color tokens a skin may set (the "enum" of the shape). */
export const SKIN_COLOR_TOKENS = [
  // Base surface — GUIs + the TUI status bar derive their palette from this.
  'background',
  // Brand accent + primary.
  'ui_accent',
  'ui_primary',
  'banner_accent',
  'banner_title',
  // Text.
  'ui_text',
  'banner_text',
  'banner_dim',
  // Structure.
  'ui_border',
  'banner_border',
  // Semantic status.
  'ui_ok',
  'ui_warn',
  'ui_error',
  'ui_label',
  // Element-specific (fall back to accent/muted when unset).
  'ui_tool',
  'ui_thinking',
  'diff_added',
  'diff_removed',
  'diff_added_word',
  'diff_removed_word',
  'syntax_string',
  'syntax_number',
  'syntax_keyword',
  'syntax_comment',
  // CLI / TUI chrome.
  'prompt',
  'input_rule',
  'response_border',
  'shell_dollar',
  'selection_bg',
  'session_label',
  'session_border',
  'status_bar_bg',
  'status_bar_text',
  'status_bar_strong',
  'status_bar_dim',
  'status_bar_good',
  'status_bar_warn',
  'status_bar_bad',
  'status_bar_critical',
  'voice_status_bg',
  'completion_menu_bg',
  'completion_menu_current_bg',
  'completion_menu_meta_bg',
  'completion_menu_meta_current_bg'
] as const

export type SkinColorToken = (typeof SKIN_COLOR_TOKENS)[number]

/** Canonical branding/string tokens. */
export const SKIN_BRANDING_TOKENS = [
  'agent_name',
  'welcome',
  'goodbye',
  'response_label',
  'prompt_symbol',
  'help_header'
] as const

export type SkinBrandingToken = (typeof SKIN_BRANDING_TOKENS)[number]

/** Hex color per token. Open-ended so back-compat / niche keys still round-trip. */
export type SkinColors = Partial<Record<SkinColorToken, string>> & { [key: string]: string | undefined }

/** Branding strings per token. Open-ended for the same reason. */
export type SkinBranding = Partial<Record<SkinBrandingToken, string>> & { [key: string]: string | undefined }

/** The resolved skin payload (matches Python's `resolve_skin()`). */
export interface HermesSkin {
  name?: string
  description?: string
  colors?: SkinColors
  /** Hand-tuned palette overlay for dark terminals (light-authored skins).
   *  A resolver picks colors/light_colors/dark_colors by the terminal's
   *  detected polarity — see the TUI's `themeForSkin`. */
  dark_colors?: SkinColors
  /** Hand-tuned palette overlay for light terminals (dark-authored skins). */
  light_colors?: SkinColors
  branding?: SkinBranding
  banner_logo?: string
  banner_hero?: string
  tool_prefix?: string
  help_header?: string
}
