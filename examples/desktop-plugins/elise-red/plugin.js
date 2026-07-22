// Elise Red — example assistant-bubble theme for Hermes Desktop
//
// This is a worked TEMPLATE showing how to give the assistant message its own
// surface color using the opt-in `assistantBubble` theme color (added by PR for
// issue #67290). Drop this file at:
//
//   $HERMES_HOME/desktop-plugins/elise-red/plugin.js
//
// (folder name must equal the plugin id). The app hot-reloads it within seconds;
// toggle in Settings -> Plugins. No app rebuild needed.
//
// REQUIRES the upstream change that adds `--dt-assistant-bubble` and wires it into
// the assistant message root. Without that build, this theme still applies (it is a
// valid DesktopTheme) but the assistant bubble stays unset.
//
// To make your OWN color: change DRAGON_RED (and the dark-mode surface) below.
//
// Disk plugin: plain ESM, no build step. Only @hermes/plugin-sdk / react /
// react/jsx-runtime may be imported. A theme contribution needs no JSX, just a data object.

import { THEMES_AREA } from '@hermes/plugin-sdk'

const NOUS_BLUE = '#0053FD'
const PSYCHE_WARM = '#FFE6CB'
const nousTint = (pct) => `color-mix(in srgb, ${NOUS_BLUE} ${pct}%, #FFFFFF)`
const nousTintTransparent = (pct) => `color-mix(in srgb, ${NOUS_BLUE} ${pct}%, transparent)`

// === CHANGE THIS to make the assistant bubble your color =========================
const DRAGON_RED = '#C8102E' // light-mode assistant bubble surface
const DRAGON_RED_DARK = '#3A0E18' // dark-mode assistant bubble surface (deeper reads better)
// =================================================================================

// Nous palette (mirrors the built-in "Nous" theme) + the assistant bubble color.
const eliseRedTheme = {
  name: 'elise-red',
  label: 'Elise Red',
  description: 'Nous palette with a dragon-red assistant bubble',
  colors: {
    background: '#F8FAFF',
    foreground: '#17171A',
    card: '#FFFFFF',
    cardForeground: '#17171A',
    muted: nousTint(5),
    mutedForeground: '#666678',
    popover: '#FFFFFF',
    popoverForeground: '#17171A',
    primary: NOUS_BLUE,
    primaryForeground: '#FCFCFC',
    secondary: nousTint(7),
    secondaryForeground: '#242432',
    accent: nousTint(10),
    accentForeground: '#202030',
    border: nousTintTransparent(22),
    input: nousTintTransparent(30),
    ring: NOUS_BLUE,
    midground: NOUS_BLUE,
    composerRing: NOUS_BLUE,
    destructive: '#C72E4D',
    destructiveForeground: '#FFFFFF',
    sidebarBackground: '#F3F7FF',
    sidebarBorder: nousTintTransparent(18),
    userBubble: nousTint(6),
    userBubbleBorder: nousTintTransparent(24),
    // The only addition: give the assistant its own surface.
    assistantBubble: DRAGON_RED
  },
  darkColors: {
    background: '#0D2F86',
    foreground: PSYCHE_WARM,
    card: '#12378F',
    cardForeground: PSYCHE_WARM,
    muted: '#183F9A',
    mutedForeground: '#B5C7F3',
    popover: '#123A96',
    popoverForeground: PSYCHE_WARM,
    primary: PSYCHE_WARM,
    primaryForeground: '#0D2F86',
    secondary: '#1B45A4',
    secondaryForeground: '#E0E8FF',
    accent: '#1540B1',
    accentForeground: '#F0F4FF',
    border: '#3158AD',
    input: '#0B2566',
    ring: PSYCHE_WARM,
    midground: NOUS_BLUE,
    composerRing: PSYCHE_WARM,
    destructive: '#C0473A',
    destructiveForeground: '#FEF2F2',
    sidebarBackground: '#09286F',
    sidebarBorder: '#234A9C',
    userBubble: '#143B91',
    userBubbleBorder: '#3A63BD',
    // Dark mode: a deep red surface reads better than bright red.
    assistantBubble: DRAGON_RED_DARK
  },
  typography: {
    fontSans:
      '"Segoe WPC", "Segoe UI", -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif',
    fontMono:
      '"Cascadia Code", "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace'
  }
}

export default {
  id: 'elise-red',
  name: 'Elise Red Theme',
  register(ctx) {
    ctx.register({ id: 'theme', area: THEMES_AREA, data: eliseRedTheme })
  }
}
