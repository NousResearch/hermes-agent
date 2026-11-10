import type { DesktopTheme } from './types'

// Portal /ds primitives and Figma Brand rev, adapted to readable desktop pairs.
// Blue stays saturated on filled actions; dark-mode text uses a lighter blue.
export const portalTheme: DesktopTheme = {
  name: 'portal',
  label: 'Hermes Portal',
  description: 'Portal typography and the new Hermes desktop surfaces',
  colors: {
    background: '#f2f2f2',
    foreground: '#171719',
    card: '#fafafa',
    cardForeground: '#171719',
    muted: '#e8e8eb',
    mutedForeground: '#62626b',
    popover: '#fafafa',
    popoverForeground: '#171719',
    primary: '#0000f2',
    primaryForeground: '#f2f2f2',
    secondary: '#e5e5eb',
    secondaryForeground: '#303037',
    accent: '#e2e2f2',
    accentForeground: '#171719',
    border: '#d8d8df',
    input: '#c5c5cf',
    ring: '#0000f2',
    destructive: '#b3293d',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#ededf0',
    sidebarBorder: '#d8d8df',
    userBubble: '#e5e5eb',
    userBubbleBorder: '#d8d8df'
  },
  darkColors: {
    background: '#0e0e10',
    foreground: '#f2f2f2',
    card: '#18181d',
    cardForeground: '#f2f2f2',
    muted: '#202026',
    mutedForeground: '#a1a1ab',
    popover: '#202026',
    popoverForeground: '#f2f2f2',
    primary: '#8585ff',
    primaryForeground: '#0e0e10',
    secondary: '#26262e',
    secondaryForeground: '#f2f2f2',
    accent: '#29293a',
    accentForeground: '#f2f2f2',
    border: '#303038',
    input: '#4e4e5b',
    ring: '#8585ff',
    destructive: '#f0949e',
    destructiveForeground: '#0e0e10',
    sidebarBackground: '#111114',
    sidebarBorder: '#28282f',
    userBubble: '#202026',
    userBubbleBorder: '#303038'
  },
  typography: {
    fontSans: '"Mona Sans", system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", emoji',
    fontUrl: '',
    lineHeight: '1.5',
    letterSpacing: '0'
  },
  customCSS: `
    :root {
      --portal-brand-blue: #0000f2;
      --portal-brand-hover: #2020ff;
      --portal-on-brand: #f2f2f2;
    }

    :is(
      [data-slot="dropdown-menu-content"],
      [data-slot="dropdown-menu-sub-content"],
      [data-slot="context-menu-content"],
      [data-slot="context-menu-sub-content"],
      [data-slot="select-content"],
      [data-slot="popover-content"]:not([data-variant="accent"]),
      .hermes-floating-surface
    ) {
      --popover-surface: var(--ui-bg-elevated);
      background-color: var(--ui-bg-elevated);
      backdrop-filter: none;
      box-shadow: none;
    }

    [data-slot="popover-arrow"],
    [data-slot="dialog-overlay"] {
      backdrop-filter: none;
    }

    [data-slot="dialog-content"] {
      background-color: var(--ui-bg-elevated);
      box-shadow: none;
    }
  `
}
