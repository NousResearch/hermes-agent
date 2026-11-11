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
    muted: '#e8e8e8',
    mutedForeground: '#62626b',
    popover: '#fafafa',
    popoverForeground: '#171719',
    primary: '#0000f2',
    primaryForeground: '#f2f2f2',
    secondary: '#e5e5e5',
    secondaryForeground: '#303037',
    accent: '#e8e8e8',
    accentForeground: '#171719',
    border: '#d8d8d8',
    input: '#c5c5c5',
    ring: '#0000f2',
    destructive: '#b3293d',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#eeeeee',
    sidebarBorder: '#d8d8d8',
    userBubble: '#e5e5e5',
    userBubbleBorder: '#d8d8d8'
  },
  darkColors: {
    background: '#07071a',
    foreground: '#f2f2f2',
    card: '#131325',
    cardForeground: '#f2f2f2',
    muted: '#1c1c20',
    mutedForeground: '#a1a1ab',
    popover: '#0d0d1e',
    popoverForeground: '#f2f2f2',
    primary: '#8585ff',
    primaryForeground: '#0e0e10',
    secondary: '#202024',
    secondaryForeground: '#f2f2f2',
    accent: '#242429',
    accentForeground: '#f2f2f2',
    border: '#29292f',
    input: '#383840',
    ring: '#8585ff',
    destructive: '#f0949e',
    destructiveForeground: '#0e0e10',
    sidebarBackground: '#0a0a1a',
    sidebarBorder: '#171727',
    userBubble: '#131325',
    userBubbleBorder: '#131325'
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
      --ui-bg-chrome: var(--theme-background-seed);
      --ui-bg-elevated: var(--theme-elevated-seed);
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
