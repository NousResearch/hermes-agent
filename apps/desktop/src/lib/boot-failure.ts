import collapseBold from '@nous-research/ui/fonts/Collapse-Bold.woff2?url'

import { isTransparentWindow } from './transparent-windows'

type BootPalette = Record<'accent' | 'accentForeground' | 'background' | 'foreground', string>

/** The theme's snapshot that themes/boot-palette.ts writes, or null. Kept here rather than
 *  imported, so the renderer entry never needs another chunk before it can show this notice. */
export function readBootPalette(): BootPalette | null {
  try {
    const value: unknown = JSON.parse(localStorage.getItem('hermes-boot-palette') ?? 'null')
    const fields = ['accent', 'accentForeground', 'background', 'foreground'] as const

    const standalone = (field: (typeof fields)[number]) => {
      const color: unknown = (value as Partial<BootPalette>)[field]

      return typeof color === 'string' && color.trim() !== '' && !/var\(/i.test(color)
    }

    return typeof value === 'object' && value !== null && fields.every(standalone) ? (value as BootPalette) : null
  } catch {
    return null
  }
}

async function loadWordmarkFont(): Promise<void> {
  if (typeof FontFace === 'undefined' || !document.fonts) {
    return
  }

  try {
    const face = new FontFace('Collapse', `url("${collapseBold}")`, { weight: '700', display: 'swap' })
    document.fonts.add(face)
    await face.load()
  } catch {
    // The font is optional: a failed request must leave the system lettering usable.
  }
}

// Keep this independent of the app graph: its JavaScript or CSS may have failed to load.
export function showBootFailure(error: unknown, reload: () => void = () => location.reload()): void {
  console.error('Hermes failed to load', error)

  const root = document.getElementById('root')

  // A transparent window (the pet, the wake indicator) is click-through or too small for a
  // notice, and would only paint it over the user's desktop: there the cause is logged alone.
  if (!root || root.hasChildNodes() || isTransparentWindow()) {
    return
  }

  // With the theme's snapshot the notice paints its own background as well as its ink, so the
  // pair holds whatever the app's stylesheet did or did not paint behind it. Without one it
  // inherits the page's colours.
  const palette = readBootPalette()
  const notice = document.createElement('main')
  notice.lang = 'en'
  Object.assign(notice.style, {
    ...(palette && {
      background: palette.background,
      color: palette.foreground,
      inset: '0',
      overflow: 'auto',
      position: 'fixed'
    }),
    alignItems: 'center',
    boxSizing: 'border-box',
    colorScheme: 'inherit',
    display: 'flex',
    flexDirection: 'column',
    font: '16px/1.5 system-ui, sans-serif',
    justifyContent: 'center',
    minHeight: '100vh',
    padding: '24px',
    textAlign: 'center'
  })

  const wordmark = document.createElement('p')
  wordmark.textContent = 'HERMES AGENT'
  wordmark.setAttribute('aria-hidden', 'true')
  Object.assign(wordmark.style, {
    color: palette?.accent ?? 'currentColor',
    fontFamily: 'Collapse, system-ui, sans-serif',
    fontSize: 'clamp(2.5rem, 9vw, 6rem)',
    fontWeight: '700',
    letterSpacing: '0.08em',
    lineHeight: '0.9',
    margin: '0 0 24px',
    textTransform: 'uppercase'
  })

  // English is intentional: i18n lives in the app graph that failed to load.
  const heading = document.createElement('h1')
  heading.textContent = 'Hermes couldn’t load'
  heading.style.cssText = 'color: inherit; font-size: 1.125rem; font-weight: 600; margin: 0'

  const message = document.createElement('p')
  message.textContent = 'Part of the app didn’t load; reloading usually fixes it.'
  message.style.cssText = 'font-size: 0.875rem; opacity: 0.7; margin: 16px 0; max-width: 36em'

  const button = document.createElement('button')
  button.type = 'button'
  button.textContent = 'Reload'
  Object.assign(button.style, {
    font: '500 0.875rem/1.25rem system-ui, sans-serif',
    color: palette?.accentForeground ?? 'inherit',
    background: palette?.accent ?? 'transparent',
    border: palette ? 'none' : '1px solid currentColor',
    borderRadius: '2.5px',
    padding: '8px 20px',
    cursor: 'pointer'
  })
  button.addEventListener('click', () => reload())

  const alert = document.createElement('div')
  alert.setAttribute('role', 'alert')
  alert.append(heading, message)
  notice.append(wordmark, alert, button)
  root.append(notice)
  void loadWordmarkFont()
}
