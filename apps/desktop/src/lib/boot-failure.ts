import { isTransparentWindow } from './transparent-windows'

// Keep this independent of the app graph: its JavaScript or CSS may have failed to load.
export function showBootFailure(error: unknown, reload: () => void = () => location.reload()): void {
  console.error('Hermes failed to load', error)

  const root = document.getElementById('root')

  // A transparent window (the pet, the wake indicator) is click-through or too small for a
  // notice, and would only paint it over the user's desktop: there the cause is logged alone.
  if (!root || root.hasChildNodes() || isTransparentWindow()) {
    return
  }

  const notice = document.createElement('main')
  notice.lang = 'en'
  Object.assign(notice.style, {
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

  // English is intentional: i18n lives in the app graph that failed to load.
  const heading = document.createElement('h1')
  heading.textContent = 'Hermes couldn’t load'
  heading.style.cssText = 'font-size: 1.5em; margin: 0'

  const message = document.createElement('p')
  message.textContent = 'Part of the app didn’t load; reloading usually fixes it.'
  message.style.cssText = 'margin: 16px 0; max-width: 36em'

  const button = document.createElement('button')
  button.type = 'button'
  button.textContent = 'Reload'
  button.style.cssText =
    'font: inherit; color: inherit; background: transparent; border: 1px solid currentColor; padding: 8px 16px; cursor: pointer'
  button.addEventListener('click', () => reload())

  const alert = document.createElement('div')
  alert.setAttribute('role', 'alert')
  alert.append(heading, message)
  notice.append(alert, button)
  root.append(notice)
}
