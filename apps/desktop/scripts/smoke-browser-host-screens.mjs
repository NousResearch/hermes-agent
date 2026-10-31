// The startup screens a working Webapp never shows, titled from the English catalog the renderer
// renders (the smoke's Chromium runs in English).
import { enBoot } from '../src/i18n/en_boot.ts'

const { failure } = enBoot.boot

const STARTUP_FAILURE_SCREENS = [
  // A gateway that is down also ends here, so HERMES_BROWSER_ALLOW_GATEWAY_FAILURE tolerates it.
  {
    allowedWithGatewayFailure: true,
    name: 'boot-failure screen',
    titles: [failure.title, failure.remoteTitle, failure.cloudDownTitle]
  },
  { allowedWithGatewayFailure: false, name: 'sign-in screen', titles: [failure.webappTitle] }
]

export const STARTUP_FAILURE_TITLES = STARTUP_FAILURE_SCREENS.flatMap(screen => screen.titles)

/** Page-side expression: the startup failure titles in the page's visible text. */
export const shownStartupTitlesExpression = `${JSON.stringify(STARTUP_FAILURE_TITLES)}.filter(title =>
  (document.body.innerText || '').includes(title))`

/** One failure line per startup failure screen whose title the page shows. */
export function startupScreenFailures(shownTitles, { allowGatewayFailure }) {
  return STARTUP_FAILURE_SCREENS.flatMap(screen => {
    const title = screen.titles.find(candidate => shownTitles.includes(candidate))

    if (!title || (allowGatewayFailure && screen.allowedWithGatewayFailure)) return []

    return [`Webapp shows the ${screen.name} (${JSON.stringify(title)})`]
  })
}
