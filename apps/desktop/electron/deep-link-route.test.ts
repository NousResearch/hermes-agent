import { expect, it } from 'vitest'

import { routeDeepLink } from './deep-link-route'

it.each([
  ['copilot-key', 'start', 'quick-entry'],
  ['copilot-key', 'stop', 'ignore'],
  ['copilot-key', '', 'ignore'],
  ['copilot-key', 'toggle', 'ignore'],
  ['blueprint', 'morning-brief', 'renderer'],
  ['', '', 'renderer']
] as const)('%s/%s → %s', (host: string, pathname: string, expected: string): void => {
  expect(routeDeepLink(host, pathname)).toBe(expected)
})
