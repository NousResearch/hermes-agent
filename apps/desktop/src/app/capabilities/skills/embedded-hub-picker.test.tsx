/**
 * #135315: the Capabilities → Skills hub iframe must keep the same sandbox
 * and clipboard posture the Bot Mode Skills Hub got for #91612. Without
 * allow-popups, every window.open / target=_blank on the hub page is killed
 * inside the sandboxed frame — external links (docs, GitHub, Discord) never
 * reach the main-process window-open delegation, so they silently do
 * nothing.
 */
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { EmbeddedHubPicker } from './embedded-hub-picker'

const HUB_PICKER_URL = 'https://hermes-agent.nousresearch.com/docs/skills?embed=picker'

afterEach(cleanup)

describe('EmbeddedHubPicker', () => {
  it('pins the hub frame to the required sandbox and clipboard posture (#135315)', () => {
    const { container } = render(<EmbeddedHubPicker installedNames={new Set()} />)

    const frame = container.querySelector('iframe')
    expect(frame).toBeTruthy()
    expect(frame?.getAttribute('src')).toBe(HUB_PICKER_URL)
    // Same-origin (the hub's own routing), scripts, and popups — external
    // links then reach the OS browser via the main-process window-open
    // delegation, never a popup window.
    expect(frame?.getAttribute('sandbox')).toBe(
      'allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox'
    )
    // The Copy controls write to the clipboard; the session permission
    // handlers grant clipboard-sanitized-write only to the hub origins.
    expect(frame?.getAttribute('allow')).toBe('clipboard-write')
  })
})
