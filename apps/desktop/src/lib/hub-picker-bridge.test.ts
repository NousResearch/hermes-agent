import { describe, expect, it } from 'vitest'

import { bridgedHubOpenUrl, HUB_PICKER_OPEN_MESSAGE } from './hub-picker-bridge'

const URL_UNDER_TEST = 'https://github.com/NousResearch/hermes-agent'

describe('bridgedHubOpenUrl', () => {
  it('accepts the bridge message for an openable scheme', () => {
    expect(bridgedHubOpenUrl({ type: HUB_PICKER_OPEN_MESSAGE, url: URL_UNDER_TEST })).toBe(URL_UNDER_TEST)
    expect(bridgedHubOpenUrl({ type: HUB_PICKER_OPEN_MESSAGE, url: 'mailto:plugins@nousresearch.com' })).toBe(
      'mailto:plugins@nousresearch.com'
    )
  })

  it('ignores anything that is not a bridge message', () => {
    expect(bridgedHubOpenUrl(null)).toBeNull()
    expect(bridgedHubOpenUrl(undefined)).toBeNull()
    expect(bridgedHubOpenUrl('nope')).toBeNull()
    expect(bridgedHubOpenUrl({ type: 'hermes-plugin-pick', url: URL_UNDER_TEST })).toBeNull()
    expect(bridgedHubOpenUrl({ type: HUB_PICKER_OPEN_MESSAGE })).toBeNull()
    expect(bridgedHubOpenUrl({ type: HUB_PICKER_OPEN_MESSAGE, url: '' })).toBeNull()
    expect(bridgedHubOpenUrl({ type: HUB_PICKER_OPEN_MESSAGE, url: 42 })).toBeNull()
  })

  it('refuses schemes the audited external opener never handles', () => {
    for (const url of [
      'javascript:alert(1)',
      'file:///etc/passwd',
      'data:text/html,<script>alert(1)</script>',
      'hermes://plugin/install?catalog=x',
      'chrome://settings',
      '/docs/plugins/accent-picker'
    ]) {
      expect(bridgedHubOpenUrl({ type: HUB_PICKER_OPEN_MESSAGE, url })).toBeNull()
    }
  })
})
