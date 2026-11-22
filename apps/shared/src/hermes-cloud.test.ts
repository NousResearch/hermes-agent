import { describe, expect, it } from 'vitest'

import { isHermesCloudUrl } from './hermes-cloud'

describe('isHermesCloudUrl', () => {
  it('recognises production and staging agent hosts, nothing else', () => {
    expect(isHermesCloudUrl('https://ares-3009.agents.nousresearch.com')).toBe(true)
    expect(isHermesCloudUrl('https://ares--app.agents.staging-nousresearch.com/')).toBe(true)
    expect(isHermesCloudUrl('https://agents.nousresearch.com.evil.example')).toBe(false)
    expect(isHermesCloudUrl('https://my-box.tailnet.ts.net')).toBe(false)
    expect(isHermesCloudUrl('not a url')).toBe(false)
    expect(isHermesCloudUrl(undefined)).toBe(false)
  })
})
