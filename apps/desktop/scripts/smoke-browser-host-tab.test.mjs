import { describe, expect, it } from 'vitest'

import { documentAddress, ViewportTab } from './smoke-browser-host-tab.mjs'

const LAUNCH = 'http://127.0.0.1:9120/webapp/launch#ticket=' + 't'.repeat(43)

/** A CDP client whose navigations load at once; it records what the smoke asked of the tab. */
function fakeClient() {
  const calls = []
  const waiting = new Map()

  return {
    calls,
    async send(method, params = {}) {
      calls.push([method, params])

      if (method === 'Page.navigate') {
        for (const event of ['Page.domContentEventFired', 'Page.loadEventFired']) {
          waiting.get(event)?.()
          waiting.delete(event)
        }
      }

      return {}
    },
    waitFor(event) {
      return new Promise(resolve => waiting.set(event, resolve))
    }
  }
}

const navigations = calls => calls.filter(([method]) => method === 'Page.navigate').map(([, params]) => params.url)

describe('browser smoke viewport tab', () => {
  // The smoke's input URL may be the one-use launch link `hermes webapp --no-open` prints, and only
  // the tab that spent it keeps the grant it bought. Every viewport therefore loads in one tab: the
  // first opens the input URL, and each later one reopens the address the first one landed on.
  it('spends a launch link once and reopens the Webapp it landed on for every later viewport', async () => {
    const client = fakeClient()
    const tab = new ViewportTab(client, LAUNCH)
    const loads = []

    loads.push(await tab.open({ height: 390, width: 844, mobile: true }, address => loads.push(['before', address])))
    await tab.settle(async expression =>
      expression === 'location.href' ? 'http://127.0.0.1:9120/?profile=work#/' : null
    )
    loads.push(await tab.open({ height: 800, width: 1280 }, address => loads.push(['before', address])))
    loads.push(await tab.open({ height: 844, width: 390 }, address => loads.push(['before', address])))

    expect(navigations(client.calls)).toEqual([
      LAUNCH,
      'about:blank',
      'http://127.0.0.1:9120/?profile=work',
      'about:blank',
      'http://127.0.0.1:9120/?profile=work'
    ])
    // HTTP responses never carry the fragment: each load's status is read for the address without it.
    expect(loads).toEqual([
      ['before', 'http://127.0.0.1:9120/webapp/launch'],
      'http://127.0.0.1:9120/webapp/launch',
      ['before', 'http://127.0.0.1:9120/?profile=work'],
      'http://127.0.0.1:9120/?profile=work',
      ['before', 'http://127.0.0.1:9120/?profile=work'],
      'http://127.0.0.1:9120/?profile=work'
    ])
  })

  it('leaves the previous page before a later viewport starts counting its errors', async () => {
    const client = fakeClient()
    const tab = new ViewportTab(client, 'http://127.0.0.1:9120/')
    await tab.open({ height: 390, width: 844 }, () => {})
    await tab.settle(async () => 'http://127.0.0.1:9120/#/')

    let callsBeforeLoad = -1
    await tab.open({ height: 800, width: 1280 }, () => {
      callsBeforeLoad = client.calls.length
    })

    const blank = client.calls.findIndex(
      ([method, params]) => method === 'Page.navigate' && params.url === 'about:blank'
    )
    expect(blank).toBeGreaterThanOrEqual(0)
    expect(callsBeforeLoad).toBeGreaterThan(blank)
  })

  it('sizes the tab for each viewport and clears the mobile browser-controls override', async () => {
    const client = fakeClient()
    const tab = new ViewportTab(client, 'http://127.0.0.1:9120/')
    await tab.open({ height: 800, width: 1280 }, () => {})

    expect(client.calls).toContainEqual([
      'Emulation.setDeviceMetricsOverride',
      { deviceScaleFactor: 1, height: 800, mobile: false, width: 1280 }
    ])
    expect(client.calls).toContainEqual(['Emulation.setSmallViewportHeightDifferenceOverride', { difference: 0 }])
  })

  it('reads a document address without its fragment', () => {
    expect(documentAddress(LAUNCH)).toBe('http://127.0.0.1:9120/webapp/launch')
  })
})
