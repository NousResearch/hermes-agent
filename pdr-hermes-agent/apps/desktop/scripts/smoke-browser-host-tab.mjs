// The browser smoke checks every viewport in one tab, for the launch link's sake.

/** *href* without its fragment: the address the document is requested from. A launch link's ticket
 *  rides in the fragment, which navigation keeps and HTTP responses never contain. */
export function documentAddress(href) {
  const address = new URL(href)
  address.hash = ''

  return address.href
}

/**
 * Loads each smoke viewport in one browser tab. The smoke's input URL may be the one-use launch link
 * `hermes webapp --no-open` prints: loading it spends the ticket, and only the tab that spent it keeps
 * the grant it bought (the Webapp keeps it in that tab's sessionStorage). So the first viewport opens
 * the input URL, and every later one reopens, in the same tab, the address the first one landed on.
 */
export class ViewportTab {
  constructor(client, url) {
    this.client = client
    this.url = url
    this.landed = null
  }

  /**
   * Sizes the tab for *viewport* and loads its page, after leaving the previous viewport's page for
   * about:blank so that page's teardown is over before *beforeLoad* runs. *beforeLoad* gets the
   * address this load requests, the one whose HTTP status the checks read. Resolves with it once the
   * page's DOM is ready.
   */
  async open(viewport, beforeLoad) {
    const url = this.landed ?? this.url

    if (this.landed) {
      const blank = this.client.waitFor('Page.loadEventFired', 30_000)
      await this.client.send('Page.navigate', { url: 'about:blank' })
      await blank
    }

    beforeLoad(documentAddress(url))
    await this.client.send('Emulation.setDeviceMetricsOverride', {
      width: viewport.width,
      height: viewport.height,
      deviceScaleFactor: 1,
      mobile: viewport.mobile ?? false
    })
    // A mobile viewport's browser-controls override outlives its page in this tab.
    await this.client.send('Emulation.setSmallViewportHeightDifferenceOverride', { difference: 0 })
    const domReady = this.client.waitFor('Page.domContentEventFired', 30_000)
    await this.client.send('Page.navigate', { url })
    await domReady

    return documentAddress(url)
  }

  /** Remembers where the first viewport's page landed: the Webapp itself, for a launch link. */
  async settle(evaluate) {
    this.landed ??= documentAddress(await evaluate('location.href'))
  }
}
