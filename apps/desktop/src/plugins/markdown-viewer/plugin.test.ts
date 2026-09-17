import { afterEach, describe, expect, it } from 'vitest'

async function loadPlugin() {
  // The bundled source intentionally remains plain ESM so it can also be
  // distributed through Hermes's uncompiled runtime-plugin door.
  // @ts-expect-error -- TypeScript does not ingest the intentional .js source.
  return import('./plugin.js')
}

afterEach(() => {
  document.body.replaceChildren()
})

describe('Markdown Viewer transcript targets', () => {
  it('opens a complete Markdown media path', async () => {
    const { markdownPathFromClickTarget } = await loadPlugin()
    const anchor = document.createElement('a')
    anchor.href = `#media:${encodeURIComponent('/Users/example/Team Documents/docs/guide.md')}`

    expect(markdownPathFromClickTarget(anchor)).toBe('/Users/example/Team Documents/docs/guide.md')
  })

  it('recovers a Markdown path split by an older desktop renderer', async () => {
    const { markdownPathFromClickTarget } = await loadPlugin()
    const wrapper = document.createElement('div')
    const anchor = document.createElement('a')
    anchor.setAttribute('href', '#media:%2FUsers%2Fexample%2FTeam')
    wrapper.append(anchor, ' Documents/docs/guide.md')

    expect(markdownPathFromClickTarget(anchor)).toBe('/Users/example/Team Documents/docs/guide.md')
  })

  it('uses the attachment preview control without hijacking Download', async () => {
    const { markdownPathFromClickTarget } = await loadPlugin()
    const card = document.createElement('div')
    const title = document.createElement('span')
    const download = document.createElement('button')
    const preview = document.createElement('button')
    title.title = '/Users/example/Team Documents/docs/guide.md'
    download.title = 'Download'
    card.append(title, download, preview)

    expect(markdownPathFromClickTarget(download)).toBeNull()
    expect(markdownPathFromClickTarget(preview)).toBe('/Users/example/Team Documents/docs/guide.md')
  })
})
