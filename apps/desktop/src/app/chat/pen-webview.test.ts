import { afterEach, describe, expect, it } from 'vitest'

import { destroyPenWebview, ensurePenWebview, penWebviewAlive } from './pen-webview'

afterEach(() => {
  destroyPenWebview()
})

describe('pen webview guest', () => {
  it('reuses one guest; only destroy kills it', () => {
    ensurePenWebview('https://app.pen.dev/new?embed')
    ensurePenWebview('https://app.pen.dev/new?embed')

    expect(penWebviewAlive()).toBe(true)
    expect(document.querySelectorAll('webview')).toHaveLength(1)

    destroyPenWebview()

    expect(penWebviewAlive()).toBe(false)
    expect(document.querySelectorAll('webview')).toHaveLength(0)
  })
})
