import { describe, expect, it } from 'vitest'

import { renderedMessageHtml } from './message-clipboard-html'

function mount(html: string): HTMLElement {
  const host = document.createElement('div')

  host.innerHTML = html
  document.body.appendChild(host)

  return host
}

describe('renderedMessageHtml', () => {
  it('returns the rendered prose without reasoning, tool chrome or theme attributes', () => {
    const host = mount(`
      <div data-slot="aui_assistant-message-root">
        <div data-slot="aui_assistant-message-content">
          <div data-slot="aui_reasoning-text"><div class="aui-md prose"><p>secret thinking</p></div></div>
          <div class="aui-md prose text-foreground" data-x="1">
            <p class="prose-p" style="color: red">Hello <strong>world</strong> <a href="https://example.com" class="lnk">link</a></p>
            <div data-slot="code-card" class="group/code">
              <button>Copy code</button>
              <div data-slot="code-card-body"><pre><code><span class="tok">const</span> <span>a = 1</span></code></pre></div>
            </div>
            <table><tr><td colspan="2">cell</td></tr></table>
          </div>
        </div>
        <div data-slot="aui_msg-actions"><button id="copy">copy</button></div>
      </div>
    `)

    const html = renderedMessageHtml(host.querySelector('#copy'))

    expect(html).not.toBeNull()
    expect(html).toContain('<p>Hello <strong>world</strong> <a href="https://example.com">link</a></p>')
    expect(html).toContain('<pre><code>const a = 1</code></pre>')
    expect(html).toContain('<td colspan="2">cell</td>')
    expect(html).not.toContain('secret thinking')
    expect(html).not.toContain('Copy code')
    expect(html).not.toContain('class=')
    expect(html).not.toContain('style=')
    expect(html).not.toContain('data-')
  })

  it('returns null outside a message or when the message has no prose', () => {
    const host = mount(`
      <div data-slot="aui_assistant-message-root">
        <div data-slot="aui_assistant-message-content"><div data-slot="aui_turn-activity">working…</div></div>
        <button id="copy">copy</button>
      </div>
      <button id="stray">stray</button>
    `)

    expect(renderedMessageHtml(host.querySelector('#copy'))).toBeNull()
    expect(renderedMessageHtml(host.querySelector('#stray'))).toBeNull()
    expect(renderedMessageHtml(null)).toBeNull()
  })
})
