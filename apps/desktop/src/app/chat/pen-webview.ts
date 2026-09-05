/** The editor guest lives on document.body. Hiding a chat must not destroy it —
 *  Pencil's MCP tools run in that page, visible or not. */

let webview: HTMLElement | null = null

const HIDDEN =
  'position:fixed;left:-10000px;top:0;width:4px;height:4px;border:0;visibility:hidden;pointer-events:none'

export function ensurePenWebview(url: string): HTMLElement {
  if (webview) {
    return webview
  }

  const node = document.createElement('webview')

  node.setAttribute('src', url)
  node.setAttribute('style', HIDDEN)
  document.body.append(node)
  webview = node

  return node
}

export function layoutPenWebview(host: HTMLElement | null): void {
  if (!webview) {
    return
  }

  if (!host) {
    webview.setAttribute('style', HIDDEN)

    return
  }

  const box = host.getBoundingClientRect()

  webview.setAttribute(
    'style',
    `position:fixed;left:${box.left}px;top:${box.top}px;width:${box.width}px;height:${box.height}px;border:0;visibility:visible`
  )
}

export function destroyPenWebview(): void {
  webview?.remove()
  webview = null
}

export function penWebviewAlive(): boolean {
  return Boolean(webview)
}
