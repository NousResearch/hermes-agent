/**
 * Guest-page element picker for the Browser preview.
 *
 * `PICKER_SCRIPT` is injected with `webview.executeJavaScript`. Hovering
 * outlines the node under the pointer; clicking resolves the script's promise
 * with a structured snapshot that `PreviewPane` formats and inserts into the
 * composer. Escape, a second click on the toolbar button, or a navigation
 * cancels the pick — the last two through `CANCEL_PICKER_SCRIPT`.
 *
 * The snapshot is what a person would otherwise copy out of DevTools by hand:
 * page URL and title, tag, role, accessible name, CSS selector, XPath, the
 * attributes that identify the node, the computed colors and font, the
 * bounding box, the visible text, the surrounding block's text, and an HTML
 * excerpt.
 *
 * Two things are deliberately never captured: the value of any input (a
 * password field reports only that it is one), and `script` / `style` /
 * `noscript` / `template` bodies inside the HTML excerpt.
 */

export const PICKER_MAX_TEXT = 8_000
export const PICKER_MAX_HTML = 8_000
export const PICKER_MAX_ATTR = 240

export interface PickedElement {
  accessibleName?: string
  attributes?: Record<string, string>
  htmlExcerpt?: string
  nearbyText?: string
  pageTitle: string
  pageUrl: string
  rect?: { height: number; width: number; x: number; y: number }
  role?: string
  selector?: string
  style?: {
    backgroundColor?: string
    color?: string
    display?: string
    fontFamily?: string
    fontSize?: string
    fontWeight?: string
  }
  tagName: string
  text?: string
  xpath?: string
}

export type PickerResult = { element: PickedElement; status: 'selected' } | { status: 'cancelled' }

export function isPickerResult(value: unknown): value is PickerResult {
  if (!value || typeof value !== 'object') {
    return false
  }

  const result = value as { element?: unknown; status?: unknown }

  if (result.status === 'cancelled') {
    return true
  }

  if (result.status !== 'selected' || !result.element || typeof result.element !== 'object') {
    return false
  }

  const element = result.element as Partial<PickedElement>

  return typeof element.pageUrl === 'string' && typeof element.tagName === 'string'
}

function clipText(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}\n\n[truncated]` : value
}

/** Fence long enough that a ``` line inside the body cannot close the block. */
export function fenceFor(body: string, info = ''): { close: string; open: string } {
  const longest = (body.match(/`+/g) ?? []).reduce((max, run) => Math.max(max, run.length), 0)
  const ticks = '`'.repeat(Math.max(3, longest + 1))

  return { close: ticks, open: info ? `${ticks}${info}` : ticks }
}

export function formatPickedElement(picked: PickedElement): string {
  const lines = [
    '## Element',
    `URL: ${picked.pageUrl}`,
    `Title: ${picked.pageTitle || '(untitled)'}`,
    `Tag: ${picked.tagName}`
  ]

  const add = (label: string, value?: string) => {
    const trimmed = value?.trim()

    if (trimmed) {
      lines.push(`${label}: ${clipText(trimmed, PICKER_MAX_TEXT)}`)
    }
  }

  add('Role', picked.role)
  add('Accessible name', picked.accessibleName)
  add('Selector', picked.selector)
  add('XPath', picked.xpath)

  if (picked.attributes && Object.keys(picked.attributes).length > 0) {
    add(
      'Attributes',
      Object.entries(picked.attributes)
        .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
        .join(' ')
    )
  }

  add('Color', picked.style?.color)
  add('Background', picked.style?.backgroundColor)

  const font = [picked.style?.fontSize, picked.style?.fontFamily].filter(Boolean).join(' ')
  add('Font', font)
  add('Font weight', picked.style?.fontWeight)
  add('Display', picked.style?.display)

  if (picked.rect) {
    lines.push(
      `Rect: x=${Math.round(picked.rect.x)}, y=${Math.round(picked.rect.y)}, width=${Math.round(picked.rect.width)}, height=${Math.round(picked.rect.height)}`
    )
  }

  const block = (label: string, value: string | undefined, info = '') => {
    const trimmed = value?.trim()

    if (!trimmed) {
      return
    }

    const body = clipText(trimmed, PICKER_MAX_TEXT)
    const fence = fenceFor(body, info)
    lines.push('', `${label}:`, fence.open, body, fence.close)
  }

  block('Text', picked.text)
  block('Nearby context', picked.nearbyText)
  block('HTML excerpt', picked.htmlExcerpt, 'html')

  return lines.join('\n')
}

/**
 * Injected into the guest page, so it must stay self-contained: no imports, no
 * TypeScript, and ES5 syntax only — the page may run under any engine Electron
 * hands it, and the string is evaluated as-is.
 */
export const PICKER_SCRIPT = `(function () {
  var KEY = '__hermesWebElementPicker'
  var existing = window[KEY]
  if (existing && typeof existing.cancel === 'function') existing.cancel()

  var MAX_TEXT = ${PICKER_MAX_TEXT}
  var MAX_HTML = ${PICKER_MAX_HTML}
  var MAX_ATTR = ${PICKER_MAX_ATTR}

  function clip(value, max) {
    var text = String(value == null ? '' : value).replace(/\\s+/g, ' ').trim()
    return text.length > max ? text.slice(0, max) + '...' : text
  }

  function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(value)
    return String(value).replace(/[^a-zA-Z0-9_-]/g, '\\\\$&')
  }

  function visibleText(node) {
    if (node instanceof HTMLInputElement) {
      if (node.type.toLowerCase() === 'password') return '[masked password input]'
      return clip(node.getAttribute('aria-label') || node.getAttribute('placeholder') || node.name || node.type, MAX_TEXT)
    }
    if (node instanceof HTMLTextAreaElement) {
      return clip(node.getAttribute('aria-label') || node.getAttribute('placeholder') || node.name || 'textarea', MAX_TEXT)
    }
    return clip(node.innerText || node.textContent, MAX_TEXT)
  }

  function impliedRole(node) {
    var tag = node.tagName.toLowerCase()
    if (tag === 'button') return 'button'
    if (tag === 'a' && node.hasAttribute('href')) return 'link'
    if (tag === 'img') return 'img'
    if (tag === 'input') {
      var type = (node.getAttribute('type') || 'text').toLowerCase()
      if (type === 'checkbox') return 'checkbox'
      if (type === 'radio') return 'radio'
      if (type === 'range') return 'slider'
      if (type === 'button' || type === 'submit' || type === 'reset') return 'button'
      return 'textbox'
    }
    if (tag === 'textarea') return 'textbox'
    if (tag === 'select') return 'combobox'
    if (tag === 'nav') return 'navigation'
    if (tag === 'main') return 'main'
    if (tag === 'form') return 'form'
    if (/^h[1-6]$/.test(tag)) return 'heading'
    return ''
  }

  function accessibleName(node) {
    var labelled = node.getAttribute('aria-labelledby')
    if (labelled) {
      var joined = labelled
        .split(/\\s+/)
        .map(function (id) {
          var el = document.getElementById(id)
          return el ? el.textContent || '' : ''
        })
        .join(' ')
      var clipped = clip(joined, MAX_TEXT)
      if (clipped) return clipped
    }
    return clip(
      node.getAttribute('aria-label') ||
        node.getAttribute('alt') ||
        node.getAttribute('title') ||
        node.getAttribute('placeholder') ||
        visibleText(node),
      MAX_TEXT
    )
  }

  function attrs(node) {
    var out = {}
    Array.from(node.attributes).forEach(function (attr) {
      var name = attr.name.toLowerCase()
      var keep =
        name === 'id' ||
        name === 'class' ||
        name === 'href' ||
        name === 'src' ||
        name === 'alt' ||
        name === 'title' ||
        name === 'name' ||
        name === 'type' ||
        name === 'placeholder' ||
        name.indexOf('aria-') === 0
      if (keep && name !== 'value') out[name] = clip(attr.value, MAX_ATTR)
    })
    return out
  }

  function selectorOf(node) {
    if (node.id) return '#' + cssEscape(node.id)
    var parts = []
    var current = node
    while (current && current.nodeType === Node.ELEMENT_NODE && parts.length < 8) {
      var tag = current.tagName.toLowerCase()
      if (current.id) {
        parts.unshift(tag + '#' + cssEscape(current.id))
        break
      }
      var part = tag + Array.from(current.classList).filter(Boolean).slice(0, 2).map(function (cls) {
        return '.' + cssEscape(cls)
      }).join('')
      var parent = current.parentElement
      if (parent) {
        var same = Array.from(parent.children).filter(function (child) {
          return child instanceof Element && child.tagName === current.tagName
        })
        if (same.length > 1) part += ':nth-of-type(' + (same.indexOf(current) + 1) + ')'
      }
      parts.unshift(part)
      current = parent
    }
    return parts.join(' > ')
  }

  function xpathOf(node) {
    var parts = []
    var current = node
    while (current && current.nodeType === Node.ELEMENT_NODE && parts.length < 12) {
      var tag = current.tagName.toLowerCase()
      var parent = current.parentElement
      if (!parent) {
        parts.unshift('/' + tag)
        break
      }
      var index =
        Array.from(parent.children).filter(function (child) {
          return child instanceof Element && child.tagName === current.tagName
        }).indexOf(current) + 1
      parts.unshift(tag + '[' + index + ']')
      current = parent
    }
    return ('/' + parts.join('/')).replace(/^\\/\\//, '/')
  }

  function nearby(node) {
    var root = node.closest('article, section, main, form, li, tr, dialog') || node.parentElement || node
    return clip(root.innerText || root.textContent, MAX_TEXT)
  }

  function htmlOf(node) {
    var clone = node.cloneNode(true)
    if (!(clone instanceof Element)) return ''
    clone.querySelectorAll('script, style, noscript, template').forEach(function (el) {
      el.remove()
    })
    clone.querySelectorAll('input, textarea').forEach(function (el) {
      if (el instanceof HTMLInputElement) {
        el.removeAttribute('value')
        if (el.type.toLowerCase() === 'password') el.setAttribute('type', 'password')
      }
      if (el instanceof HTMLTextAreaElement) el.textContent = ''
    })
    return clip(clone.outerHTML, MAX_HTML)
  }

  function hex(channel) {
    return Math.max(0, Math.min(255, Math.round(channel))).toString(16).padStart(2, '0').toUpperCase()
  }

  function colorOf(raw) {
    var text = String(raw || '').trim()
    var match = /^rgba?\\(\\s*([0-9.]+)(?:,|\\s)+([0-9.]+)(?:,|\\s)+([0-9.]+)(?:\\s*[,/]\\s*([0-9.]+%?))?\\s*\\)$/i.exec(text)
    if (!match) return text
    var alpha = match[4] ? (String(match[4]).endsWith('%') ? Number(match[4].slice(0, -1)) / 100 : Number(match[4])) : 1
    if ([match[1], match[2], match[3], alpha].some(function (n) { return Number.isNaN(Number(n)) })) return text
    if (alpha <= 0) return 'transparent'
    return '#' + hex(Number(match[1])) + hex(Number(match[2])) + hex(Number(match[3]))
  }

  function styleOf(node) {
    var computed = window.getComputedStyle(node)
    var background = colorOf(computed.backgroundColor)
    var out = {
      color: colorOf(computed.color),
      display: computed.display,
      fontFamily: clip(computed.fontFamily, 160),
      fontSize: computed.fontSize,
      fontWeight: computed.fontWeight
    }
    if (background !== 'transparent') out.backgroundColor = background
    return out
  }

  function snapshot(node) {
    var box = node.getBoundingClientRect()
    return {
      pageUrl: location.href,
      pageTitle: document.title,
      tagName: node.tagName.toLowerCase(),
      role: node.getAttribute('role') || impliedRole(node) || undefined,
      accessibleName: accessibleName(node) || undefined,
      selector: selectorOf(node),
      xpath: xpathOf(node),
      text: visibleText(node) || undefined,
      nearbyText: nearby(node) || undefined,
      htmlExcerpt: htmlOf(node) || undefined,
      attributes: attrs(node),
      rect: { x: box.x, y: box.y, width: box.width, height: box.height },
      style: styleOf(node)
    }
  }

  var overlay = document.createElement('div')
  overlay.setAttribute('data-hermes-web-element-picker', 'overlay')
  Object.assign(overlay.style, {
    background: 'rgba(37, 99, 235, 0.12)',
    border: '2px solid #2563eb',
    borderRadius: '4px',
    boxShadow: '0 0 0 9999px rgba(15, 23, 42, 0.10)',
    boxSizing: 'border-box',
    display: 'none',
    left: '0',
    pointerEvents: 'none',
    position: 'fixed',
    top: '0',
    zIndex: '2147483647'
  })

  var tip = document.createElement('div')
  Object.assign(tip.style, {
    backdropFilter: 'blur(10px)',
    background: 'rgba(17, 24, 39, 0.92)',
    border: '1px solid rgba(255, 255, 255, 0.14)',
    borderRadius: '18px',
    boxShadow: '0 18px 38px rgba(15, 23, 42, 0.28)',
    boxSizing: 'border-box',
    color: '#f9fafb',
    display: 'none',
    font: '12px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
    left: '0',
    maxWidth: 'calc(100vw - 16px)',
    minWidth: '214px',
    padding: '12px 18px 14px',
    pointerEvents: 'none',
    position: 'fixed',
    top: '0',
    width: 'min(320px, calc(100vw - 16px))',
    zIndex: '2147483647'
  })

  document.documentElement.append(overlay, tip)

  var hover = null
  var settled = false
  var finish = null

  var PRESS_EVENTS = ['pointerdown', 'mousedown', 'mouseup', 'pointerup', 'contextmenu', 'dblclick']

  function teardown() {
    document.removeEventListener('mousemove', onMove, true)
    document.removeEventListener('click', onClick, true)
    document.removeEventListener('keydown', onKey, true)
    PRESS_EVENTS.forEach(function (name) {
      document.removeEventListener(name, onPress, true)
    })
    window.removeEventListener('scroll', onReflow, true)
    window.removeEventListener('resize', onReflow, true)
    overlay.remove()
    tip.remove()
    delete window[KEY]
    document.documentElement.style.cursor = ''
  }

  function placeTip(box) {
    var width = tip.offsetWidth || 240
    var height = tip.offsetHeight || 90
    var maxLeft = Math.max(8, window.innerWidth - width - 8)
    var maxTop = Math.max(8, window.innerHeight - height - 8)
    var left = Math.max(8, Math.min(maxLeft, box.left + box.width / 2 - width / 2))
    var below = box.bottom + 12
    var top = below + height <= window.innerHeight - 8 ? below : Math.max(8, box.top - height - 12)
    tip.style.left = left + 'px'
    tip.style.top = Math.max(8, Math.min(maxTop, top)) + 'px'
  }

  function paint(node) {
    var box = node.getBoundingClientRect()
    overlay.style.display = 'block'
    overlay.style.left = Math.max(0, box.left) + 'px'
    overlay.style.top = Math.max(0, box.top) + 'px'
    overlay.style.width = box.width + 'px'
    overlay.style.height = box.height + 'px'
    tip.style.display = 'block'
    tip.textContent = ''
    var title = document.createElement('div')
    title.textContent = node.tagName.toLowerCase() + '  ' + Math.round(box.width) + 'x' + Math.round(box.height)
    title.style.fontWeight = '800'
    tip.append(title)
    var name = accessibleName(node)
    if (name) {
      var line = document.createElement('div')
      line.textContent = name
      line.style.opacity = '0.8'
      line.style.marginTop = '4px'
      tip.append(line)
    }
    placeTip(box)
  }

  function onMove(event) {
    var target = event.target
    if (!(target instanceof Element)) return
    hover = target
    paint(target)
  }

  /* The outline is drawn in viewport coordinates, so anything that moves the
     page under the pointer leaves it pointing at empty space. */
  function onReflow() {
    if (hover && hover.isConnected) paint(hover)
  }

  /* A page that acts on mousedown (a drag start, a menu, a link that navigates
     itself) would run while the user is only aiming. The pick is the click, so
     everything before and around it is swallowed. */
  function onPress(event) {
    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()
  }

  function onClick(event) {
    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()
    if (!hover) {
      finish && finish({ status: 'cancelled' })
      return
    }
    finish && finish({ status: 'selected', element: snapshot(hover) })
  }

  function onKey(event) {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      finish && finish({ status: 'cancelled' })
    }
  }

  return new Promise(function (resolve) {
    finish = function (result) {
      if (settled) return
      settled = true
      teardown()
      resolve(result)
    }
    window[KEY] = { cancel: function () { finish({ status: 'cancelled' }) } }
    document.documentElement.style.cursor = 'crosshair'
    document.addEventListener('mousemove', onMove, true)
    document.addEventListener('click', onClick, true)
    document.addEventListener('keydown', onKey, true)
    PRESS_EVENTS.forEach(function (name) {
      document.addEventListener(name, onPress, true)
    })
    window.addEventListener('scroll', onReflow, true)
    window.addEventListener('resize', onReflow, true)
  })
})()`

export const CANCEL_PICKER_SCRIPT = `(() => {
  const picker = window.__hermesWebElementPicker
  if (picker && typeof picker.cancel === 'function') picker.cancel()
})()`
