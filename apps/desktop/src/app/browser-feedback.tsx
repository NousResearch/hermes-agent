import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Textarea } from '@/components/ui/textarea'
import { notify, notifyError } from '@/store/notifications'

export interface PickedBrowserElement {
  attributes: Record<string, string>
  className: string
  id: string
  outerHtml: string
  role: string
  selector: string
  tagName: string
  text: string
  url: string
  xpath: string
  rect: {
    height: number
    width: number
    x: number
    y: number
  }
}

interface BrowserFeedbackWindowProps {
  minimized: boolean
  onClose: () => void
  onInsertPrompt: (prompt: string) => void
  onMinimizedChange: (minimized: boolean) => void
  open: boolean
}

type BrowserFeedbackWebview = HTMLElement & {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  executeJavaScript?: <T = unknown>(code: string, userGesture?: boolean) => Promise<T>
  getURL?: () => string
  goBack?: () => void
  goForward?: () => void
  isLoading?: () => boolean
  loadURL?: (url: string, options?: { userAgent?: string }) => Promise<void>
  reload?: () => void
  setUserAgent?: (userAgent: string) => void
  stop?: () => void
}

type BrowserViewportMode = 'desktop' | 'mobile'

interface BrowserViewportPreset {
  height: number | null
  label: string
  mode: BrowserViewportMode
  value: string
  width: number | null
}

interface PickerConsoleResult {
  cancelled?: boolean
  element?: PickedBrowserElement | null
}

const BLANK_URL = 'about:blank'
const PICKER_RESULT_PREFIX = '__HERMES_BROWSER_PICK__'
const MOBILE_USER_AGENT =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1'

const VIEWPORT_PRESETS: BrowserViewportPreset[] = [
  { height: null, label: 'Responsive desktop', mode: 'desktop', value: 'desktop-responsive', width: null },
  { height: 900, label: 'Desktop 1440 × 900', mode: 'desktop', value: 'desktop-1440x900', width: 1440 },
  { height: 1080, label: 'Desktop 1920 × 1080', mode: 'desktop', value: 'desktop-1920x1080', width: 1920 },
  { height: 667, label: 'iPhone SE 375 × 667', mode: 'mobile', value: 'mobile-375x667', width: 375 },
  { height: 844, label: 'iPhone 13/14 390 × 844', mode: 'mobile', value: 'mobile-390x844', width: 390 },
  { height: 932, label: 'iPhone 15 Pro Max 430 × 932', mode: 'mobile', value: 'mobile-430x932', width: 430 },
  { height: 915, label: 'Pixel 7/8 412 × 915', mode: 'mobile', value: 'mobile-412x915', width: 412 },
  { height: 1024, label: 'iPad 768 × 1024', mode: 'mobile', value: 'mobile-768x1024', width: 768 }
]

function viewportPresetFor(value: string): BrowserViewportPreset {
  return VIEWPORT_PRESETS.find(preset => preset.value === value) ?? VIEWPORT_PRESETS[0]
}

function userAgentForViewport(mode: BrowserViewportMode): string {
  return mode === 'mobile' ? MOBILE_USER_AGENT : window.navigator.userAgent
}
const PICKER_SCRIPT = String.raw`
(() => {
  const resultPrefix = '__HERMES_BROWSER_PICK__';
  const previous = window.__hermesElementPickerCleanup;
  if (typeof previous === 'function') previous();

  const overlayRoot = document.createElement('div');
  overlayRoot.style.cssText = [
    'position: fixed',
    'inset: 0',
    'z-index: 2147483647',
    'pointer-events: none',
    'contain: layout style paint'
  ].join(';');

  const makeBox = (background, border) => {
    const box = document.createElement('div');
    box.style.cssText = [
      'position: fixed',
      'left: 0',
      'top: 0',
      'width: 0',
      'height: 0',
      'pointer-events: none',
      'box-sizing: border-box',
      'display: none',
      'background: ' + background,
      'border: 1px solid ' + border,
      'transition: left 45ms linear, top 45ms linear, width 45ms linear, height 45ms linear'
    ].join(';');
    return box;
  };

  // Chrome DevTools-like box model colors: margin/orange, border/yellow,
  // padding/green, content/blue. Each layer is separate so the selected
  // element is visibly outlined even on complex pages.
  const marginBox = makeBox('rgba(246, 178, 107, .28)', 'rgba(245, 158, 11, .95)');
  const borderBox = makeBox('rgba(250, 204, 21, .30)', 'rgba(234, 179, 8, .95)');
  const paddingBox = makeBox('rgba(134, 239, 172, .32)', 'rgba(34, 197, 94, .95)');
  const contentBox = makeBox('rgba(56, 189, 248, .34)', 'rgba(14, 165, 233, .98)');

  const label = document.createElement('div');
  label.textContent = 'Hermes picker active: hover and click an element';
  label.style.cssText = [
    'position: fixed',
    'z-index: 2147483647',
    'left: 12px',
    'top: 12px',
    'pointer-events: none',
    'border-radius: 4px',
    'background: #0f172a',
    'color: white',
    'font: 12px/1.4 system-ui, sans-serif',
    'padding: 5px 8px',
    'max-width: min(420px, calc(100vw - 24px))',
    'white-space: nowrap',
    'overflow: hidden',
    'text-overflow: ellipsis',
    'box-shadow: 0 8px 24px rgba(0,0,0,.25)',
    'border: 1px solid rgba(148, 163, 184, .35)'
  ].join(';');

  overlayRoot.append(marginBox, borderBox, paddingBox, contentBox, label);
  (document.body || document.documentElement).append(overlayRoot);

  const cssEscape = value => {
    if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(value);
    return String(value).replace(/[^a-zA-Z0-9_-]/g, char => '\\' + char);
  };

  const selectorFor = element => {
    if (!(element instanceof Element)) return '';
    if (element.id) return element.tagName.toLowerCase() + '#' + cssEscape(element.id);

    const parts = [];
    let node = element;
    while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 5) {
      const tag = node.tagName.toLowerCase();
      const classes = Array.from(node.classList || []).slice(0, 3).map(cls => '.' + cssEscape(cls)).join('');
      const parent = node.parentElement;
      let nth = '';
      if (parent) {
        const sameTag = Array.from(parent.children).filter(child => child.tagName === node.tagName);
        if (sameTag.length > 1) nth = ':nth-of-type(' + (sameTag.indexOf(node) + 1) + ')';
      }
      parts.unshift(tag + classes + nth);
      node = parent;
    }
    return parts.join(' > ');
  };

  const xpathFor = element => {
    if (!(element instanceof Element)) return '';
    if (element.id) return '//*[@id="' + element.id.replace(/"/g, '\\"') + '"]';
    const parts = [];
    let node = element;
    while (node && node.nodeType === Node.ELEMENT_NODE) {
      let index = 1;
      let sibling = node.previousElementSibling;
      while (sibling) {
        if (sibling.tagName === node.tagName) index += 1;
        sibling = sibling.previousElementSibling;
      }
      parts.unshift(node.tagName.toLowerCase() + '[' + index + ']');
      node = node.parentElement;
    }
    return '/' + parts.join('/');
  };

  const dataFor = element => {
    const rect = element.getBoundingClientRect();
    const attrs = {};
    for (const attr of Array.from(element.attributes || [])) attrs[attr.name] = attr.value;
    return {
      attributes: attrs,
      className: element.className ? String(element.className) : '',
      id: element.id || '',
      outerHtml: (element.outerHTML || '').slice(0, 2000),
      rect: {
        height: Math.round(rect.height),
        width: Math.round(rect.width),
        x: Math.round(rect.x),
        y: Math.round(rect.y)
      },
      role: element.getAttribute('role') || '',
      selector: selectorFor(element),
      tagName: element.tagName.toLowerCase(),
      text: (element.innerText || element.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 500),
      url: location.href,
      xpath: xpathFor(element)
    };
  };

  const emit = payload => {
    console.info(resultPrefix + JSON.stringify(payload));
  };

  const px = value => {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : 0;
  };

  const placeBox = (box, left, top, width, height) => {
    if (width <= 0 || height <= 0) {
      box.style.display = 'none';
      return;
    }

    box.style.display = 'block';
    box.style.left = Math.max(0, left) + 'px';
    box.style.top = Math.max(0, top) + 'px';
    box.style.width = Math.max(1, width) + 'px';
    box.style.height = Math.max(1, height) + 'px';
  };

  const elementFromEvent = event => {
    if (typeof document.elementFromPoint === 'function' && typeof event.clientX === 'number') {
      const hit = document.elementFromPoint(event.clientX, event.clientY);
      if (hit instanceof Element && !overlayRoot.contains(hit)) return hit;
    }

    return event.target instanceof Element && !overlayRoot.contains(event.target) ? event.target : null;
  };

  const move = event => {
    const element = elementFromEvent(event);
    if (!(element instanceof Element)) return;
    const rect = element.getBoundingClientRect();
    const styles = getComputedStyle(element);
    const marginTop = px(styles.marginTop);
    const marginRight = px(styles.marginRight);
    const marginBottom = px(styles.marginBottom);
    const marginLeft = px(styles.marginLeft);
    const borderTop = px(styles.borderTopWidth);
    const borderRight = px(styles.borderRightWidth);
    const borderBottom = px(styles.borderBottomWidth);
    const borderLeft = px(styles.borderLeftWidth);
    const paddingTop = px(styles.paddingTop);
    const paddingRight = px(styles.paddingRight);
    const paddingBottom = px(styles.paddingBottom);
    const paddingLeft = px(styles.paddingLeft);

    placeBox(marginBox, rect.left - marginLeft, rect.top - marginTop, rect.width + marginLeft + marginRight, rect.height + marginTop + marginBottom);
    placeBox(borderBox, rect.left, rect.top, rect.width, rect.height);
    placeBox(paddingBox, rect.left + borderLeft, rect.top + borderTop, rect.width - borderLeft - borderRight, rect.height - borderTop - borderBottom);
    placeBox(
      contentBox,
      rect.left + borderLeft + paddingLeft,
      rect.top + borderTop + paddingTop,
      rect.width - borderLeft - borderRight - paddingLeft - paddingRight,
      rect.height - borderTop - borderBottom - paddingTop - paddingBottom
    );
    const name = element.tagName.toLowerCase() + (element.id ? '#' + element.id : '') +
      (element.classList && element.classList.length ? '.' + Array.from(element.classList).slice(0, 2).join('.') : '');
    const width = Math.round(rect.width);
    const height = Math.round(rect.height);
    const margin = [styles.marginTop, styles.marginRight, styles.marginBottom, styles.marginLeft].join(' ');
    const padding = [styles.paddingTop, styles.paddingRight, styles.paddingBottom, styles.paddingLeft].join(' ');
    label.textContent = name + ' · ' + width + '×' + height + 'px · margin ' + margin + ' · padding ' + padding;
    const labelX = Math.min(window.innerWidth - 12, Math.max(12, rect.left));
    const belowY = rect.bottom + 8;
    const aboveY = rect.top - 34;
    label.style.left = labelX + 'px';
    label.style.top = (belowY + 34 < window.innerHeight ? belowY : Math.max(12, aboveY)) + 'px';
  };

  const cleanup = () => {
    document.removeEventListener('mousemove', move, true);
    window.removeEventListener('mousemove', move, true);
    document.removeEventListener('pointermove', move, true);
    window.removeEventListener('pointermove', move, true);
    document.removeEventListener('pointerover', move, true);
    window.removeEventListener('pointerover', move, true);
    document.removeEventListener('pointerdown', pick, true);
    window.removeEventListener('pointerdown', pick, true);
    document.removeEventListener('click', pick, true);
    window.removeEventListener('click', pick, true);
    document.removeEventListener('keydown', cancel, true);
    window.removeEventListener('keydown', cancel, true);
    overlayRoot.remove();
    delete window.__hermesElementPickerCleanup;
  };

  const stop = event => {
    event.preventDefault();
    event.stopPropagation();
    if (typeof event.stopImmediatePropagation === 'function') event.stopImmediatePropagation();
  };

  const pick = event => {
    stop(event);
    const element = elementFromEvent(event);
    cleanup();
    emit({ cancelled: false, element: element instanceof Element ? dataFor(element) : null });
  };

  const cancel = event => {
    if (event.key !== 'Escape') return;
    stop(event);
    cleanup();
    emit({ cancelled: true });
  };

  window.__hermesElementPickerCleanup = cleanup;
  document.addEventListener('mousemove', move, true);
  window.addEventListener('mousemove', move, true);
  document.addEventListener('pointermove', move, true);
  window.addEventListener('pointermove', move, true);
  document.addEventListener('pointerover', move, true);
  window.addEventListener('pointerover', move, true);
  document.addEventListener('pointerdown', pick, true);
  window.addEventListener('pointerdown', pick, true);
  document.addEventListener('click', pick, true);
  window.addEventListener('click', pick, true);
  document.addEventListener('keydown', cancel, true);
  window.addEventListener('keydown', cancel, true);

  return true;
})()
`

export function normalizeBrowserFeedbackUrl(input: string): string {
  const trimmed = input.trim()

  if (!trimmed) {
    return BLANK_URL
  }

  if (/^[a-z][a-z0-9+.-]*:/i.test(trimmed)) {
    return trimmed
  }

  return `https://${trimmed}`
}

function addressForUrl(url: string): string {
  return url === BLANK_URL ? '' : url
}

function formatAttributes(attributes: Record<string, string>): string {
  return Object.entries(attributes)
    .slice(0, 20)
    .map(([key, value]) => `${key}="${value}"`)
    .join(' ')
}

export function buildElementChangePrompt(element: PickedBrowserElement, comment: string): string {
  const attrs = formatAttributes(element.attributes)
  const text = element.text ? `text: ${element.text}` : 'text: —'
  const html = element.outerHtml ? `outerHTML: ${element.outerHtml}` : 'outerHTML: —'

  return [
    'Visual change request from Hermes Desktop browser view.',
    '',
    `URL: ${element.url}`,
    '',
    'Selected element:',
    `- tag: ${element.tagName}`,
    `- id: ${element.id || '—'}`,
    `- classes: ${element.className || '—'}`,
    `- role: ${element.role || '—'}`,
    `- selector: ${element.selector || '—'}`,
    `- xpath: ${element.xpath || '—'}`,
    `- rect: x=${element.rect.x}, y=${element.rect.y}, width=${element.rect.width}, height=${element.rect.height}`,
    `- attributes: ${attrs || '—'}`,
    `- ${text}`,
    `- ${html}`,
    '',
    'User comment:',
    comment.trim() || 'Please inspect this selected element and suggest or implement the appropriate change.'
  ].join('\n')
}

export function BrowserFeedbackWindow({
  minimized,
  onClose,
  onInsertPrompt,
  onMinimizedChange,
  open
}: BrowserFeedbackWindowProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const webviewRef = useRef<BrowserFeedbackWebview | null>(null)
  const pickerTimeoutRef = useRef<number | null>(null)
  const [address, setAddress] = useState('')
  const [currentUrl, setCurrentUrl] = useState(BLANK_URL)
  const [comment, setComment] = useState('')
  const [selectedElement, setSelectedElement] = useState<PickedBrowserElement | null>(null)
  const [picking, setPicking] = useState(false)
  const [loading, setLoading] = useState(true)
  const [ready, setReady] = useState(false)
  const [canGoBack, setCanGoBack] = useState(false)
  const [canGoForward, setCanGoForward] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [viewportValue, setViewportValue] = useState(VIEWPORT_PRESETS[0].value)
  const viewportPreset = viewportPresetFor(viewportValue)
  const viewportStyle: CSSProperties = viewportPreset.width
    ? {
        height: `${viewportPreset.height}px`,
        maxHeight: '100%',
        maxWidth: '100%',
        width: `${viewportPreset.width}px`
      }
    : { height: '100%', width: '100%' }
  const promptPreview = useMemo(
    () => (selectedElement ? buildElementChangePrompt(selectedElement, comment) : ''),
    [comment, selectedElement]
  )

  const clearPickerTimeout = () => {
    if (pickerTimeoutRef.current !== null) {
      window.clearTimeout(pickerTimeoutRef.current)
      pickerTimeoutRef.current = null
    }
  }

  const completePicker = (element: PickedBrowserElement | null) => {
    clearPickerTimeout()
    setPicking(false)

    if (element) {
      setSelectedElement(element)
      notify({ kind: 'success', message: element.selector || element.tagName, title: 'Element selected' })
    }
  }

  useEffect(() => {
    if (!open) {
      return
    }

    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    webviewRef.current = null
    setReady(false)
    setLoading(true)
    setLoadError(null)
    setCanGoBack(false)
    setCanGoForward(false)

    const webview = document.createElement('webview') as BrowserFeedbackWebview
    webview.className = 'flex h-full w-full flex-1 bg-background'
    webview.setAttribute('partition', 'persist:hermes-browser-feedback')
    webview.setAttribute('src', currentUrl)
    webview.setAttribute('useragent', userAgentForViewport(viewportPreset.mode))
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const syncNavigationState = (nextUrl?: string) => {
      const resolvedUrl = nextUrl || webview.getURL?.()

      if (resolvedUrl) {
        setCurrentUrl(resolvedUrl)
        setAddress(addressForUrl(resolvedUrl))
      }

      setCanGoBack(Boolean(webview.canGoBack?.()))
      setCanGoForward(Boolean(webview.canGoForward?.()))
      setLoading(Boolean(webview.isLoading?.()))
    }

    const onNavigate = (event: Event) => {
      const detail = event as Event & { url?: string }
      setLoadError(null)
      syncNavigationState(detail.url)
    }
    const onDomReady = () => {
      setReady(true)
      syncNavigationState()
    }
    const onStart = () => {
      clearPickerTimeout()
      setPicking(false)
      setLoading(true)
      setLoadError(null)
      setSelectedElement(null)
    }
    const onStop = () => {
      setLoading(false)
      syncNavigationState()
    }
    const onFail = (event: Event) => {
      const detail = event as Event & { errorCode?: number; errorDescription?: string; validatedURL?: string }

      if (detail.errorCode === -3) {
        return
      }

      setLoadError(detail.errorDescription || detail.validatedURL || 'Page failed to load.')
      setLoading(false)
      syncNavigationState(detail.validatedURL)
    }
    const onConsole = (event: Event) => {
      const detail = event as Event & { message?: string }
      const message = detail.message || ''

      if (!message.startsWith(PICKER_RESULT_PREFIX)) {
        return
      }

      try {
        const parsed = JSON.parse(message.slice(PICKER_RESULT_PREFIX.length)) as PickerConsoleResult

        if (parsed.cancelled) {
          clearPickerTimeout()
          setPicking(false)
          return
        }

        completePicker(parsed.element ?? null)
      } catch (error) {
        clearPickerTimeout()
        setPicking(false)
        notifyError(error, 'Element picker failed')
      }
    }

    webview.addEventListener('console-message', onConsole)
    webview.addEventListener('dom-ready', onDomReady)
    webview.addEventListener('did-fail-load', onFail)
    webview.addEventListener('did-navigate', onNavigate)
    webview.addEventListener('did-navigate-in-page', onNavigate)
    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    host.appendChild(webview)
    webviewRef.current = webview

    return () => {
      clearPickerTimeout()
      webview.removeEventListener('console-message', onConsole)
      webview.removeEventListener('dom-ready', onDomReady)
      webview.removeEventListener('did-fail-load', onFail)
      webview.removeEventListener('did-navigate', onNavigate)
      webview.removeEventListener('did-navigate-in-page', onNavigate)
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.remove()
      webviewRef.current = null
    }
  }, [open])

  const applyViewportPreset = (nextValue: string) => {
    const nextPreset = viewportPresetFor(nextValue)
    const webview = webviewRef.current
    const userAgent = userAgentForViewport(nextPreset.mode)

    setViewportValue(nextValue)

    if (!webview) {
      return
    }

    webview.setAttribute('useragent', userAgent)
    webview.setUserAgent?.(userAgent)

    if (currentUrl !== BLANK_URL) {
      webview.reload?.()
    }
  }

  if (!open) {
    return null
  }

  const loadAddress = () => {
    const nextUrl = normalizeBrowserFeedbackUrl(address)
    const webview = webviewRef.current
    setSelectedElement(null)
    setLoadError(null)
    setLoading(true)
    setCurrentUrl(nextUrl)
    setAddress(addressForUrl(nextUrl))

    if (webview?.loadURL) {
      void webview.loadURL(nextUrl, { userAgent: userAgentForViewport(viewportPreset.mode) }).catch(error => {
        setLoading(false)
        setLoadError(error instanceof Error ? error.message : String(error))
      })

      return
    }

    webview?.setAttribute('src', nextUrl)
  }

  const pickElement = async () => {
    const webview = webviewRef.current

    if (!webview?.executeJavaScript || !ready) {
      notify({ kind: 'warning', message: 'Wait for the page to finish loading, then try Pick element again.', title: 'Element picker' })
      return
    }

    setPicking(true)
    setSelectedElement(null)
    clearPickerTimeout()

    try {
      await webview.executeJavaScript<boolean>(PICKER_SCRIPT, true)
      pickerTimeoutRef.current = window.setTimeout(() => {
        setPicking(false)
        pickerTimeoutRef.current = null
        notify({
          kind: 'warning',
          message: 'No element was selected. Click Pick element again and then click inside the page, or press Esc to cancel.',
          title: 'Element picker'
        })
      }, 30_000)
    } catch (err) {
      clearPickerTimeout()
      setPicking(false)
      notifyError(err, 'Element picker failed')
    }
  }

  const insertPrompt = () => {
    if (!selectedElement) {
      notify({ kind: 'warning', message: 'Pick an element in the browser first.', title: 'Web Browser' })
      return
    }

    onInsertPrompt(buildElementChangePrompt(selectedElement, comment))
    setComment('')
    notify({ kind: 'success', message: 'The selected element request was added to the chat composer.', title: 'Web Browser' })
  }

  return (
    <>
      <div
        aria-hidden={minimized}
        className={`fixed bottom-3 left-3 right-3 top-[calc(var(--titlebar-height)+0.75rem)] z-[80] flex flex-col overflow-hidden rounded-2xl border border-border/70 bg-background/95 shadow-2xl backdrop-blur [-webkit-app-region:no-drag] ${minimized ? 'pointer-events-none invisible opacity-0' : ''}`}
      >
      <div className="flex min-h-12 items-center gap-2 border-b border-border/70 px-3">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Codicon name="globe" />
          Web Browser
        </div>
        <form
          className="flex min-w-0 flex-1 items-center gap-2"
          onSubmit={event => {
            event.preventDefault()
            loadAddress()
          }}
        >
          <Button
            disabled={!ready || !canGoBack}
            onClick={() => webviewRef.current?.goBack?.()}
            size="sm"
            title="Back"
            type="button"
            variant="ghost"
          >
            <Codicon name="arrow-left" />
          </Button>
          <Button
            disabled={!ready || !canGoForward}
            onClick={() => webviewRef.current?.goForward?.()}
            size="sm"
            title="Forward"
            type="button"
            variant="ghost"
          >
            <Codicon name="arrow-right" />
          </Button>
          <Button
            disabled={!ready}
            onClick={() => webviewRef.current?.reload?.()}
            size="sm"
            title="Reload"
            type="button"
            variant="ghost"
          >
            <Codicon name={loading ? 'loading' : 'refresh'} spinning={loading} />
          </Button>
          <input
            className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-3 text-xs text-foreground outline-none ring-offset-background transition focus-visible:ring-2 focus-visible:ring-ring"
            onChange={event => setAddress(event.target.value)}
            value={address}
          />
          <Button size="sm" type="submit" variant="secondary">
            Load
          </Button>
        </form>
        <label className="flex items-center gap-1 text-xs text-muted-foreground" title="Viewport mode and resolution">
          <span className="hidden xl:inline">Viewport</span>
          <select
            className="h-8 max-w-48 rounded-md border border-input bg-background px-2 text-xs text-foreground outline-none ring-offset-background transition focus-visible:ring-2 focus-visible:ring-ring"
            onChange={event => applyViewportPreset(event.target.value)}
            value={viewportValue}
          >
            <optgroup label="Desktop">
              {VIEWPORT_PRESETS.filter(preset => preset.mode === 'desktop').map(preset => (
                <option key={preset.value} value={preset.value}>
                  {preset.label}
                </option>
              ))}
            </optgroup>
            <optgroup label="Mobile / tablet">
              {VIEWPORT_PRESETS.filter(preset => preset.mode === 'mobile').map(preset => (
                <option key={preset.value} value={preset.value}>
                  {preset.label}
                </option>
              ))}
            </optgroup>
          </select>
        </label>
        <Button onClick={() => onMinimizedChange(true)} size="sm" title="Minimize Web Browser" type="button" variant="ghost">
          <Codicon name="chrome-minimize" />
        </Button>
        <Button onClick={onClose} size="sm" title="Close Web Browser" type="button" variant="ghost">
          <Codicon name="close" />
        </Button>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_22rem] bg-muted/20">
        <div className="relative min-h-0 overflow-auto border-r border-border/70 bg-muted/30 p-3">
          <div
            className="mx-auto flex overflow-hidden rounded-lg border border-border/70 bg-background shadow-sm"
            ref={hostRef}
            style={viewportStyle}
          />
          {(loading || loadError) && (
            <div className="pointer-events-none absolute left-3 top-3 rounded-full border border-border/70 bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow">
              {loadError ? `Load failed: ${loadError}` : 'Loading…'}
            </div>
          )}
        </div>
        <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto bg-background/95 p-3 text-sm">
          <div>
            <div className="font-medium text-foreground">1. Select element</div>
            <p className="mt-1 text-xs text-muted-foreground">
              Click the picker, then click the exact element inside the loaded page.
            </p>
            <Button className="mt-2 w-full" disabled={picking || loading || !ready} onClick={() => void pickElement()} type="button">
              <Codicon name={picking ? 'loading' : 'inspect'} spinning={picking} />
              {picking ? 'Waiting for click…' : loading ? 'Wait for page load…' : 'Pick element'}
            </Button>
          </div>

          <div>
            <div className="font-medium text-foreground">2. Comment</div>
            <Textarea
              className="mt-2 min-h-28 text-xs"
              onChange={event => setComment(event.target.value)}
              placeholder="Example: make this button larger, change this text, move this section…"
              value={comment}
            />
          </div>

          <div className="min-h-0 flex-1">
            <div className="font-medium text-foreground">Selected metadata</div>
            <pre className="mt-2 max-h-64 overflow-auto rounded-lg border border-border/70 bg-muted/50 p-2 text-[0.6875rem] text-muted-foreground">
              {selectedElement ? JSON.stringify(selectedElement, null, 2) : 'No element selected yet.'}
            </pre>
          </div>

          <Button disabled={!selectedElement} onClick={insertPrompt} type="button">
            <Codicon name="comment-discussion" />
            Add request to chat
          </Button>

          {promptPreview && (
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer text-foreground">Prompt preview</summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg border border-border/70 bg-muted/50 p-2">
                {promptPreview}
              </pre>
            </details>
          )}
        </aside>
      </div>
    </div>
    </>
  )
}
