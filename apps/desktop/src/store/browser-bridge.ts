import {
  $browserEnabled,
  $browserTabs,
  appendBrowserActionEvent,
  appendBrowserScreenshotEntry,
  type BrowserTabId,
  type BrowserTabState,
  clearBrowserConsoleEntries,
  clearBrowserNetworkEvents,
  getBrowserConsoleEntries,
  getBrowserNetworkEvents,
  isBrowserTabId
} from './browser'
import { clearGuestNonce, GUEST_SENTINEL_PREFIX, issueGuestNonce } from './browser-guest-bus'

export type BrowserBridgeCommand =
  | 'accessibilityAudit'
  | 'clearConsole'
  | 'clearNetwork'
  | 'click'
  | 'clickRef'
  | 'designHandoff'
  | 'doubleClick'
  | 'doubleClickRef'
  | 'evaluate'
  | 'fillRef'
  | 'getConsole'
  | 'getImages'
  | 'getNetwork'
  | 'getState'
  | 'goBack'
  | 'goForward'
  | 'hover'
  | 'hoverRef'
  | 'inspectElement'
  | 'navigate'
  | 'press'
  | 'reload'
  | 'rightClick'
  | 'rightClickRef'
  | 'screenshot'
  | 'scroll'
  | 'selectElement'
  | 'snapshot'
  | 'stop'
  | 'type'

interface BrowserBridgeWebview {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  capturePage?: () => Promise<{ toDataURL?: () => string } | string>
  executeJavaScript?: <T = unknown>(script: string) => Promise<T>
  getTitle?: () => string
  getURL?: () => string
  goBack?: () => void
  goForward?: () => void
  reload?: () => void
  sendInputEvent?: (event: Record<string, unknown>) => void
  setAttribute?: (name: string, value: string) => void
  stop?: () => void
}

interface ClickParams {
  x?: number
  y?: number
}

interface RefParams {
  ref?: string
}

interface FillParams extends RefParams {
  text?: string
}

interface EvaluateParams {
  expression?: string
  script?: string
}

interface NavigateParams {
  url?: string
}

interface PressParams {
  key?: string
}

interface ScrollParams {
  amount?: number
  direction?: string
}

interface TypeParams {
  text?: string
}

interface DesignHandoffParams {
  goal?: string
  refs?: unknown
}

const registry = new Map<BrowserTabId, BrowserBridgeWebview>()

const observeCommands = new Set<BrowserBridgeCommand>([
  'accessibilityAudit',
  'clearConsole',
  'clearNetwork',
  'designHandoff',
  'getConsole',
  'getImages',
  'getNetwork',
  'getState',
  'inspectElement',
  'screenshot',
  'selectElement',
  'snapshot'
])

const controlCommands = new Set<BrowserBridgeCommand>([
  'click',
  'clickRef',
  'doubleClick',
  'doubleClickRef',
  'evaluate',
  'fillRef',
  'goBack',
  'goForward',
  'hover',
  'hoverRef',
  'navigate',
  'press',
  'reload',
  'rightClick',
  'rightClickRef',
  'scroll',
  'stop',
  'type'
])

export function registerBrowserWebview(tabId: BrowserTabId, webview: BrowserBridgeWebview): () => void {
  registry.set(tabId, webview)

  return () => {
    if (registry.get(tabId) === webview) {
      registry.delete(tabId)
    }
  }
}

export function resolveBrowserBridgeTargetTabId(tabId?: unknown, sessionId?: string | null): BrowserTabId | null {
  const tabs = $browserTabs.get()

  if (isBrowserTabId(tabId)) {
    const tab = tabs.find(candidate => candidate.id === tabId)

    if (!tab) {
      return null
    }

    if (sessionId && tab.sessionId !== sessionId) {
      throw new Error(`Browser tab ${tabId} belongs to a different session`)
    }

    return tabId
  }

  const matchesSession = (tab: BrowserTabState) => !sessionId || tab.sessionId === sessionId
  const isBound = (tab: BrowserTabState) => tab.controlMode === 'observe' || tab.controlMode === 'control'

  return tabs.find(tab => matchesSession(tab) && tab.controlMode === 'control')?.id
    ?? tabs.find(tab => matchesSession(tab) && isBound(tab))?.id
    ?? null
}

export async function runBrowserBridgeCommand(
  tabId: BrowserTabId,
  command: BrowserBridgeCommand,
  params: Record<string, unknown> = {}
): Promise<unknown> {
  if (!$browserEnabled.get()) {
    throw new Error('Browser is disabled')
  }

  const webview = registry.get(tabId)

  if (!webview) {
    throw new Error(`Browser tab ${tabId} is not registered`)
  }

  const tab = $browserTabs.get().find(candidate => candidate.id === tabId)

  if (!tab || tab.controlMode === 'idle' || tab.controlMode === 'paused') {
    throw new Error(`Browser tab ${tabId} is not bound for agent access`)
  }

  if (controlCommands.has(command) && tab.controlMode !== 'control') {
    throw new Error(`Browser command ${command} requires control consent`)
  }

  if (!observeCommands.has(command) && !controlCommands.has(command)) {
    throw new Error(`Unsupported browser command: ${command}`)
  }

  try {
    const result = await executeBrowserBridgeCommand(webview, tab, command, params)

    appendBrowserActionEvent(tabId, {
      command,
      result: summarizeActionResult(result),
      status: 'success',
      target: summarizeActionTarget(params)
    })

    return result
  } catch (error) {
    appendBrowserActionEvent(tabId, {
      command,
      error: error instanceof Error ? error.message : String(error),
      status: 'error',
      target: summarizeActionTarget(params)
    })

    throw error
  }
}

async function executeBrowserBridgeCommand(
  webview: BrowserBridgeWebview,
  tab: BrowserTabState,
  command: BrowserBridgeCommand,
  params: Record<string, unknown>
): Promise<unknown> {
  const tabId = tab.id

  switch (command) {
    case 'getConsole':
      return { messages: getBrowserConsoleEntries(tabId) }

    case 'clearConsole':
      clearBrowserConsoleEntries(tabId)

      return { ok: true }

    case 'getNetwork':
      return { events: getBrowserNetworkEvents(tabId) }

    case 'clearNetwork':
      clearBrowserNetworkEvents(tabId)

      return { ok: true }

    case 'getState':
      return {
        canGoBack: Boolean(webview.canGoBack?.()),
        canGoForward: Boolean(webview.canGoForward?.()),
        title: webview.getTitle?.() || tab.title,
        url: webview.getURL?.() || tab.url
      }

    case 'inspectElement':
      return webview.executeJavaScript?.(inspectElementScript((params as RefParams).ref)) ?? { element: null }
    case 'selectElement': {
      const selectionParams = params as DesignHandoffParams & RefParams
      const refs = Array.isArray(selectionParams.refs) ? selectionParams.refs : [selectionParams.ref]

      return webview.executeJavaScript?.(selectElementsScript(refs)) ?? { selected: [] }
    }

    case 'designHandoff': {
      const handoffParams = params as DesignHandoffParams
      const refs = Array.isArray(handoffParams.refs) ? handoffParams.refs : [(params as RefParams).ref]

      return webview.executeJavaScript?.(designHandoffScript(refs, handoffParams.goal)) ?? { selected: [] }
    }

    case 'accessibilityAudit':
      return webview.executeJavaScript?.(ACCESSIBILITY_AUDIT_SCRIPT) ?? { findings: [], summary: { error: 0, warning: 0 } }
    case 'navigate': {
      const url = normalizeCommandUrl((params as NavigateParams).url)
      webview.setAttribute?.('src', url)

      return { url }
    }

    case 'goBack':
      webview.goBack?.()

      return { ok: true }

    case 'goForward':
      webview.goForward?.()

      return { ok: true }

    case 'reload':
      webview.reload?.()

      return { ok: true }

    case 'stop':
      webview.stop?.()

      return { ok: true }
    case 'click': {
      const { x, y } = normalizePoint(params as ClickParams)
      sendMouseClick(webview, x, y, 'left', 1)

      return { x, y }
    }

    case 'doubleClick': {
      const { x, y } = normalizePoint(params as ClickParams)
      sendMouseClick(webview, x, y, 'left', 2)

      return { x, y }
    }

    case 'rightClick': {
      const { x, y } = normalizePoint(params as ClickParams)
      sendMouseClick(webview, x, y, 'right', 1)

      return { x, y }
    }

    case 'hover': {
      const { x, y } = normalizePoint(params as ClickParams)
      webview.sendInputEvent?.({ type: 'mouseMove', x, y })

      return { x, y }
    }

    case 'clickRef':
      return webview.executeJavaScript?.(refActionScript((params as RefParams).ref, 'click')) ?? { ok: false }

    case 'doubleClickRef':
      return webview.executeJavaScript?.(refActionScript((params as RefParams).ref, 'doubleClick')) ?? { ok: false }

    case 'rightClickRef':
      return webview.executeJavaScript?.(refActionScript((params as RefParams).ref, 'rightClick')) ?? { ok: false }

    case 'hoverRef':
      return webview.executeJavaScript?.(refActionScript((params as RefParams).ref, 'hover')) ?? { ok: false }

    case 'fillRef':
      return webview.executeJavaScript?.(refActionScript((params as FillParams).ref, 'fill', (params as FillParams).text)) ?? { ok: false }

    case 'scroll':
      return webview.executeJavaScript?.(scrollScript((params as ScrollParams).direction, (params as ScrollParams).amount)) ?? { ok: false }
    case 'evaluate': {
      const expression = normalizeExpression(params as EvaluateParams)

      return webview.executeJavaScript?.(expression) ?? null
    }

    case 'getImages':
      return webview.executeJavaScript?.(GET_IMAGES_SCRIPT) ?? []
    case 'type': {
      const rawText = (params as TypeParams).text
      const text = typeof rawText === 'string' ? rawText : ''
      webview.sendInputEvent?.({ text, type: 'char' })

      return { characters: text.length }
    }

    case 'press': {
      const key = typeof (params as PressParams).key === 'string' ? (params as PressParams).key : 'Enter'
      webview.sendInputEvent?.({ keyCode: key, type: 'keyDown' })
      webview.sendInputEvent?.({ keyCode: key, type: 'keyUp' })

      return { key }
    }

    case 'screenshot': {
      const image = await webview.capturePage?.()
      const dataUrl = typeof image === 'string' ? image : image?.toDataURL?.()

      if (!dataUrl) {
        throw new Error('Browser webview does not support screenshot capture')
      }

      appendBrowserScreenshotEntry(tabId, {
        dataUrl,
        title: webview.getTitle?.() || tab.title,
        url: webview.getURL?.() || tab.url
      })

      return { dataUrl }
    }

    case 'snapshot':
      return webview.executeJavaScript?.(SNAPSHOT_SCRIPT) ?? { url: tab.url, title: tab.title, text: '' }
  }
}

function normalizeCommandUrl(value: unknown): string {
  const trimmed = typeof value === 'string' ? value.trim() : ''

  if (!trimmed) {
    return 'about:blank'
  }

  if (/^(localhost|127(?:\.\d{1,3}){3}|\[[^\]]+\]|[\w.-]+):\d+(?:\/|$)/i.test(trimmed)) {
    return `http://${trimmed}`
  }

  if (!/^[a-zA-Z][a-zA-Z\d+.-]*:/.test(trimmed)) {
    return `https://${trimmed}`
  }

  try {
    const parsed = new URL(trimmed)

    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.href
    }

    if (parsed.protocol === 'about:' && parsed.href === 'about:blank') {
      return 'about:blank'
    }
  } catch {
    // Fall through to a safe HTTPS search URL.
  }

  return `https://www.google.com/search?q=${encodeURIComponent(trimmed)}`
}

function normalizePoint(params: ClickParams): { x: number; y: number } {
  return {
    x: Number.isFinite(params.x) ? Math.max(0, Math.round(params.x ?? 0)) : 0,
    y: Number.isFinite(params.y) ? Math.max(0, Math.round(params.y ?? 0)) : 0
  }
}

function sendMouseClick(
  webview: BrowserBridgeWebview,
  x: number,
  y: number,
  button: 'left' | 'right',
  clickCount: number
): void {
  webview.sendInputEvent?.({ button, clickCount, type: 'mouseDown', x, y })
  webview.sendInputEvent?.({ button, clickCount, type: 'mouseUp', x, y })
}

function normalizeExpression(params: EvaluateParams): string {
  const expression = typeof params.expression === 'string'
    ? params.expression
    : typeof params.script === 'string'
      ? params.script
      : ''

  if (!expression.trim()) {
    throw new Error('Browser evaluate command requires an expression')
  }

  return expression
}

function summarizeActionTarget(params: Record<string, unknown>): string | undefined {
  const candidate = params.ref ?? params.url ?? params.key ?? params.direction

  if (typeof candidate === 'string') {
    return candidate
  }

  if (typeof params.x === 'number' || typeof params.y === 'number') {
    return `${params.x ?? 0},${params.y ?? 0}`
  }

  return undefined
}

function summarizeActionResult(result: unknown): string | undefined {
  if (typeof result === 'string') {
    return result
  }

  if (result && typeof result === 'object') {
    const raw = result as Record<string, unknown>
    const candidate = raw.url ?? raw.title ?? raw.ref ?? raw.stableRef ?? raw.key ?? raw.direction

    if (typeof candidate === 'string') {
      return candidate
    }

    try {
      return JSON.stringify(result).slice(0, 512)
    } catch {
      return undefined
    }
  }

  if (typeof result === 'number' || typeof result === 'boolean') {
    return String(result)
  }

  return undefined
}

type RefAction = 'click' | 'doubleClick' | 'fill' | 'hover' | 'rightClick'

function refActionScript(ref: unknown, action: RefAction, text = ''): string {
  const safeRef = JSON.stringify(typeof ref === 'string' ? ref : '')
  const safeText = JSON.stringify(typeof text === 'string' ? text : '')

  return `(() => {
    ${ELEMENT_HELPERS_SOURCE}
    const element = hermesResolveRef(${safeRef});
    element.scrollIntoView?.({ block: 'center', inline: 'center' });
    element.focus?.();
    const action = ${JSON.stringify(action)};
    if (action === 'fill') {
      const value = ${safeText};
      if ('value' in element) element.value = value;
      else element.textContent = value;
      element.dispatchEvent(new InputEvent('input', { bubbles: true, data: value }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
      return { ok: true, ref: hermesRefForElement(element), stableRef: hermesStableRef(element), filled: value.length };
    }
    if (action === 'hover') {
      element.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
      element.dispatchEvent(new MouseEvent('mousemove', { bubbles: true }));
      return { ok: true, ref: hermesRefForElement(element), stableRef: hermesStableRef(element), hovered: true };
    }
    if (action === 'doubleClick') {
      element.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, detail: 2 }));
      return { ok: true, ref: hermesRefForElement(element), stableRef: hermesStableRef(element), doubleClicked: true };
    }
    if (action === 'rightClick') {
      element.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, button: 2 }));
      return { ok: true, ref: hermesRefForElement(element), stableRef: hermesStableRef(element), rightClicked: true };
    }
    element.click();
    return { ok: true, ref: hermesRefForElement(element), stableRef: hermesStableRef(element), text: hermesElementText(element).slice(0, 160) };
  })()`
}

function scrollScript(direction: unknown, amount: unknown): string {
  const dir = direction === 'up' || direction === 'left' || direction === 'right' ? direction : 'down'
  const rawAmount = typeof amount === 'number' && Number.isFinite(amount) ? Math.abs(amount) : 700
  const distance = Math.max(1, Math.min(5000, Math.round(rawAmount)))
  const top = dir === 'up' ? -distance : dir === 'down' ? distance : 0
  const left = dir === 'left' ? -distance : dir === 'right' ? distance : 0

  return `(() => { window.scrollBy({ top: ${top}, left: ${left}, behavior: 'instant' }); return { ok: true, direction: ${JSON.stringify(dir)}, amount: ${distance} }; })()`
}

function inspectElementScript(ref: unknown): string {
  const safeRef = JSON.stringify(typeof ref === 'string' ? ref : '')

  return `(() => {
    ${ELEMENT_HELPERS_SOURCE}
    ${DESIGN_HELPERS_SOURCE}
    const element = hermesResolveRef(${safeRef});
    return { element: hermesInspectElement(element) };
  })()`
}

function selectElementsScript(refs: unknown[]): string {
  const safeRefs = JSON.stringify(refs.filter((ref): ref is string => typeof ref === 'string' && Boolean(ref.trim())).slice(0, 12))

  return `(() => {
    ${ELEMENT_HELPERS_SOURCE}
    ${DESIGN_HELPERS_SOURCE}
    const refs = ${safeRefs};
    const selected = refs.map(ref => hermesInspectElement(hermesResolveRef(ref)));
    return { selected };
  })()`
}

function designHandoffScript(refs: unknown[], goal: unknown): string {
  const safeGoal = JSON.stringify(typeof goal === 'string' ? goal.slice(0, 2000) : '')
  const safeRefs = JSON.stringify(refs.filter((ref): ref is string => typeof ref === 'string' && Boolean(ref.trim())).slice(0, 12))

  return `(() => {
    ${ELEMENT_HELPERS_SOURCE}
    ${DESIGN_HELPERS_SOURCE}
    const goal = ${safeGoal};
    const refs = ${safeRefs};
    const selected = refs.map(ref => hermesInspectElement(hermesResolveRef(ref)));
    const prompt = [
      'Hermes Browser Design Mode handoff (safe agent-mediated replacement for direct visual editing).',
      'Goal: ' + (goal || 'Inspect selected elements and propose code changes.'),
      'Do not mutate the live DOM as the source of truth. Find the owning source files, propose a patch, run tests, and report verification.',
      'Selected elements JSON:',
      JSON.stringify(selected, null, 2)
    ].join('\\n');
    return {
      mode: 'agent-mediated',
      unsafeDirectDomMutation: false,
      goal,
      selected,
      prompt,
      nextSteps: ['locate owning source/component', 'draft patch', 'run focused visual/UI tests', 'summarize diff and verification']
    };
  })()`
}

const INTERACTIVE_ELEMENT_SELECTOR = 'a[href],button,input,textarea,select,summary,[contenteditable="true"],[onclick],[aria-label],[role="button"],[role="link"],[role="textbox"],[role="menuitem"],[role="checkbox"],[role="radio"],[role="tab"],[tabindex]:not([tabindex="-1"])'

const VISIBLE_ELEMENT_SOURCE = `(element) => {
  const style = window.getComputedStyle(element)
  const rect = element.getBoundingClientRect()
  return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0
}`

export const ELEMENT_HELPERS_SOURCE = `
  const HERMES_INTERACTIVE_SELECTOR = ${JSON.stringify(INTERACTIVE_ELEMENT_SELECTOR)};
  const hermesVisible = ${VISIBLE_ELEMENT_SOURCE};
  const hermesElements = () => Array.from(document.querySelectorAll(HERMES_INTERACTIVE_SELECTOR)).filter(hermesVisible);
  const hermesElementText = (element) => String(
    element.innerText
    || element.textContent
    || element.value
    || element.getAttribute('aria-label')
    || element.getAttribute('name')
    || element.getAttribute('title')
    || ''
  ).trim();
  const hermesCssPath = (element) => {
    const parts = [];
    let node = element;
    while (node && node.nodeType === Node.ELEMENT_NODE && parts.length < 5) {
      const tag = node.tagName.toLowerCase();
      const id = node.id ? '#' + node.id : '';
      const parent = node.parentElement;
      const index = parent ? Array.from(parent.children).filter(child => child.tagName === node.tagName).indexOf(node) + 1 : 1;
      parts.unshift(tag + id + ':nth-of-type(' + index + ')');
      node = parent;
    }
    return parts.join('>');
  };
  const hermesHash = (value) => {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(36);
  };
  const hermesStableRef = (element) => '@s' + hermesHash([
    element.tagName.toLowerCase(),
    element.getAttribute('role') || '',
    element.getAttribute('href') || '',
    element.getAttribute('aria-label') || '',
    element.getAttribute('name') || '',
    hermesElementText(element).slice(0, 120),
    hermesCssPath(element)
  ].join('|'));
  const hermesRefForElement = (element) => '@e' + hermesElements().indexOf(element);
  const hermesResolveRef = (ref) => {
    const normalized = String(ref || '').trim();
    const elements = hermesElements();
    const indexMatch = /^@?e(\\d+)$/i.exec(normalized);
    if (indexMatch) {
      const element = elements[Number(indexMatch[1])];
      if (!element) throw new Error('Browser ref ' + normalized + ' not found');
      return element;
    }
    const stable = normalized.startsWith('@') ? normalized : '@' + normalized;
    if (/^@s[a-z0-9]+$/i.test(stable)) {
      const element = elements.find(candidate => hermesStableRef(candidate) === stable);
      if (!element) throw new Error('Stable browser ref ' + stable + ' not found');
      return element;
    }
    throw new Error('Invalid browser ref: ' + normalized);
  };
  const hermesDescribeElement = (element, index) => {
    const tag = element.tagName.toLowerCase();
    return {
      ref: '@e' + index,
      stableRef: hermesStableRef(element),
      index,
      label: tag === 'a' ? 'link' : tag === 'input' || tag === 'textarea' || tag === 'select' ? 'input' : 'control',
      text: hermesElementText(element).slice(0, 160),
      href: element.getAttribute('href') || undefined,
      role: element.getAttribute('role') || undefined,
      tag,
      name: element.getAttribute('name') || undefined,
      ariaLabel: element.getAttribute('aria-label') || undefined
    };
  };
`

export const DESIGN_HELPERS_SOURCE = `
  const hermesStyleSample = (style) => ({
    color: style.color || '',
    backgroundColor: style.backgroundColor || '',
    fontSize: style.fontSize || '',
    fontWeight: style.fontWeight || '',
    lineHeight: style.lineHeight || '',
    display: style.display || '',
    position: style.position || '',
    flexDirection: style.flexDirection || '',
    justifyContent: style.justifyContent || '',
    alignItems: style.alignItems || '',
    flexWrap: style.flexWrap || '',
    gap: style.gap || '',
    overflow: style.overflow || '',
    margin: style.margin || '',
    marginTop: style.marginTop || '',
    marginRight: style.marginRight || '',
    marginBottom: style.marginBottom || '',
    marginLeft: style.marginLeft || '',
    padding: style.padding || '',
    paddingTop: style.paddingTop || '',
    paddingRight: style.paddingRight || '',
    paddingBottom: style.paddingBottom || '',
    paddingLeft: style.paddingLeft || '',
    border: style.border || '',
    borderRadius: style.borderRadius || '',
    width: style.width || '',
    height: style.height || '',
    opacity: style.opacity || '',
    transform: style.transform || ''
  });
  const hermesAttributes = (element) => Object.fromEntries(Array.from(element.attributes || [])
    .filter(attr => ['id', 'class', 'role', 'aria-label', 'name', 'type', 'href', 'src', 'alt', 'title'].includes(attr.name))
    .slice(0, 16)
    .map(attr => [attr.name, String(attr.value).slice(0, 500)]));
  const hermesInspectElement = (element) => {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return {
      ...hermesDescribeElement(element, hermesElements().indexOf(element)),
      attributes: hermesAttributes(element),
      className: String(element.getAttribute('class') || '').slice(0, 500),
      cssPath: hermesCssPath(element),
      htmlPreview: String(element.outerHTML || '').replace(/\\s+/g, ' ').slice(0, 1200),
      accessibility: {
        role: element.getAttribute('role') || '',
        name: hermesElementText(element).slice(0, 300),
        ariaLabel: element.getAttribute('aria-label') || '',
        ariaHidden: element.getAttribute('aria-hidden') || '',
        disabled: Boolean(element.disabled || element.getAttribute('aria-disabled') === 'true')
      },
      layout: {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        top: Math.round(rect.top),
        left: Math.round(rect.left),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      },
      styles: hermesStyleSample(style)
    };
  };
`

const ACCESSIBILITY_AUDIT_SCRIPT = `(() => {
  ${ELEMENT_HELPERS_SOURCE}
  const findings = [];
  const push = (severity, rule, message, element) => findings.push({
    severity,
    rule,
    message,
    element: element ? hermesDescribeElement(element, Math.max(0, hermesElements().indexOf(element))) : undefined
  });
  const accessibleName = (element) => hermesElementText(element)
    || element.getAttribute('aria-label')
    || element.getAttribute('alt')
    || element.getAttribute('title')
    || '';
  document.querySelectorAll('img').forEach(img => {
    if (!img.hasAttribute('alt') || !String(img.getAttribute('alt') || '').trim()) {
      push('error', 'image-alt', 'Image is missing non-empty alt text.', img);
    }
  });
  document.querySelectorAll('button,[role="button"],summary,[tabindex]:not([tabindex="-1"])').forEach(control => {
    if (!accessibleName(control).trim()) {
      push('error', 'control-name', 'Interactive control has no accessible name.', control);
    }
  });
  document.querySelectorAll('a[href]').forEach(link => {
    if (!accessibleName(link).trim()) {
      push('error', 'link-name', 'Link has no accessible name.', link);
    }
  });
  const cssEscape = (value) => window.CSS && typeof window.CSS.escape === 'function'
    ? window.CSS.escape(value)
    : String(value).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
  document.querySelectorAll('input,textarea,select').forEach(input => {
    const id = input.getAttribute('id');
    const labeled = Boolean(
      input.getAttribute('aria-label')
      || input.getAttribute('aria-labelledby')
      || input.getAttribute('title')
      || (id && document.querySelector('label[for="' + cssEscape(id) + '"]'))
      || input.closest('label')
    );
    if (!labeled && input.getAttribute('type') !== 'hidden') {
      push('error', 'input-label', 'Form control is missing a label or aria-label.', input);
    }
  });
  document.querySelectorAll('[aria-hidden="true"] a[href],[aria-hidden="true"] button,[aria-hidden="true"] input,[aria-hidden="true"] [tabindex]:not([tabindex="-1"])').forEach(element => {
    push('warning', 'aria-hidden-focusable', 'Focusable content is hidden from assistive technology.', element);
  });
  const summary = findings.reduce((counts, finding) => {
    counts[finding.severity] = (counts[finding.severity] || 0) + 1;
    return counts;
  }, { error: 0, warning: 0 });
  return { findings: findings.slice(0, 120), summary, url: location.href, title: document.title };
})()`

const SNAPSHOT_SCRIPT = `(() => {
  ${ELEMENT_HELPERS_SOURCE}
  const elements = hermesElements().slice(0, 120).map(hermesDescribeElement)

  return {
    title: document.title,
    url: location.href,
    text: (document.body?.innerText || '').trim().slice(0, 16000),
    elements
  }
})()`

const GET_IMAGES_SCRIPT = `(() => Array.from(document.images)
  .map(img => ({
    src: img.currentSrc || img.src,
    alt: img.alt || '',
    width: img.naturalWidth || img.width || 0,
    height: img.naturalHeight || img.height || 0,
    title: img.title || ''
  }))
  .filter(img => img.src && !img.src.startsWith('data:')))`

// ---------------------------------------------------------------------------
// Interactive guest tooling — element picker / component tree / design mode.
//
// Runs guest JS via executeJavaScript (no preload, by design). Uses a robust
// `data-hermes-ref` handle so ANY element (not only the interactive set the
// `@e`/`@s` refs cover) can be targeted, and reports discrete events back over
// the nonce-bound console sentinel (see browser-guest-bus.ts). Consent-gated and
// (deliberately) NOT action-recorded so live design preview can't spam the
// Timeline or thrash the persistent atom.
// ---------------------------------------------------------------------------

const HANDLE_ATTR = 'data-hermes-ref'

/** Resolve a host-issued handle (a `data-hermes-ref` uid OR a css path) to a live
 *  node, and inspect ANY element (`hermesInspectElement`'s `@e` ref is
 *  interactive-only, so the handle is preserved as the ref). */
export const GUEST_HANDLE_HELPERS_SOURCE = `
  ${ELEMENT_HELPERS_SOURCE}
  ${DESIGN_HELPERS_SOURCE}
  const HERMES_HANDLE_ATTR = ${JSON.stringify(HANDLE_ATTR)};
  const hermesResolveHandle = (handle) => {
    const value = String(handle == null ? '' : handle);
    if (!value) return null;
    const byAttr = document.querySelector('[' + HERMES_HANDLE_ATTR + '=' + JSON.stringify(value) + ']');
    if (byAttr) return byAttr;
    try { return document.querySelector(value); } catch (e) { return null; }
  };
  const hermesInspectHandle = (handle) => {
    const el = hermesResolveHandle(handle);
    if (!el) return null;
    const data = hermesInspectElement(el);
    data.ref = String(handle);
    return data;
  };
`

const GUEST_HIGHLIGHT_SOURCE = `
  const hermesHighlight = (el, persist) => {
    let box = document.querySelector('[data-hermes-highlight]');
    if (!el) { if (box) box.style.display = 'none'; return; }
    if (!box) {
      box = document.createElement('div');
      box.setAttribute('data-hermes-highlight', '1');
      Object.assign(box.style, { position: 'fixed', pointerEvents: 'none', zIndex: '2147483647', border: '2px solid #4f8cff', background: 'rgba(79,140,255,0.12)', boxSizing: 'border-box', borderRadius: '2px', top: '0', left: '0', width: '0', height: '0' });
      (document.documentElement || document.body).appendChild(box);
    }
    const r = el.getBoundingClientRect();
    box.style.display = 'block';
    box.style.top = r.top + 'px'; box.style.left = r.left + 'px';
    box.style.width = r.width + 'px'; box.style.height = r.height + 'px';
    if (!persist) { setTimeout(() => { try { box.remove(); } catch (e) {} }, 1200); }
  };
`

function requireGuestTab(tabId: BrowserTabId, mode: 'control' | 'observe'): BrowserBridgeWebview {
  if (!$browserEnabled.get()) {
    throw new Error('Browser is disabled')
  }

  const webview = registry.get(tabId)

  if (!webview) {
    throw new Error(`Browser tab ${tabId} is not registered`)
  }

  const tab = $browserTabs.get().find(candidate => candidate.id === tabId)

  if (!tab || tab.controlMode === 'idle' || tab.controlMode === 'paused') {
    throw new Error(`Browser tab ${tabId} is not bound for agent access`)
  }

  if (mode === 'control' && tab.controlMode !== 'control') {
    throw new Error('Operation requires control consent')
  }

  return webview
}

/** Run an arbitrary guest IIFE, consent-gated, WITHOUT recording an action event.
 *  Live design-mode preview fires on every slider input — recording it would spam
 *  the Timeline pane and thrash the persistent tabs atom. */
export async function runGuestScript(
  tabId: BrowserTabId,
  source: string,
  mode: 'control' | 'observe' = 'observe'
): Promise<unknown> {
  const webview = requireGuestTab(tabId, mode)

  return webview.executeJavaScript?.(source)
}

/** Host re-pull of the authoritative (trusted) description for a handle the guest
 *  reported over the (untrusted) sentinel. The sentinel only carries a handle;
 *  THIS is the data the host trusts and shows the user. */
export async function inspectGuestElement(tabId: BrowserTabId, handle: string): Promise<unknown> {
  return runGuestScript(
    tabId,
    `(() => { ${GUEST_HANDLE_HELPERS_SOURCE} return hermesInspectHandle(${JSON.stringify(handle)}); })()`,
    'observe'
  )
}

/** Flash a highlight box around the element a handle resolves to (tree hover/select). */
export async function highlightGuestElement(tabId: BrowserTabId, handle: string): Promise<void> {
  await runGuestScript(
    tabId,
    `(() => { ${GUEST_HANDLE_HELPERS_SOURCE} ${GUEST_HIGHLIGHT_SOURCE} hermesHighlight(hermesResolveHandle(${JSON.stringify(handle)}), false); })()`,
    'observe'
  )
}

/** Begin user element-picking: inject a nonce'd hover-highlight + click-select
 *  overlay. The selecting click tags the chosen element with a `data-hermes-ref`
 *  uid and reports it over the console sentinel; the host then re-pulls the
 *  authoritative description via `inspectGuestElement`. */
export async function startGuestPick(tabId: BrowserTabId): Promise<void> {
  const webview = requireGuestTab(tabId, 'observe')
  const nonce = issueGuestNonce(tabId)

  const source = `(() => {
    ${GUEST_HANDLE_HELPERS_SOURCE}
    ${GUEST_HIGHLIGHT_SOURCE}
    if (window.__hermesPicker) { try { window.__hermesPicker(); } catch (e) {} }
    const NONCE = ${JSON.stringify(nonce)};
    const SENTINEL = ${JSON.stringify(GUEST_SENTINEL_PREFIX)};
    let raf = 0; let last = null;
    const targetAt = (x, y) => {
      const box = document.querySelector('[data-hermes-highlight]');
      const prev = box ? box.style.display : '';
      if (box) box.style.display = 'none';
      const el = document.elementFromPoint(x, y);
      if (box) box.style.display = prev;
      if (!el || (box && el === box)) return null;
      return el;
    };
    const onMove = (e) => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const el = targetAt(e.clientX, e.clientY);
        if (el !== last) { last = el; hermesHighlight(el, true); }
      });
    };
    const onClick = (e) => {
      const el = targetAt(e.clientX, e.clientY);
      if (!el) return;
      e.preventDefault(); e.stopImmediatePropagation();
      try { const prev = document.querySelector('[' + HERMES_HANDLE_ATTR + ']'); if (prev) prev.removeAttribute(HERMES_HANDLE_ATTR); } catch (err) {}
      const uid = 'p' + Date.now().toString(36) + Math.floor(Math.random() * 1e6).toString(36);
      try { el.setAttribute(HERMES_HANDLE_ATTR, uid); } catch (err) {}
      try { console.log(SENTINEL + NONCE + ':' + JSON.stringify({ kind: 'picked', ref: uid })); } catch (err) {}
    };
    const onScroll = () => { if (last) hermesHighlight(last, true); };
    document.addEventListener('mousemove', onMove, true);
    document.addEventListener('click', onClick, true);
    window.addEventListener('scroll', onScroll, true);
    window.__hermesPicker = () => {
      document.removeEventListener('mousemove', onMove, true);
      document.removeEventListener('click', onClick, true);
      window.removeEventListener('scroll', onScroll, true);
      const box = document.querySelector('[data-hermes-highlight]');
      if (box) { try { box.remove(); } catch (e) {} }
      window.__hermesPicker = null;
    };
    return true;
  })()`

  await webview.executeJavaScript?.(source)
}

/** Tear down the picker overlay + listeners and forget the tab's nonce. Safe to
 *  call when the webview is gone (no-op). */
export async function stopGuestPick(tabId: BrowserTabId): Promise<void> {
  clearGuestNonce(tabId)

  const webview = registry.get(tabId)

  if (!webview) {
    return
  }

  await webview.executeJavaScript?.(
    `(() => { if (window.__hermesPicker) { try { window.__hermesPicker(); } catch (e) {} } const b = document.querySelector('[data-hermes-highlight]'); if (b) { try { b.remove(); } catch (e) {} } })()`
  )
}
