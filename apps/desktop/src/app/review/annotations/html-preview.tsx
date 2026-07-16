import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { $annotations, beginAnnotation, documentReviewContext, type ReviewContext } from '@/store/annotations'

const MESSAGE_SELECTION = 'hermes:annotation:selection'
const MESSAGE_READY = 'hermes:annotation:ready'
const MESSAGE_APPLY = 'hermes:annotation:apply'
const MESSAGE_ZOOM = 'hermes:annotation:zoom'
const MESSAGE_PAN = 'hermes:annotation:pan'
export const HTML_ZOOM_MIN = 50
export const HTML_ZOOM_MAX = 200
export const HTML_ZOOM_STEP = 10

export function clampHtmlZoom(zoom: number): number {
  return Math.max(HTML_ZOOM_MIN, Math.min(HTML_ZOOM_MAX, zoom))
}

export function HtmlZoomControls({
  onPanChange,
  onZoomChange,
  panEnabled = false,
  zoom
}: {
  onPanChange?: (enabled: boolean) => void
  onZoomChange: (zoom: number) => void
  panEnabled?: boolean
  zoom: number
}) {
  const { t } = useI18n()
  const copy = t.desktop.annotations.preview.html

  return (
    <div
      aria-label={copy.zoom}
      className="flex h-5 shrink-0 items-stretch overflow-hidden rounded border border-(--ui-stroke-secondary) bg-(--ui-surface-secondary) text-[0.625rem] tabular-nums text-foreground shadow-xs"
      role="group"
    >
      <Button
        aria-label={copy.decrease}
        className="grid w-5 place-items-center hover:bg-accent hover:text-foreground disabled:opacity-35"
        disabled={zoom <= HTML_ZOOM_MIN}
        onClick={() => onZoomChange(clampHtmlZoom(zoom - HTML_ZOOM_STEP))}
        size="icon"
        title={copy.decrease}
        type="button"
        variant="ghost"
      >
        −
      </Button>
      <Button
        aria-label={copy.reset(zoom)}
        className="min-w-10 border-x border-border/70 px-1 hover:bg-accent hover:text-foreground"
        onClick={() => onZoomChange(100)}
        size="sm"
        title={copy.reset(zoom)}
        type="button"
        variant="ghost"
      >
        {zoom}%
      </Button>
      <Button
        aria-label={copy.increase}
        className="grid w-5 place-items-center hover:bg-accent hover:text-foreground disabled:opacity-35"
        disabled={zoom >= HTML_ZOOM_MAX}
        onClick={() => onZoomChange(clampHtmlZoom(zoom + HTML_ZOOM_STEP))}
        size="icon"
        title={copy.increase}
        type="button"
        variant="ghost"
      >
        +
      </Button>
      {onPanChange && (
        <Button
          aria-label={copy.pan}
          aria-pressed={panEnabled}
          className="grid w-6 place-items-center border-l border-border/70 hover:bg-accent hover:text-foreground aria-pressed:bg-accent aria-pressed:text-foreground"
          onClick={() => onPanChange(!panEnabled)}
          size="icon"
          title={copy.pan}
          type="button"
          variant="ghost"
        >
          <Codicon name="grabber" size="0.7rem" />
        </Button>
      )}
    </div>
  )
}

const BRIDGE_SCRIPT = `
(() => {
  const selectionType = ${JSON.stringify(MESSAGE_SELECTION)};
  const readyType = ${JSON.stringify(MESSAGE_READY)};
  const applyType = ${JSON.stringify(MESSAGE_APPLY)};
  const zoomType = ${JSON.stringify(MESSAGE_ZOOM)};
  const panType = ${JSON.stringify(MESSAGE_PAN)};
  let panEnabled = false;
  let panGesture = null;

  function nodePath(node) {
    const result = [];
    let current = node;
    while (current && current !== document.body) {
      const parent = current.parentNode;
      if (!parent) return null;
      result.unshift(Array.prototype.indexOf.call(parent.childNodes, current));
      current = parent;
    }
    return current === document.body ? result : null;
  }

  function nodeAtPath(path) {
    if (!Array.isArray(path)) return null;
    let current = document.body;
    for (const index of path) {
      if (!Number.isInteger(index) || index < 0 || !current?.childNodes[index]) return null;
      current = current.childNodes[index];
    }
    return current;
  }

  function textNodes() {
    const nodes = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (!node.parentElement?.closest('script,style')) nodes.push(node);
    }
    return nodes;
  }

  function textOffset(node, offset) {
    const range = document.createRange();
    range.selectNodeContents(document.body);
    range.setEnd(node, offset);
    return range.toString().length;
  }

  function finishPan() {
    panGesture = null;
    document.documentElement.style.cursor = panEnabled ? 'grab' : '';
  }

  document.addEventListener('pointerdown', event => {
    if (!panEnabled || event.button !== 0) return;
    panGesture = {
      clientX: event.clientX,
      clientY: event.clientY,
      scrollX: window.scrollX,
      scrollY: window.scrollY
    };
    document.documentElement.style.cursor = 'grabbing';
    event.preventDefault();
    event.stopPropagation();
  }, true);

  window.addEventListener('pointermove', event => {
    if (!panGesture) return;
    window.scrollTo(
      panGesture.scrollX + panGesture.clientX - event.clientX,
      panGesture.scrollY + panGesture.clientY - event.clientY
    );
    event.preventDefault();
  }, true);

  window.addEventListener('pointerup', finishPan, true);
  window.addEventListener('pointercancel', finishPan, true);
  window.addEventListener('blur', finishPan);
  document.addEventListener('click', event => {
    if (panEnabled || event.target?.closest?.('a,area,button,input,select,textarea')) {
      event.preventDefault();
      event.stopPropagation();
    }
  }, true);
  document.addEventListener('submit', event => {
    event.preventDefault();
    event.stopPropagation();
  }, true);

  function clearMarks() {
    if (CSS.highlights) CSS.highlights.delete('hermes-annotations');
    for (const mark of Array.from(document.querySelectorAll('mark[data-hermes-annotation]'))) {
      mark.replaceWith(document.createTextNode(mark.textContent || ''));
    }
    document.body.normalize();
  }

  function rangeFromAnchor(anchor) {
    const start = nodeAtPath(anchor.startPath);
    const end = nodeAtPath(anchor.endPath);
    if (start?.nodeType === Node.TEXT_NODE && end?.nodeType === Node.TEXT_NODE) {
      try {
        const range = document.createRange();
        range.setStart(start, anchor.startNodeOffset || 0);
        range.setEnd(end, anchor.endNodeOffset || 0);
        if (range.toString() === anchor.quote) return range;
      } catch {}
    }

    const nodes = textNodes();
    const fullText = nodes.map(node => node.nodeValue || '').join('');
    let index = -1;
    let cursor = 0;
    while ((cursor = fullText.indexOf(anchor.quote, cursor)) >= 0) {
      const prefix = fullText.slice(Math.max(0, cursor - (anchor.prefix || '').length), cursor);
      const suffix = fullText.slice(cursor + anchor.quote.length, cursor + anchor.quote.length + (anchor.suffix || '').length);
      if ((!anchor.prefix || prefix === anchor.prefix) && (!anchor.suffix || suffix === anchor.suffix)) {
        index = cursor;
        break;
      }
      cursor += Math.max(1, anchor.quote.length);
    }
    if (index < 0) return null;

    let consumed = 0;
    let startNode = null;
    let endNode = null;
    let startOffset = 0;
    let endOffset = 0;
    for (const node of nodes) {
      const length = (node.nodeValue || '').length;
      if (!startNode && index <= consumed + length) {
        startNode = node;
        startOffset = index - consumed;
      }
      if (index + anchor.quote.length <= consumed + length) {
        endNode = node;
        endOffset = index + anchor.quote.length - consumed;
        break;
      }
      consumed += length;
    }
    if (!startNode || !endNode) return null;
    const range = document.createRange();
    range.setStart(startNode, startOffset);
    range.setEnd(endNode, endOffset);
    return range;
  }

  document.addEventListener('mouseup', () => {
    if (panEnabled) return;
    const selection = window.getSelection();
    const text = selection && !selection.isCollapsed ? selection.toString() : '';
    if (!text.trim() || !selection || selection.rangeCount === 0) return;
    const range = selection.getRangeAt(0);
    const startPath = nodePath(range.startContainer);
    const endPath = nodePath(range.endContainer);
    if (!startPath || !endPath) return;
    const rect = range.getBoundingClientRect();
    const fullText = textNodes().map(node => node.nodeValue || '').join('');
    const index = textOffset(range.startContainer, range.startOffset);
    parent.postMessage({
      type: selectionType,
      text,
      startOffset: index,
      endOffset: index + text.length,
      startPath,
      endPath,
      startNodeOffset: range.startOffset,
      endNodeOffset: range.endOffset,
      prefix: fullText.slice(Math.max(0, index - 32), index),
      suffix: fullText.slice(index + text.length, index + text.length + 32),
      rect: { x: rect.left, y: rect.top, width: rect.width, height: rect.height }
    }, '*');
  });

  window.addEventListener('message', event => {
    if (!event.data) return;
    if (event.data.type === zoomType && Number.isFinite(event.data.zoom)) {
      document.documentElement.style.zoom = String(Math.max(.5, Math.min(2, event.data.zoom)));
      return;
    }
    if (event.data.type === panType) {
      panEnabled = Boolean(event.data.enabled);
      document.documentElement.style.cursor = panEnabled ? 'grab' : '';
      document.documentElement.style.userSelect = panEnabled ? 'none' : '';
      return;
    }
    if (event.data.type === applyType && Array.isArray(event.data.anchors)) {
      clearMarks();
      const ranges = event.data.anchors.map(rangeFromAnchor).filter(Boolean);
      if (CSS.highlights && typeof Highlight === 'function') {
        CSS.highlights.set('hermes-annotations', new Highlight(...ranges));
      } else {
        for (const range of ranges) {
          if (range.startContainer !== range.endContainer) continue;
          const mark = document.createElement('mark');
          mark.dataset.hermesAnnotation = 'true';
          range.surroundContents(mark);
        }
      }
    }
  });

  parent.postMessage({ type: readyType }, '*');
})();
`

function safeHtmlDocument(html: string): string {
  const documentNode = new DOMParser().parseFromString(html, 'text/html')

  documentNode
    .querySelectorAll('script,iframe,object,embed,base,form,link,meta[http-equiv="refresh" i]')
    .forEach(node => node.remove())
  documentNode.querySelectorAll('*').forEach(node => {
    for (const attribute of [...node.attributes]) {
      const name = attribute.name.toLowerCase()
      const value = attribute.value.trim().toLowerCase()
      const activeUrl = ['action', 'background', 'data', 'formaction', 'href', 'poster', 'src', 'srcset', 'xlink:href']
      const safeImageSource = name === 'src' && node.localName === 'img' && /^data:image\//.test(value)
      const safeFragment = (name === 'href' || name === 'xlink:href') && value.startsWith('#')

      if (
        name.startsWith('on') ||
        name === 'srcdoc' ||
        (activeUrl.includes(name) && !safeImageSource && !safeFragment)
      ) {
        node.removeAttribute(attribute.name)
      }
    }
  })

  const policy = documentNode.createElement('meta')
  policy.httpEquiv = 'Content-Security-Policy'
  policy.content =
    "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; font-src data:"
  documentNode.head.prepend(policy)

  const viewport = documentNode.createElement('meta')
  viewport.name = 'viewport'
  viewport.content = 'width=device-width, initial-scale=1'
  documentNode.head.prepend(viewport)

  const highlightStyle = documentNode.createElement('style')
  highlightStyle.textContent =
    '::highlight(hermes-annotations), mark[data-hermes-annotation] { background: color-mix(in srgb, Highlight 34%, transparent); color: inherit; }'
  documentNode.head.append(highlightStyle)

  const bridge = documentNode.createElement('script')
  bridge.textContent = BRIDGE_SCRIPT
  documentNode.body.append(bridge)

  return `<!doctype html>${documentNode.documentElement.outerHTML}`
}

export function HtmlAnnotationPreview({
  filePath,
  html,
  contentHash,
  reviewContext,
  panEnabled = false,
  zoom = 100
}: {
  contentHash?: string
  filePath: string
  html: string
  panEnabled?: boolean
  reviewContext?: ReviewContext
  zoom?: number
}) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null)
  const annotations = useStore($annotations)
  const srcDoc = useMemo(() => safeHtmlDocument(html), [html])

  const htmlAnchors = useMemo(
    () =>
      annotations
        .filter(item => item.anchor.kind === 'html' && item.anchor.path === filePath)
        .map(item => (item.anchor.kind === 'html' ? item.anchor : null))
        .filter(anchor => anchor !== null),
    [annotations, filePath]
  )

  const applyHighlights = useCallback(() => {
    iframeRef.current?.contentWindow?.postMessage({ type: MESSAGE_APPLY, anchors: htmlAnchors }, '*')
    iframeRef.current?.contentWindow?.postMessage({ type: MESSAGE_ZOOM, zoom: zoom / 100 }, '*')
    iframeRef.current?.contentWindow?.postMessage({ type: MESSAGE_PAN, enabled: panEnabled }, '*')
  }, [htmlAnchors, panEnabled, zoom])

  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (event.source !== iframeRef.current?.contentWindow || !event.data) {
        return
      }

      if (event.data.type === MESSAGE_READY) {
        applyHighlights()
      } else if (event.data.type === MESSAGE_SELECTION && typeof event.data.text === 'string') {
        const iframeRect = iframeRef.current?.getBoundingClientRect()
        const rect = event.data.rect as { height?: unknown; width?: unknown; x?: unknown; y?: unknown } | undefined

        beginAnnotation(
          {
            contentHash,
            endOffset: typeof event.data.endOffset === 'number' ? event.data.endOffset : String(event.data.text).length,
            kind: 'html',
            path: filePath,
            prefix: typeof event.data.prefix === 'string' ? event.data.prefix : undefined,
            quote: event.data.text,
            endNodeOffset: typeof event.data.endNodeOffset === 'number' ? event.data.endNodeOffset : undefined,
            endPath: Array.isArray(event.data.endPath) ? event.data.endPath : undefined,
            startOffset: typeof event.data.startOffset === 'number' ? event.data.startOffset : 0,
            startNodeOffset: typeof event.data.startNodeOffset === 'number' ? event.data.startNodeOffset : undefined,
            startPath: Array.isArray(event.data.startPath) ? event.data.startPath : undefined,
            suffix: typeof event.data.suffix === 'string' ? event.data.suffix : undefined
          },
          iframeRect && rect && typeof rect.x === 'number' && typeof rect.y === 'number'
            ? {
                boundary: iframeRect
                  ? {
                      height: iframeRect.height,
                      width: iframeRect.width,
                      x: iframeRect.left,
                      y: iframeRect.top
                    }
                  : undefined,
                height: typeof rect.height === 'number' ? rect.height : 0,
                width: typeof rect.width === 'number' ? rect.width : 0,
                x: iframeRect.left + rect.x,
                y: iframeRect.top + rect.y
              }
            : null,
          reviewContext ?? documentReviewContext(filePath, contentHash)
        )
      }
    }

    window.addEventListener('message', onMessage)

    return () => window.removeEventListener('message', onMessage)
  }, [applyHighlights, contentHash, filePath, reviewContext])

  useEffect(applyHighlights, [applyHighlights])

  return (
    <div className="absolute inset-0 min-h-0 min-w-0 overflow-hidden" data-annotation-surface>
      <iframe
        className="absolute inset-0 block h-full w-full border-0 bg-white"
        onLoad={applyHighlights}
        ref={iframeRef}
        sandbox="allow-scripts"
        srcDoc={srcDoc}
        title={filePath}
      />
    </div>
  )
}

export const sanitizeHtmlAnnotationDocument = safeHtmlDocument
