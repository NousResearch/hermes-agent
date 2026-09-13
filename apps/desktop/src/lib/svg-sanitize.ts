import DOMPurify from 'dompurify'

const FORBIDDEN_RESOURCE_TAGS = [
  'animate',
  'animateMotion',
  'animateTransform',
  'audio',
  'cursor',
  'discard',
  'filter',
  'foreignObject',
  'iframe',
  'image',
  'mpath',
  'script',
  'set',
  'style',
  'use',
  'video'
]

const FORBIDDEN_RESOURCE_ATTRIBUTES = [
  'cursor',
  'externalResourcesRequired',
  'filter',
  'href',
  'src',
  'style',
  'xlink:href',
  'xml:base'
]

// Local clip paths, masks, markers, and paint servers are safe references to
// inert geometry and gradients retained below. Filters are intentionally
// asymmetric: the DOMPurify profile disables the complete SVG filter graph,
// so both <filter> and filter= are banned rather than leaving dead references
// or making future filter-profile expansion silently security-sensitive.

const LOCAL_REFERENCE_RE = /^url\(\s*['"]?#[A-Za-z_][\w:.-]*['"]?\s*\)$/i
const LOCAL_REFERENCE_ATTRIBUTES = new Set(['clip-path', 'marker', 'marker-end', 'marker-mid', 'marker-start', 'mask'])
const PAINT_ATTRIBUTES = new Set(['fill', 'stroke'])
const SAFE_REFERENCE_KEYWORDS = new Set(['inherit', 'initial', 'none', 'revert', 'revert-layer', 'unset'])

function hasUnsafeResourceValue(name: string, value: string): boolean {
  const normalizedName = name.toLowerCase()
  const normalizedValue = value.trim()

  if (LOCAL_REFERENCE_ATTRIBUTES.has(normalizedName)) {
    return !LOCAL_REFERENCE_RE.test(normalizedValue) && !SAFE_REFERENCE_KEYWORDS.has(normalizedValue.toLowerCase())
  }

  if (!PAINT_ATTRIBUTES.has(normalizedName)) {
    return false
  }

  if (LOCAL_REFERENCE_RE.test(normalizedValue)) {
    return false
  }

  // CSS escapes and comments can disguise the `url` token. Paint values that
  // contain either are removed rather than handed to Chromium's CSS parser.
  return /url|\\|\/\*/i.test(normalizedValue)
}

/**
 * Sanitize model-authored SVG and then apply a stricter no-resource policy.
 *
 * DOMPurify provides the markup/XSS boundary. The explicit second layer is
 * required because the SVG profile intentionally permits resource-capable SVG
 * features (for example image/use/style/filter and CSS url()). Assistant SVG
 * may draw local shapes, text, and gradients, but it may not load any URL.
 */
export function sanitizeSvgMarkup(code: string): string {
  const purified = DOMPurify.sanitize(code, {
    FORBID_ATTR: FORBIDDEN_RESOURCE_ATTRIBUTES,
    FORBID_TAGS: FORBIDDEN_RESOURCE_TAGS,
    USE_PROFILES: { svg: true, svgFilters: false }
  })

  const template = document.createElement('template')

  template.innerHTML = String(purified)

  const topLevelElements = Array.from(template.content.children)

  if (topLevelElements.length === 0 || topLevelElements.some(element => element.localName !== 'svg')) {
    return ''
  }

  for (const element of template.content.querySelectorAll('*')) {
    for (const attribute of Array.from(element.attributes)) {
      if (hasUnsafeResourceValue(attribute.name, attribute.value)) {
        element.removeAttribute(attribute.name)
      }
    }
  }

  return template.innerHTML
}
