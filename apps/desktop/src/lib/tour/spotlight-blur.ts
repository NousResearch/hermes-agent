/**
 * SPOTLIGHT BLUR — a blurred scrim that keeps the highlighted element crisp.
 *
 * driver.js paints its scrim as ONE full-viewport <svg> whose fill path has the
 * spotlight punched out. A `backdrop-filter` on that element blurs the cutout
 * too, because backdrop-filter clips to the element's BOX, not its fill — which
 * is why the highlighted control came out fuzzy.
 *
 * So the blur goes on its own layer, masked to everything-except-the-spotlight.
 * Two details make it not look cheap:
 *
 *   - The mask is an SVG <mask> whose hole is a ROUNDED rect run through a
 *     feGaussianBlur, so the blur fades out across a soft band instead of
 *     ending on a hard edge. (`clip-path` can't do this — it's binary. A
 *     data-URI mask can, but rebuilding and reparsing that string every frame
 *     is wasteful; an inline mask lets a frame be four attribute writes.)
 *   - The hole follows driver.js's OWN animated rect (`__activeStagePosition`,
 *     eased per frame by driver itself) rather than the target's static box,
 *     so the blur travels with the cutout between steps instead of jumping to
 *     the destination while the cutout is still gliding.
 *
 * App surface only — a preview guest page keeps driver.js's stock look.
 */

/** Matches DialogOverlay's scrim treatment. The actual effect (grayscale/fade)
 *  lives in app-tour.css so it can be tuned in one place. */
const FEATHER_PX = 12

/** Matches TOUR_STYLE.stagePadding/stageRadius, so the soft hole sits over
 *  driver's own cutout rather than beside it. */
const STAGE_PADDING = 6
const STAGE_RADIUS = 8

const LAYER_ID = '__hermes-tour-spotlight-blur'
const MASK_ID = '__hermes-tour-spotlight-mask'
const SVG_NS = 'http://www.w3.org/2000/svg'

/** One rect's worth of geometry — driver's stage, or an element's box. */
export interface Stage {
  height: number
  width: number
  x: number
  y: number
}

/** Reads driver.js's live (animated) stage rect. Supplied by run-tour. */
type StageSource = () => null | Stage

let frame = 0
let stageSource: StageSource | undefined
let hole: SVGRectElement | undefined
let outer: SVGRectElement | undefined

function teardownDom(): void {
  document.getElementById(LAYER_ID)?.remove()
  document.getElementById(`${LAYER_ID}-defs`)?.remove()
  hole = undefined
  outer = undefined
}

/** The blur layer plus the mask that shapes it. Built once per tour. */
function mount(): void {
  if (document.getElementById(LAYER_ID)) {
    return
  }

  const svg = document.createElementNS(SVG_NS, 'svg')

  svg.id = `${LAYER_ID}-defs`
  svg.setAttribute('aria-hidden', 'true')
  svg.style.cssText = 'position:fixed;width:0;height:0;pointer-events:none'

  const mask = document.createElementNS(SVG_NS, 'mask')

  mask.id = MASK_ID
  // Viewport coordinates, so the rects below take plain CSS pixels.
  mask.setAttribute('maskUnits', 'userSpaceOnUse')

  const filter = document.createElementNS(SVG_NS, 'filter')

  filter.id = `${MASK_ID}-feather`

  // Roomy region: a blur clipped to the shape's own box would square off the
  // very edge we're trying to soften.
  for (const [name, value] of [
    ['x', '-50%'],
    ['y', '-50%'],
    ['width', '200%'],
    ['height', '200%']
  ]) {
    filter.setAttribute(name, value)
  }

  const blur = document.createElementNS(SVG_NS, 'feGaussianBlur')

  blur.setAttribute('stdDeviation', String(FEATHER_PX / 2))
  filter.appendChild(blur)

  // White shows the blur, black hides it: paint everything, then punch a
  // feathered hole where the spotlight is.
  outer = document.createElementNS(SVG_NS, 'rect')
  outer.setAttribute('fill', 'white')

  hole = document.createElementNS(SVG_NS, 'rect')
  hole.setAttribute('fill', 'black')
  hole.setAttribute('rx', String(STAGE_RADIUS + STAGE_PADDING / 2))
  hole.setAttribute('filter', `url(#${filter.id})`)

  mask.append(outer, hole)
  svg.append(filter, mask)
  document.body.appendChild(svg)

  const layer = document.createElement('div')

  layer.id = LAYER_ID
  layer.setAttribute('aria-hidden', 'true')
  layer.style.cssText = [
    'position:fixed',
    'inset:0',
    'pointer-events:none',
    // Under driver.js's own overlay (10000) so the scrim tint reads on top of
    // this, and far under the popover, which stays perfectly sharp.
    'z-index:9998',
    `mask:url(#${MASK_ID})`,
    `-webkit-mask:url(#${MASK_ID})`
  ].join(';')

  document.body.appendChild(layer)
}

/** Move the feathered hole onto `stage`. Four attribute writes. */
function paint(stage: null | Stage): void {
  if (!hole || !outer) {
    return
  }

  outer.setAttribute('width', String(window.innerWidth))
  outer.setAttribute('height', String(window.innerHeight))

  if (!stage) {
    // A narration step highlights nothing: blur everything, no hole.
    hole.setAttribute('width', '0')
    hole.setAttribute('height', '0')

    return
  }

  hole.setAttribute('x', String(stage.x - STAGE_PADDING))
  hole.setAttribute('y', String(stage.y - STAGE_PADDING))
  hole.setAttribute('width', String(Math.max(0, stage.width + STAGE_PADDING * 2)))
  hole.setAttribute('height', String(Math.max(0, stage.height + STAGE_PADDING * 2)))
}

/** Follow the stage every frame — driver eases it, we just mirror it. */
function tick(): void {
  frame = requestAnimationFrame(tick)

  if (!document.querySelector('svg.driver-overlay')) {
    // The tour ended (last step, Esc, ✕, overlay click).
    stopSpotlightBlur()

    return
  }

  mount()
  paint(stageSource?.() ?? null)
}

/**
 * Track the live spotlight with a blurred, soft-edged layer. Idempotent: safe
 * to call on every tour action. Everything it creates dies with driver.js's
 * overlay, however the tour ended, so nothing lingers between tours.
 */
export function syncSpotlightBlur(stage: StageSource): void {
  stageSource = stage

  if (!frame) {
    frame = requestAnimationFrame(tick)
  }
}

/** Stop following and remove the layer. */
export function stopSpotlightBlur(): void {
  if (frame) {
    cancelAnimationFrame(frame)
    frame = 0
  }

  stageSource = undefined
  teardownDom()
}
