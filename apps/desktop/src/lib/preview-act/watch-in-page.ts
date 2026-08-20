/**
 * WATCH OVERLAY — paints the agent's actions onto the live preview page so a
 * person can follow along: the field of things it can reach, a box round the
 * one it is about to touch, and a cursor that goes there.
 *
 * There are exactly two things on screen, and everything the agent does is said
 * with one of them:
 *
 *   - THE CURSOR — one arrow, built once, never removed. It is the
 *     collaborative-editing kind, because that is exactly what this is: a second
 *     party moving around a document you are also looking at. Arrow geometry is
 *     Liveblocks' (Apache-2.0). Plain black or white against the guest's own
 *     colour scheme rather than a brand colour: it stands for a hand, and a hand
 *     is not a feature of this app.
 *   - THE MARK — a rectangle glued to an element. A candidate, a lock, a hit
 *     and a pin are all the same object with a different row in `kinds`: how
 *     heavy the rule is, how long it lives, how it arrives. Marks are a design
 *     tool's selection blue, deliberately NOT the
 *     app's accent — this is chrome drawn on top of someone else's page, and
 *     every design tool has trained people to read that blue as "a tool has
 *     selected this", where a branded colour would read as part of the page.
 *
 * Every mark is two elements: a shell the tracker positions, and a skin that
 * carries the border and any animation. They are split because both want
 * `transform`, and a mark that animates the property the tracker rewrites every
 * frame either stops moving or stops animating.
 *
 * Injected as source (`watchInPage.toString()`) exactly like the act engine, so
 * the same rule holds: no imports, no closure references, no module-scope
 * constants — everything the body needs is declared inside it.
 *
 * Three constraints shape the implementation, each one learned from how
 * Playwright, Stagehand and browser-use solve the same problem:
 *
 *   - A page can set `z-index: 2147483647` too, and the browser's top layer
 *     (`<dialog>`, `popover`, fullscreen) outranks every z-index there is. So
 *     the host joins the top layer via `popover="manual"` instead of trying to
 *     outbid the page, and keeps the z-index only as a fallback.
 *   - A strict CSP rejects `style-src` at CSS PARSE time, which kills
 *     `cssText`, `setAttribute('style', …)` and any `<style>` we inject — but
 *     never `style.setProperty()` or `element.animate()`. So every style here
 *     is set per-property, and everything that moves is a Web Animation.
 *   - The overlay must stay out of the agent's own element inventory. It lives
 *     in a closed shadow root, which `document.querySelectorAll` does not
 *     descend into, so `collect()` cannot reach it — no exclusion list to keep
 *     in sync, and page scripts can't reach in either.
 */

/** Where the overlay reads its target from — the act engine's holder. */
export interface WatchHolder {
  /** The element the agent is currently working on, parked by `locate`. */
  aimed?: Element | null
  /** Every handle the agent holds on this page, so a mark can be labelled with
   *  the name the agent actually calls that element by. */
  book?: { el: Element; ref: string }[]
  /** What is on screen right now, for the field to outline. */
  field?: Element[]
  /** The addressable subset, in document order. */
  nodes?: Element[]
}

/**
 * - `aim` — box the target and send the cursor to it (the agent is looking).
 * - `strike` — mark the target (the agent just acted).
 * - `sweep` — outline and name the whole field, scattered.
 * - `strobe` — rattle through the field one box at a time, out of order.
 * - `read` — wipe down the page's text, in the order a person would read it.
 * - `think` / `rest` — start and stop the idle pulse between actions.
 * - `clear` — pack the transients away. Pins survive it.
 * - `pin` / `unpin` — put up a mark that outlives the action, or take it down.
 * - `hold` — promote the whole field to pins, so the scan stops decaying.
 */
export type WatchStage =
  | 'aim'
  | 'clear'
  | 'hold'
  | 'pin'
  | 'read'
  | 'rest'
  | 'strike'
  | 'strobe'
  | 'sweep'
  | 'think'
  | 'unpin'

/** What a mark means. Everything else about it is shared. */
type WatchKind = 'candidate' | 'flash' | 'held' | 'hit' | 'lock'

interface WatchSpec {
  /** The element the mark is glued to, re-measured every frame. */
  a: Element
  /** Stagger, in ms, applied to both ends of its life. */
  beat?: number
  kind: WatchKind
  label?: string
  /** Overrides the kind's hold time, for marks that want an uneven rhythm. */
  life?: number
}

/** A spec plus whatever is on screen for it. The elements are optional because
 *  a pin outlives the shadow tree it was drawn into and gets redrawn. */
interface WatchMark {
  cell?: HTMLElement
  skin?: HTMLElement
  /** Which corner the label is hanging off, so it is only re-styled on a flip. */
  slot?: string
  spec: WatchSpec
  tag?: HTMLElement
  /** The label's measured width, kept so the fit pass doesn't force a layout. */
  wide?: number
}

/** An element's rectangle in viewport coordinates. */
interface WatchSpot {
  height: number
  left: number
  top: number
  width: number
}

/** Draw one stage of the agent's action. `label` is only read by `pin`.
 *  Self-contained. */
export function watchInPage(doc: Document, holder: WatchHolder, stage: WatchStage, label?: string): void {
  const win = doc.defaultView as (Window & { __hermesWatch?: Record<string, unknown> }) | null

  if (!win || !doc.documentElement) {
    return
  }

  // Process magenta. A selection blue is what every design tool uses, which is
  // the problem: half the pages this gets drawn over are already blue, and the
  // marks sink into them. Magenta is the one hue almost nothing on the web is,
  // so the chrome reads as chrome. Not #FF00FF — pure magenta is too light to
  // carry the white label text. One colour, deliberately: a hue per element was
  // tried and a page of marks in eight colours is unreadable — the outlines stop
  // reading as one tool's chrome and start competing with the content.
  const line = '#EC008C'
  // Field outlines are the hairline; whatever is actually locked gets the
  // heavier one, so the target is legible in a screen full of candidates.
  const weight = '1.5px'
  const heavy = '2px'
  const pad = 4
  // The label's side padding. The tab itself sits flush with the outline — the
  // two are the same colour and meet at a corner, so any offset between their
  // edges reads as a misprint. That puts the glyphs just inside the stroke,
  // which is where the box's own content starts anyway.
  const nudge = 2
  // How long a mark stays up. Far longer than the action that made it: the
  // input is instant on purpose, and a cue that lasts as long as the input is
  // one nobody can read. Marks hold at full strength and then cut.
  const dwell = 6_000
  // The window the field's arrivals and departures are spread across.
  const cascade = 900
  // The lock is the long one — a whole action happens underneath it.
  const hold = 7_500
  // The strobe. Every number here is a floor with a random span on top, because
  // an even interval is the one thing that stops this reading as a machine
  // chewing through a page — a fixed tempo is a progress bar. Hits come in
  // clumps of one to three, most gaps are inside a tenth of a second, and every
  // so often it stalls for a quarter of one, which is the beat that sells it.
  const blink = 90
  const flick = 200
  const tempo = 26
  const gasp = 210
  const burst = 5_000
  // The read. A beam down the page and a short hold under each block: reading
  // is the one thing the agent does to a page in an order a person could
  // follow, so it is the one stage that is allowed to be a wipe. Holding the
  // blocks the way a sweep holds its field would bury the text under the very
  // outlines that are supposed to be pointing at it.
  const wipe = 1_800
  const glance = 1_400
  // A ceiling on how many blocks one pass schedules. Past this the beam is
  // solid anyway and the only thing another hundred animations buys is jank.
  const page = 160
  // The idle pulse, which runs while the model is reasoning between actions.
  // Far sparser than the strobe: this is the page being held in mind, not read.
  // Most ticks fire nothing, which is what keeps it from reading as a spinner.
  const throb = 260
  const ponder = 240
  const quiet = 0.55

  // Big animated overlays are a vestibular hazard. Honour the user's setting by
  // keeping the marks and dropping the motion, rather than showing nothing.
  const still = !!(win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches)
  // The arrow reads against the PAGE, so it follows the guest's scheme, not the
  // app's: white on a dark page, black on a light one.
  const dark = !!(win.matchMedia && win.matchMedia('(prefers-color-scheme: dark)').matches)
  const ink = dark ? '#FFFFFF' : '#111111'
  // A grotesque, not the app's mono. Mono was the wrong instinct at label size:
  // its glyphs are uniformly wide, so the same string takes about a third more
  // box, and a field of tags that each overhang their own rectangle is most of
  // what makes a dense page unreadable. Installed OS faces only — the guest page
  // is a different document and never loads our @font-face, so naming a bundled
  // family here would just be a step the browser skips.
  const face = "-apple-system, 'Helvetica Neue', Arial, sans-serif"
  // Past this many boxes a field is a texture, not a list, and naming every one
  // of them buries the outlines under a wall of colliding tags. The outlines are
  // the readout at that size; the names are noise.
  const busy = 25
  const state = win.__hermesWatch || (win.__hermesWatch = {})

  // One row per kind of mark. `life` is how long it stays up; 0 means it stays
  // until the agent takes it down, which is the whole difference between a mark
  // that narrates an action and one that records a finding.
  const kinds = {
    candidate: { gap: 0, life: dwell, rule: weight },
    flash: { gap: pad, life: blink, rule: heavy },
    held: { gap: pad, life: 0, rule: heavy },
    hit: { gap: 0, life: dwell, rule: heavy },
    lock: { gap: pad, life: hold, rule: heavy }
  }

  const style = (el: Element, props: Record<string, string>) => {
    for (const name in props) {
      ;(el as HTMLElement).style.setProperty(name, props[name])
    }
  }

  // ── NAMING ────────────────────────────────────────────────────────────────

  /** Name one node: the most specific thing about it that fits in a few
   *  characters. Position is the last resort but it is the one that saves the
   *  label — a page of anonymous links is a page of boxes all reading "a". */
  const step = (node: Element) => {
    const tag = node.tagName.toLowerCase()
    const id = node.getAttribute('id')
    const cls = (node.getAttribute('class') || '').trim().split(/\s+/)[0]
    const kind = node.getAttribute('type') || node.getAttribute('name') || node.getAttribute('role')

    if (id) {
      return tag + '#' + id
    }

    if (cls) {
      return tag + '.' + cls
    }

    if (kind) {
      return tag + '[' + kind + ']'
    }

    const parent = node.parentElement

    if (!parent) {
      return tag
    }

    let kin = 0
    let mine = 0

    for (let i = 0; i < parent.children.length; i++) {
      if (parent.children[i].tagName !== node.tagName) {
        continue
      }

      kin++

      if (parent.children[i] === node) {
        mine = kin
      }
    }

    return kin > 1 ? tag + '[' + mine + ']' : tag
  }

  /** A short name for the element.
   *
   *  The agent's own handle wins whenever it has one, because then the label on
   *  the box and the word in the transcript are the same string — someone
   *  watching can read `btn-sign-in` on screen and find that exact line in what
   *  the model said. Falling back to the tag is for everything the agent has no
   *  handle for: the field the overlay outlines is far wider than the inventory
   *  the agent is given.
   *
   *  The fallback is the leaf only. An ancestor path was tried and it is worse
   *  than nothing on a real page: three rungs of `span.titleline/a` cut to fit a
   *  box gives `…T/SPAN.SUBLINE/A` — wider than the box it sits on, severed
   *  mid-word, and identical to its neighbour's. */
  const sniff = (el: Element) => {
    const held = holder.book || []

    for (const bound of held) {
      if (bound.el === el) {
        return bound.ref
      }
    }

    const name = step(el)

    return name.length > 16 ? name.slice(0, 15) + '…' : name
  }

  // ── GEOMETRY ──────────────────────────────────────────────────────────────

  const spot = (el: Element): WatchSpot => {
    const box = el.getBoundingClientRect()

    return { height: box.height, left: box.left, top: box.top, width: box.width }
  }

  const middle = (box: WatchSpot) => ({ x: box.left + box.width / 2, y: box.top + box.height / 2 })

  // ── THE CURSOR ────────────────────────────────────────────────────────────

  const build = () => {
    // A host from an earlier build of this function may still be on the page.
    // Clear it out rather than stacking a second.
    const stale = doc.querySelectorAll('hermes-watch')

    for (let i = 0; i < stale.length; i++) {
      stale[i].remove()
    }

    const host = doc.createElement('hermes-watch')

    host.setAttribute('aria-hidden', 'true')
    host.setAttribute('popover', 'manual')

    // The popover UA sheet ships its own border/padding/background and sizes to
    // fit-content, so every one of those has to be flattened back out.
    style(host, {
      background: 'transparent',
      border: '0',
      contain: 'layout style paint',
      display: 'block',
      height: '100%',
      inset: '0',
      margin: '0',
      overflow: 'visible',
      padding: '0',
      'pointer-events': 'none',
      position: 'fixed',
      'user-select': 'none',
      width: '100%',
      'z-index': '2147483647'
    })

    const shadow = host.attachShadow({ mode: 'closed' })
    const pointer = doc.createElement('div')

    // Inherited from the host, but stated outright here too: an overlay that
    // swallows the agent's own click is a very quiet failure.
    style(pointer, { left: '0', opacity: '0', 'pointer-events': 'none', position: 'fixed', top: '0' })

    // Re-mounting the host must not teleport the cursor back to the corner, so
    // the last spot is restored before it is ever shown.
    const was = state.at as { x: number; y: number } | undefined

    if (was) {
      style(pointer, { transform: 'translate3d(' + was.x + 'px,' + was.y + 'px,0)' })
    }

    const cursor = doc.createElementNS('http://www.w3.org/2000/svg', 'svg')

    cursor.setAttribute('viewBox', '0 0 24 36')
    cursor.setAttribute('width', '28')
    cursor.setAttribute('height', '42')
    // The arrow's tip is at (0.5, 1.2) in viewBox units, ~(0.6, 1.4)px at this
    // scale, so pull it back to put the POINT on the target.
    style(cursor, { display: 'block', left: '-1px', position: 'absolute', top: '-1px' })

    const nib = doc.createElementNS('http://www.w3.org/2000/svg', 'path')

    nib.setAttribute(
      'd',
      'M5.65376 12.3673H5.46026L5.31717 12.4976L0.500002 16.8829L0.500002 1.19841L11.7841 12.3673H5.65376Z'
    )
    cursor.appendChild(nib)
    pointer.appendChild(cursor)
    shadow.appendChild(pointer)
    doc.documentElement.appendChild(host)

    // Idle drift. A cursor parked at exactly the same pixel for thirty seconds
    // reads as a screenshot; a couple of pixels of wander reads as something
    // waiting its turn. It rides the SVG rather than the wrapper because the
    // wrapper's transform is rewritten every frame by the tracking loop.
    if (!still && typeof cursor.animate === 'function') {
      cursor.animate(
        [
          { transform: 'translate(0px, 0px)' },
          { transform: 'translate(1.5px, -2px)' },
          { transform: 'translate(-1px, 1.5px)' },
          { transform: 'translate(0px, 0px)' }
        ],
        { duration: 5_200, easing: 'ease-in-out', iterations: Infinity }
      )
    }

    return { cursor, host, nib, pointer, shadow }
  }

  /** Put the hand somewhere and remember where that was. The cursor is never
   *  interpolated: beyond reading as a machine, a transition here was an outright
   *  bug on the first paint — an unset transform IS the origin, so the browser
   *  flew the cursor in from the top-left corner of every fresh document. */
  const park = (at: { x: number; y: number }) => {
    const shown = state.parts as ReturnType<typeof build> | undefined

    state.at = at

    if (shown) {
      style(shown.pointer, { transform: 'translate3d(' + at.x + 'px,' + at.y + 'px,0)' })
    }
  }

  // ── THE MARK ──────────────────────────────────────────────────────────────

  const marks = (state.marks || (state.marks = [])) as WatchMark[]

  /** The overlay has one kind of label, so a pinned box and a swept box read the
   *  same: a tab sitting on the box's top edge, exactly where a design tool puts
   *  one. Filled, not coloured-glyph-on-whatever-is-under-it — these sit
   *  directly on the page's own text, and an unbacked glyph there is illegible
   *  against half the pages on the web, reading as a rendering fault rather than
   *  a label. Solid, and the same colour as the outline it hangs off. */
  const caption = (text: string) => {
    const tag = doc.createElement('div')

    tag.textContent = text
    style(tag, {
      // Named outright, NOT `currentColor`. `currentColor` in a background
      // resolves against the element's own `color` — which is white here — so
      // the label came out white on white and vanished.
      background: line,
      color: '#FFFFFF',
      'font-family': face,
      'font-size': '8px',
      'font-weight': '700',
      // Tracked OUT, not in. Tightening is for display sizes; at 8px, and in
      // caps especially — no ascenders or descenders to separate one letter from
      // the next — the glyphs need air or they read as a smear.
      'letter-spacing': '0.04em',
      'line-height': '1.4',
      // Capped in pixels, NOT at 100% of the box. A tab is allowed to be wider
      // than the thing it names — an icon button is 32px and would have had its
      // label clipped to three letters, which is worse than an overhang. The
      // ceiling is only here for a pin whose caption the agent wrote.
      'max-width': '160px',
      overflow: 'hidden',
      padding: '0 ' + nudge + 'px',
      position: 'absolute',
      'text-transform': 'uppercase',
      'white-space': 'nowrap'
    })

    return tag
  }

  const paint = (spec: WatchSpec) => {
    const cell = doc.createElement('div')

    // The shell holds position and the colour, and nothing else. The tracker
    // rewrites its transform every frame, so anything that animates a transform
    // of its own has to be a layer inside it — and everything below inherits the
    // colour through `currentColor`, so a mark is tinted in exactly one place.
    style(cell, { color: line, left: '0', 'pointer-events': 'none', position: 'fixed', top: '0' })

    const skin = doc.createElement('div')

    style(skin, {
      border: kinds[spec.kind].rule + ' solid currentColor',
      inset: '0',
      overflow: 'hidden',
      position: 'absolute',
      'transform-origin': 'center'
    })
    cell.appendChild(skin)

    // Only the lock is scanned. It is the one mark that stands for the agent
    // reading something rather than having already found it.
    if (spec.kind === 'lock') {
      const beam = doc.createElement('div')

      style(beam, {
        background: 'currentColor',
        height: heavy,
        left: '0',
        opacity: '0',
        position: 'absolute',
        right: '0',
        top: '0'
      })
      skin.appendChild(beam)
    }

    // On the shell rather than the skin, so a label neither scales with the
    // entrance nor gets clipped by the skin's own overflow.
    const tag = spec.label ? caption(spec.label) : undefined

    if (tag) {
      cell.appendChild(tag)
    }

    return { cell, skin, tag }
  }

  /** Re-measure a mark against the element it is glued to. Returns its centre,
   *  so the one caller that cares where a mark ended up doesn't measure twice. */
  const fit = (mark: WatchMark) => {
    const cell = mark.cell as HTMLElement
    const from = spot(mark.spec.a)
    const gap = kinds[mark.spec.kind].gap

    style(cell, {
      height: from.height + gap * 2 + 'px',
      transform: 'translate3d(' + (from.left - gap) + 'px,' + (from.top - gap) + 'px,0)',
      width: from.width + gap * 2 + 'px'
    })

    // Which corner the tab hangs off. Decided here rather than at paint because
    // scrolling is what runs a box out of headroom, and re-styled only on the
    // frames where the answer actually changed.
    if (mark.tag) {
      // Above by default, below when the box is against the top of the viewport,
      // and inside its top edge only when it is pinned against both — which is a
      // box taller than the screen. Below beats inside because a tab laid over
      // the box covers the very thing it is naming.
      const tall = 13
      const row = from.top - gap >= tall ? 'up' : from.top + from.height + gap + tall <= win.innerHeight ? 'down' : 'in'
      // Measured once and kept: the text never changes after paint, and an
      // offsetWidth read is a forced layout, which is not something to do per
      // mark per frame.
      const wide = mark.wide || (mark.wide = mark.tag.offsetWidth)
      // Anchored to whichever end of the box keeps the tab on screen. A label is
      // routinely wider than the box it names, so a left-anchored one on
      // anything near the right edge is the case that runs off.
      const end = from.left - gap + wide > win.innerWidth ? 'end' : 'start'
      const slot = row + end

      if (mark.slot !== slot) {
        mark.slot = slot
        style(mark.tag, {
          bottom: row === 'up' ? '100%' : '',
          left: end === 'start' ? '0px' : '',
          right: end === 'end' ? '0px' : '',
          top: row === 'up' ? '' : row === 'down' ? '100%' : '0'
        })
      }
    }

    return middle(from)
  }

  /** How a mark arrives and when it leaves. Held marks do neither. */
  const enter = (mark: WatchMark) => {
    const spec = mark.spec
    const life = spec.life || kinds[spec.kind].life

    if (!life) {
      return
    }

    const cell = mark.cell as HTMLElement
    const skin = mark.skin as HTMLElement
    const beat = spec.beat || 0

    // Every transient leaves on a clock rather than on its animation, so the
    // reduced-motion path and the animated one retire in the same beat.
    win.setTimeout(() => cell.remove(), life + beat)

    // A mark comes and goes as ONE thing, so what fades is the CELL — which owns
    // the outline and the label both. Fading the skin instead left every tag
    // painted at full strength from the moment the mark was created: through its
    // whole delay, and past the box's exit. On a burst that is a hundred labels
    // up at once over five outlines, each one outliving the box it names.
    const blip = spec.kind === 'candidate' || spec.kind === 'flash'
    const mover = blip ? cell : skin

    if (still || typeof mover.animate !== 'function') {
      return
    }

    // Hard on, hard off. The cut comes from the OFFSETS — full strength is
    // reached in the first percent and held to the last — not from a step
    // easing. `easing` in the options bag is the ITERATION easing, applied to
    // progress before any keyframe is consulted, so `steps(1)` there pins
    // progress at 0 for the whole duration and the mark never paints at all.
    const frames = blip
      ? [
          { offset: 0, opacity: 0 },
          { offset: 0.01, opacity: 1 },
          { offset: 0.99, opacity: 1 },
          { offset: 1, opacity: 0 }
        ]
      : // A box snapping shut on a target, not a material-design tap. Scale is
        // the one thing that CAN'T move to the cell — the tracker rewrites the
        // cell's transform every frame — so it stays on the skin, where it only
        // has to not fight the opacity above.
        [
          { offset: 0, transform: spec.kind === 'hit' ? 'scale(1.35)' : 'scale(1.06)' },
          { offset: spec.kind === 'hit' ? 0.1 : 90 / life, transform: 'scale(1)' },
          { offset: 1, transform: 'scale(1)' }
        ]

    // `both`, not `forwards`. A delayed animation is in its BEFORE phase during
    // the delay, and `forwards` does not fill there — so every staggered box
    // rendered at its own natural opacity the instant it was painted, and only
    // then rewound to animate on its beat. The scatter was being drawn, and the
    // whole field was flashing up as one block before any of it ran.
    mover.animate(frames, { delay: beat, duration: life, easing: 'linear', fill: 'both' })

    // One slow pass down the lock. Repeating a fast sweep put a hard edge across
    // the element several times a second, which is both unreadable and a genuine
    // photosensitivity hazard; a single beam crossing slowly is what reads as
    // scanning anyway.
    const beam = spec.kind === 'lock' ? (skin.firstElementChild as HTMLElement | null) : null

    if (beam && typeof beam.animate === 'function') {
      beam.animate([{ opacity: 1, top: '-2px' }, { opacity: 1, top: '100%' }], {
        duration: dwell,
        easing: 'linear'
      })
    }
  }

  // Marks are glued to elements, not to coordinates, so they are re-measured for
  // as long as they are up. Overlapping fields are wanted — several actions'
  // worth of marks on screen at once IS the readout — but a mark that keeps its
  // original rectangle while the page scrolls under it stops being a mark and
  // becomes litter.
  const tick = () => {
    const shown = state.parts as ReturnType<typeof build> | undefined
    // Compacted in place rather than spliced, so the pass can run forwards: a
    // mark is drawn the first time it is seen here, and drawing them back to
    // front would put a field on screen in reverse.
    let keep = 0

    for (let i = 0; i < marks.length; i++) {
      const mark = marks[i]

      // A navigation or a re-render takes the element with it. The mark goes too
      // rather than hanging over whatever replaced it.
      if (!doc.contains(mark.spec.a)) {
        if (mark.cell) {
          mark.cell.remove()
        }

        continue
      }

      if (mark.cell && !mark.cell.isConnected) {
        // A transient retired itself on its own clock and is done. A held one
        // lost its shadow tree to a rebuild and has to come back.
        if (kinds[mark.spec.kind].life) {
          continue
        }

        mark.cell = undefined
      }

      marks[keep++] = mark

      if (!shown) {
        continue
      }

      if (!mark.cell) {
        const made = paint(mark.spec)

        mark.cell = made.cell
        mark.skin = made.skin
        mark.tag = made.tag
        mark.slot = undefined
        mark.wide = undefined
        shown.shadow.appendChild(made.cell)
        enter(mark)
      }

      const at = fit(mark)

      // The hand rides the lock. One mark owns where the cursor is, and it is
      // the one standing for the thing the agent is touching.
      if (mark.spec.kind === 'lock') {
        park(at)
      }
    }

    marks.length = keep
    state.raf = keep ? win.requestAnimationFrame(tick) : undefined
  }

  /** Draw everything outstanding now, then keep it measured. Restarting the loop
   *  rather than waiting for the next frame is what makes a mark appear on the
   *  call that added it. Called once per stage, never per mark: a pass measures
   *  every mark on screen, and pumping inside `add` would make a field of two
   *  hundred boxes two hundred passes over two hundred rectangles. */
  const pump = () => {
    if (state.raf) {
      win.cancelAnimationFrame(state.raf as number)
      state.raf = undefined
    }

    tick()
  }

  const add = (spec: WatchSpec) => marks.push({ spec })

  /** Outline a whole set. Shared by the sweep, the hold and the read, which
   *  differ only in what the marks are made of and how they are spread. `beat`
   *  is handed the rectangle as well as the index, because a wipe is timed off
   *  where a box IS and a scatter off which one it is. */
  const outline = (all: Element[], kind: WatchKind, beat: (at: number, seen: WatchSpot) => number, life?: number) => {
    for (let i = 0; i < all.length; i++) {
      const seen = spot(all[i])

      if (seen.width < 1 || seen.height < 1) {
        continue
      }

      // Named only when there is room for the tag AND few enough boxes for a
      // name to be worth reading. A hundred outlines is a picture of what the
      // agent can reach; a hundred names on top of it is a wall.
      const named = all.length <= busy && seen.width >= 16 && seen.height >= 10

      add({ a: all[i], beat: beat(i, seen), kind, label: named ? sniff(all[i]) : undefined, life })
    }
  }

  const hush = () => {
    if (state.mind) {
      win.clearInterval(state.mind as number)
      state.mind = undefined
    }
  }

  /** Take down every mark the predicate claims. */
  const retire = (claims: (spec: WatchSpec) => boolean) => {
    for (let i = marks.length - 1; i >= 0; i--) {
      if (!claims(marks[i].spec)) {
        continue
      }

      if (marks[i].cell) {
        ;(marks[i].cell as HTMLElement).remove()
      }

      marks.splice(i, 1)
    }
  }

  // ── THE PAGE'S PROSE ──────────────────────────────────────────────────────

  /** Every block of text on screen, in document order.
   *
   *  Deliberately not a tag list. `p, h1…h6, li, td` misses the single most
   *  common shape on the modern web — a `div` with words directly inside it —
   *  and a list long enough to catch that catches the wrappers too. What
   *  actually separates a block of text from its container is owning a text
   *  node outright, so that is the test. An ancestor that qualifies wins, which
   *  keeps a paragraph whole instead of lighting up every `span` inside it. */
  const prose = () => {
    if (!doc.body) {
      return []
    }

    const all = doc.body.getElementsByTagName('*')
    // The chosen blocks, for the descendants of each to measure themselves
    // against. Document order guarantees an ancestor is decided before anything
    // under it asks, so one pass is enough.
    const bag = new Set<Element>()
    const found: Element[] = []

    for (let i = 0; i < all.length && found.length < page; i++) {
      const el = all[i]
      const tag = el.tagName

      if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' || tag === 'TEMPLATE' || tag === 'OPTION') {
        continue
      }

      let words = false

      for (let node = el.firstChild; node; node = node.nextSibling) {
        if (node.nodeType === 3 && (node.nodeValue || '').trim()) {
          words = true
          break
        }
      }

      if (!words) {
        continue
      }

      let up = el.parentElement
      let inside = false

      while (up) {
        if (bag.has(up)) {
          inside = true
          break
        }

        up = up.parentElement
      }

      if (inside) {
        continue
      }

      bag.add(el)

      // Catches what a rectangle can't: `opacity: 0`, `visibility: hidden`, and
      // anything an ancestor has hidden. Guarded because it is recent enough
      // that the test environment doesn't have it.
      const look = (el as Element & { checkVisibility?: (opts?: object) => boolean }).checkVisibility

      if (typeof look === 'function' && !look.call(el, { opacityProperty: true, visibilityProperty: true })) {
        continue
      }

      const seen = spot(el)

      // On screen and big enough to be text. Off-viewport blocks are most of a
      // long document and outlining them draws nothing anyone can see.
      if (
        seen.width < 16 ||
        seen.height < 6 ||
        seen.left + seen.width <= 0 ||
        seen.top + seen.height <= 0 ||
        seen.left >= win.innerWidth ||
        seen.top >= win.innerHeight
      ) {
        continue
      }

      found.push(el)
    }

    return found
  }

  // ── STAGES ────────────────────────────────────────────────────────────────

  // The cursor is built once and reused for the life of the page. That memo has
  // to break when the code that built it changes, or an edit to its styles is
  // injected, runs, and restyles nothing: the layers it describes were built by
  // the previous version and are still sitting on the page. Same failure across
  // an app upgrade under a long-lived tab; it just looks like "HMR isn't picking
  // this up" in dev.
  const stamp = (win as unknown as { __hermesWatchTag?: number }).__hermesWatchTag

  if (state.tag !== stamp) {
    state.tag = stamp
    state.parts = undefined
    // The pulse outlives an injection, so a rebuild has to stop the old one —
    // its ticks would keep firing against the overlay it was born in.
    hush()
  }

  const mounted = state.parts as ReturnType<typeof build> | undefined
  const target = holder.aimed

  if (stage === 'clear' || stage === 'rest') {
    hush()

    if (stage === 'rest') {
      // The turn is over, not the session. Whatever is up keeps its own clock —
      // the point of `rest` is only that the page stops being thought about.
      return
    }

    holder.aimed = null
    retire(spec => !!kinds[spec.kind].life)

    if (mounted) {
      style(mounted.pointer, { opacity: '0' })
    }

    return
  }

  // The gap between actions is most of the time the agent spends on a page, and
  // the overlay used to go dark for all of it — the marks decayed and nothing
  // said the page was still being held in mind. This taps one element at a time,
  // sparsely and unevenly, for as long as the turn runs.
  //
  // Started BEFORE the there-is-nothing-to-draw bail, because a turn begins when
  // the user hits send and the agent has usually not read the page yet: gating
  // the pulse on a field that only exists after the first inventory would mean
  // it never starts on the turn that matters. The tick is the guard instead — it
  // no-ops until there is an overlay and a field, so nothing is grafted onto a
  // page the agent never touched.
  if (stage === 'think') {
    // Called through `state.pulse` rather than closing over this invocation, so
    // an interval started by an older injection still runs the newest code.
    state.pulse = () => {
      const live = (holder.field && holder.field.length ? holder.field : holder.nodes || []) as Element[]

      if (!live.length || !state.parts || Math.random() < quiet) {
        return
      }

      const el = live[Math.floor(Math.random() * live.length)]
      const seen = spot(el)

      if (seen.width >= 1 && seen.height >= 1) {
        add({ a: el, kind: 'flash', label: sniff(el), life: ponder + Math.random() * ponder })
        pump()
      }
    }

    if (!state.mind && !still) {
      state.mind = win.setInterval(() => {
        const beat = state.pulse as (() => void) | undefined

        if (beat) {
          beat()
        }
      }, throb)
    }

    return
  }

  // Taking a pin down needs no overlay of its own — and with no target it means
  // all of them, so it runs before the there-is-nothing-to-draw bail.
  if (stage === 'unpin') {
    retire(spec => spec.kind === 'held' && (!target || spec.a === target))
    holder.aimed = null

    return
  }

  // These draw the whole field rather than one target.
  const wide = stage === 'sweep' || stage === 'hold' || stage === 'strobe'
  const field = holder.field && holder.field.length ? holder.field : holder.nodes || []
  // The read draws the page's PROSE instead. The two sets barely overlap: what
  // read_preview takes away is the text, and what the field is made of is the
  // controls. Gathered here rather than in the branch so the bail below can see
  // whether there was anything to read.
  const text = stage === 'read' ? prose() : []

  // Nothing to draw on. Bail before building, so a stage with an empty set
  // never gets an overlay host grafted onto a page for nothing.
  if (stage === 'read' ? !text.length : wide ? !field.length : !target || !doc.contains(target)) {
    return
  }

  const parts = mounted && mounted.host.isConnected ? mounted : build()

  state.parts = parts
  parts.nib.setAttribute('fill', ink)

  // Re-assert on every stage: `showPopover` appends to the BACK of the top
  // layer, so a dialog the page opened after us would otherwise cover the
  // overlay for the rest of the session.
  try {
    ;(parts.host as HTMLElement & { hidePopover: () => void }).hidePopover()
    ;(parts.host as HTMLElement & { showPopover: () => void }).showPopover()
  } catch {
    // No popover support, or the host is detached — the z-index fallback stands.
  }

  if (stage === 'pin') {
    // Pinning something already pinned relabels it, rather than stacking a
    // second box on the same rectangle.
    retire(spec => spec.kind === 'held' && spec.a === target)
    add({ a: target as Element, kind: 'held', label: label || sniff(target as Element) })
    holder.aimed = null
  } else if (stage === 'hold') {
    // Freeze the whole field: the same picture a sweep paints, promoted to pins
    // so it stops decaying. Everything a sweep draws is on a clock by design —
    // it narrates an action and gets out of the way — and this is the escape
    // hatch for when the picture IS the answer and has to stay up to be looked
    // at. Held twice means held once: re-running it after the page moved on
    // replaces the picture rather than layering a second over the first.
    retire(spec => spec.kind === 'held')
    outline(field, 'held', () => 0)
    holder.aimed = null
  } else if (stage === 'sweep') {
    // Only the lock goes. The previous field is deliberately left to run out its
    // own clock underneath this one — several sweeps' worth of boxes on screen
    // at once is the readout, not a glitch, and they stay honest because every
    // one of them is still tracking its own element.
    retire(spec => spec.kind === 'lock')
    holder.aimed = null
    // Scattered, not top-to-bottom. A document-order cascade reads as a wipe
    // passing over the page; firing all over it at once reads as the whole page
    // being taken in at once, which is what actually happened. Deterministic, so
    // a re-run of the same page looks the same twice.
    outline(field, 'candidate', at => {
      const noise = Math.sin((at + 1) * 12.9898) * 43758.5453

      return (noise - Math.floor(noise)) * cascade
    })
  } else if (stage === 'read') {
    // A beam down the page. Every other stage scatters on purpose — an
    // inventory happens all at once and saying it in document order would be a
    // lie about what the agent did. Reading is the exception: it genuinely goes
    // top to bottom, so the cue does too, and the two are told apart at a
    // glance. Timed off where a block SITS rather than its index, so two
    // columns light together and the beam stays straight.
    retire(spec => spec.kind === 'lock')
    holder.aimed = null
    outline(
      text,
      'candidate',
      (_at, seen) => Math.max(0, Math.min(1, (seen.top + seen.height / 2) / win.innerHeight)) * wipe,
      glance
    )
  } else if (stage === 'strobe') {
    // A few boxes at a time, out of order, revisiting, at an uneven rhythm — a
    // machine ripping through the page rather than reading it top to bottom. A
    // sweep can't say this: it puts the field up as one picture and holds it.
    // The whole burst is scheduled here in one pass, because the alternative is
    // a bridge round-trip per box, and at ~100ms each the rhythm is the latency.
    retire(spec => spec.kind === 'lock')
    holder.aimed = null

    // Deterministic — the same page strobes the same way twice — but with a
    // fresh draw per call rather than one hash of the index, so the gaps, the
    // clump sizes and the picks are independent of each other. Read in a fixed
    // order below: reordering these calls changes the sequence.
    let seed = 0
    const roll = () => {
      const noise = Math.sin(++seed * 78.233) * 43758.5453

      return noise - Math.floor(noise)
    }

    if (still) {
      // Nothing about a strobe survives reduced motion, and the timed fallback
      // is worse than useless here: with no animation to hold a mark invisible
      // through its delay, every box in the burst would paint at once. Show the
      // field it was going to rattle through instead.
      outline(field, 'candidate', () => 0)
    } else {
      for (let at = 0; at < burst; ) {
        const clump = roll() < 0.62 ? 1 : roll() < 0.7 ? 2 : 3

        for (let k = 0; k < clump; k++) {
          // Drawn WITH replacement: hitting the same box twice in a run is the
          // point, not a flaw in the shuffle.
          const el = field[Math.floor(roll() * field.length)]
          const seen = spot(el)

          if (seen.width >= 1 && seen.height >= 1) {
            add({ a: el, beat: at, kind: 'flash', label: sniff(el), life: blink + roll() * flick })
          }
        }

        at += roll() < 0.85 ? tempo + roll() * tempo * 2 : gasp + roll() * gasp
      }
    }
  } else if (stage === 'strike') {
    add({ a: target as Element, kind: 'hit' })
  } else {
    // aim — box the target and put the hand on it.
    retire(spec => spec.kind === 'lock')

    // The lock is always named, however dense the field it is standing in. On a
    // page too busy to name the candidates it is the only tag on screen, which
    // is the right hierarchy: the outlines say what the agent can reach, and the
    // one name says what it is reaching for.
    add({ a: target as Element, kind: 'lock', label: sniff(target as Element) })
    style(parts.pointer, { opacity: '1' })
  }

  // One pass for the whole stage: draws whatever was just added, redraws any pin
  // that lost its shadow tree to a rebuild, and re-arms the tracking loop.
  pump()
}
