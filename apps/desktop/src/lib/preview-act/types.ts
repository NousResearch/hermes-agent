/**
 * PREVIEW ACT TYPES — the shapes the engine, the overlay, and the Python tool
 * all agree on.
 *
 * Declarations only. They erase at compile time, which is what lets the engine
 * modules import them while still stringifying into the guest page as plain JS
 * (see `naming.ts` for the self-containment contract they have to satisfy).
 *
 * Re-exported from `act-in-page.ts`, so every existing import site keeps
 * working and nothing outside this directory needs to know they moved.
 */

/** One interactable node, as the agent sees it. */
export interface PreviewElement {
  /** Present and true only when the control is non-interactive right now. */
  disabled?: boolean
  /** Human-readable label (aria-label, text, placeholder, value …). */
  label: string
  /** Durable handle for as long as this page is open: 'btn-sign-in'. Legible on
   *  purpose — see the ref-minting note in `actInPage`. */
  ref: string
  /** Explicit ARIA role, else the tag name. */
  role: string
  /** The element's `#id` or `[data-testid]`, when it has one. Absent otherwise
   *  — address the element by its `ref`. */
  selector?: string
  /** Current value of a form control, truncated. */
  value?: string
}

/** An element that is still itself but no longer reads the same.
 *
 *  Only the fields that actually moved are present. Role and selector are
 *  absent by construction rather than by omission: a change in either would
 *  mean this is a different element, which the re-bind ladder would have
 *  refused to match in the first place. */
export interface PreviewElementChange {
  /** Present only when the control's availability flipped. */
  disabled?: boolean
  label?: string
  ref: string
  value?: string
}

/** What changed on the page since the last look. Sent instead of the whole
 *  inventory once the agent has a baseline for the page — see `survey`. */
export interface PreviewActDelta {
  /** Elements seen for the first time, in full. */
  added?: PreviewElement[]
  /** Same handle, new label/value/disabled state — and nothing else. */
  changed?: PreviewElementChange[]
  /** Handles that are gone from the page. */
  removed?: string[]
  /** Handles whose element was destroyed and recreated by a re-render. The
   *  handle still works; nothing about them needs re-reading. */
  rebound?: string[]
  /** How many handles were on the page and untouched. */
  same?: number
}

/** A normalized action. `kind` is the verb; the rest is per-verb payload. */
export interface PreviewActAction {
  /** scroll distance in px. Defaults to ~90% of the viewport height. */
  amount?: number
  key?: string
  /** `pin`/`unpin`/`hold` never reach the engine — they resolve their targets
   *  through `locate`/`elements` and then talk to the overlay — but they arrive
   *  on the same wire. */
  kind:
    | 'click'
    | 'elements'
    | 'hold'
    | 'hover'
    | 'locate'
    | 'pin'
    | 'press'
    | 'scroll'
    | 'strobe'
    | 'type'
    | 'unpin'
  /** locate: also give the target keyboard focus, for a key press that must not
   *  be preceded by a click (which would activate the control instead). */
  focus?: boolean
  /** elements: answer with the whole inventory rather than a delta. */
  full?: boolean
  /** Cap on the returned inventory. */
  max?: number
  ref?: string
  selector?: string
  /** type: press Enter (and submit the owning form) after entering text. */
  submit?: boolean
  text?: string
  to?: 'bottom' | 'top'
}

export interface PreviewActResult {
  /** What the action landed on, for the agent's own log. */
  acted?: string
  /** What moved since the last look. Present INSTEAD of `elements` once the
   *  agent holds a baseline for this page. */
  delta?: PreviewActDelta
  /** The full inventory. Sent on the first look at a page, and again whenever
   *  the page changed too much for a delta to be the cheaper answer. */
  elements?: PreviewElement[]
  error?: string
  note?: string
  /** Viewport centre of a located target, for aiming real pointer input at it. */
  point?: { x: number; y: number }
  success: boolean
  title?: string
  /** locate: whether the target actually takes typed text. */
  typable?: boolean
  /** Live document URL after the action — a change means it navigated. */
  url?: string
}

/** One element the agent has a handle on, remembered across actions. */
export interface PreviewActBinding {
  el: Element
  /** What it read as last time. Kept field by field rather than as one hash so
   *  a change can be reported as only the part that moved. */
  label: string
  /** The accessible name at mint time, for re-finding this element after a
   *  re-render destroys and recreates its node. */
  name: string
  /** Whether the control was unavailable last time. */
  off: boolean
  /** Nearest-landmark path plus position among same-role siblings. */
  path: string
  ref: string
  role: string
  /** `id` / `name` / `data-testid` / `aria-label`, if the page provides one.
   *  The strongest re-bind signal there is, and the only one a rewrite of the
   *  surrounding markup cannot disturb. */
  stable: string
  value: string
}

/** Where the surface keeps what it knows between actions (a window global in
 *  the preview page), so a handle still means something on the next call. */
export interface PreviewActHolder {
  /** Target of the action in flight, for the watch overlay to draw onto. */
  aimed?: Element | null
  /** Every handle minted on this page, live or not yet retired. */
  book?: PreviewActBinding[]
  /** Next disambiguating suffix per ref stem, so two "Edit" buttons become
   *  `btn-edit` and `btn-edit-1`. Never rewound: a retired handle's name is
   *  not handed to a different element later in the same page. */
  coined?: Record<string, number>
  /** The on-screen subset, for the overlay to outline. Diverges from `nodes` in
   *  both directions: it drops what is below the fold, and it is not capped at
   *  the inventory's size. */
  field?: Element[]
  nodes?: Element[]
  /** URL the snapshot was taken on; a navigation retires every handle. */
  url?: string
}
