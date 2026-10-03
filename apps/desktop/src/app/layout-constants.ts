// Responsive horizontal gutter for primary content bodies (settings right side,
// skills, artifacts, command center / sessions). Ratio-based so it scales with
// the window, but clamped so it never collapses on narrow widths or runs away
// on ultrawide displays. Headers/tabs intentionally keep their own tighter
// padding.
//
// NOTE: these must stay literal strings — Tailwind's scanner only picks up
// complete class names, so do not build them via template interpolation.
export const PAGE_INSET_X = 'px-[clamp(1.25rem,4vw,4rem)]'

// Matching negative inline-margin to bleed an element (e.g. a sticky header bar)
// out to the gutter edges before re-applying PAGE_INSET_X.
export const PAGE_INSET_NEG_X = '-mx-[clamp(1.25rem,4vw,4rem)]'

// Readable cap for overlay "inner page" bodies (settings, command center). Wide
// enough to breathe, tight enough that content doesn't sprawl on ultrawide
// displays. Pair with `mx-auto w-full` to center within the pane. Literal string
// for Tailwind's scanner (see PAGE_INSET_X note).
export const PAGE_MAX_W = 'max-w-[75rem]'

// Narrowest window that still docks a rail in the grid; under it both rails
// leave the grid and become the hover-reveal overlay. Single source of truth for
// the responsive collapse point.
//
// A rail costs 237px (SIDEBAR_DEFAULT_WIDTH) and the chat beside it wants roughly
// what a popped-out session window enforces on itself (420px), so docking stops
// paying for itself around here — while still leaving an overlay band down to the
// window's own 400px minimum. Expressed as a dock floor rather than a collapse
// ceiling so half-screen splits stay docked on common laptop widths (1280 → 640).
export const SIDEBAR_DOCK_MIN_WIDTH_PX = 640
// `max-width` is inclusive: shave a hair so an exactly-640px window docks.
export const SIDEBAR_COLLAPSE_MEDIA_QUERY = `(max-width: ${SIDEBAR_DOCK_MIN_WIDTH_PX - 0.02}px)`
