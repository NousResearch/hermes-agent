import { $layoutScale, scaleMediaQuery } from '@/lib/layout-scale'
import { $portalShell } from '@/store/shell-variant'

/** Each media list's text as its stylesheet wrote it, so a second scale starts from the original. */
const written = new WeakMap<MediaList, string>()

function scaleMedia(media: MediaList, scale: number): void {
  const original = written.get(media) ?? media.mediaText
  written.set(media, original)
  const next = scaleMediaQuery(original, scale)

  if (media.mediaText !== next) {
    media.mediaText = next
  }
}

function scaleRules(rules: CSSRuleList, scale: number): void {
  for (const rule of rules) {
    if (rule instanceof CSSMediaRule) {
      scaleMedia(rule.media, scale)
    } else if (rule instanceof CSSImportRule && rule.styleSheet) {
      scaleMedia(rule.media, scale)
      scaleSheet(rule.styleSheet, scale)
    }

    // Media rules nest: in @layer and @supports blocks, and inside style rules.
    if ('cssRules' in rule && rule.cssRules instanceof CSSRuleList) {
      scaleRules(rule.cssRules, scale)
    }
  }
}

function scaleSheet(sheet: CSSStyleSheet, scale: number): void {
  let rules: CSSRuleList

  try {
    rules = sheet.cssRules
  } catch {
    // A stylesheet from another origin keeps its rules to itself.
    return
  }

  scaleRules(rules, scale)
}

const scaleDocument = (scale: number) => [...document.styleSheets].forEach(sheet => scaleSheet(sheet, scale))

/**
 * The web app's UI scale is a `zoom` on the page, which leaves the viewport as it was: at 200% the
 * app has half the room and every size query still answered for the whole window, so the desktop
 * layout was drawn in a phone's width (titles a syllable to a line, dialogs past the window's edge).
 *
 * In the redesign the layout follows the room that is left, as it does under the browser's own
 * zoom: `$layoutScale` carries the scale to the queries asked from script (lib/layout-scale.ts), and
 * here every stylesheet's size queries are rewritten to it, the ones already loaded and each one a
 * lazily loaded page brings. At 100% nothing is rewritten and nothing is watched. Classic keeps the
 * behaviour it has on main.
 */
export function installLayoutScale(): void {
  const zoom = window.hermesDesktop?.zoom

  if (document.documentElement.dataset.hermesDesktopHost !== 'browser' || !zoom?.factor) {
    return
  }

  const factor = zoom.factor
  const sync = () => $layoutScale.set($portalShell.get() ? factor() : 1)

  zoom.onChanged(sync)
  $portalShell.listen(sync)
  sync()

  let frame = 0
  let scaled = false

  const rescale = () => {
    frame = 0
    scaleDocument($layoutScale.get())
  }

  const schedule = () => {
    if (!frame) {
      frame = requestAnimationFrame(rescale)
    }
  }

  // New stylesheets: a <style> or <link> joining <head>, and a <link>'s rules arriving on load.
  const arrivals = new MutationObserver(schedule)

  $layoutScale.subscribe(scale => {
    if (scale === 1 && !scaled) {
      return
    }

    scaled = scale !== 1
    scaleDocument(scale)

    if (scaled) {
      arrivals.observe(document.head, { characterData: true, childList: true, subtree: true })
      document.addEventListener('load', schedule, true)
    } else {
      arrivals.disconnect()
      document.removeEventListener('load', schedule, true)
    }
  })
}
