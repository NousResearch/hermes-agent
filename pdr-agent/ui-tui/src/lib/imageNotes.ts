// The image notes a surface adds to a typed prompt for the model, split back off: the gateway's and the
// CLI's leading vision descriptions (`[The user sent|attached an image…image_url: <path>]`) and
// `build_native_content_parts`' trailing `[Image attached at: <path>]` hints. Mirrors
// agent/image_routing.py::split_image_notes; keep the two in step.

const IMAGE_NOTE_RE = /^\[The user (?:sent|attached) an image[\s\S]*?image_url: ([^\n]+?)(?: ~)?\](?:\n\n|$)/
const NATIVE_HINTS_RE = /\n\n((?:\[Image attached(?: at)?: [^\n]+\]\n?)+)$/
const NATIVE_HINT_RE = /\[Image attached(?: at)?: ([^\n]+)\]/g
const DEFAULT_IMAGE_CAPTION = 'What do you see in this image?'

/** What the user typed and the images the notes name; text without notes comes back unchanged. */
export function splitImageNotes(text: string): { images: string[]; text: string } {
  const images: string[] = []
  let rest = text
  let match: null | RegExpExecArray

  while ((match = IMAGE_NOTE_RE.exec(rest))) {
    images.push(match[1]!.trim())
    rest = rest.slice(match[0].length)
  }

  const hints = images.length ? null : NATIVE_HINTS_RE.exec(rest)

  if (hints) {
    images.push(...[...hints[1]!.matchAll(NATIVE_HINT_RE)].map(hit => hit[1]!.trim()))
    rest = rest.slice(0, hints.index)
    rest = rest.trim() === DEFAULT_IMAGE_CAPTION ? '' : rest
  }

  return images.length ? { images, text: rest } : { images, text }
}
