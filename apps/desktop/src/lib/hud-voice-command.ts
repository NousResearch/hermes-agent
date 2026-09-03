/**
 * Spoken (or typed) HUD commands — "HUD top left", "HUD follow me", "HUD come
 * here" — recognised as WHOLE utterances the same way the voice stop word is,
 * so "move the hud to the top left" never reaches the agent and "explain the
 * hud code" always does. Pure: the parser owns the grammar, the caller owns
 * what happens.
 */

export type HudPlaceAnchor =
  | 'bottom-center'
  | 'bottom-left'
  | 'bottom-right'
  | 'center'
  | 'top-center'
  | 'top-left'
  | 'top-right'

export type HudVoiceCommand =
  | { kind: 'come-here' }
  | { kind: 'follow'; on: boolean }
  | { kind: 'hide' }
  | { kind: 'place'; anchor: HudPlaceAnchor }

const ADDRESS = /^(?:hey |ok |okay |yo )?(?:hermes[,!]?\s+)?/
const FILLER = /\b(?:please|now|the|a|to|over|go|move|put|send|park|bring|your|yourself|window|bar|panel|mode)\b/g

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** "top right", "upper-left", "bottom", "middle" → anchor. */
function anchorOf(rest: string): HudPlaceAnchor | null {
  const t = rest.replace(/-/g, ' ')
  const vertical = /\b(top|upper)\b/.test(t) ? 'top' : /\b(bottom|lower|down)\b/.test(t) ? 'bottom' : null
  const horizontal = /\bleft\b/.test(t) ? 'left' : /\bright\b/.test(t) ? 'right' : null

  if (/\b(center|centre|middle)\b/.test(t) && !vertical && !horizontal) {
    return 'center'
  }

  if (vertical && horizontal) {
    return `${vertical}-${horizontal}`
  }

  if (vertical) {
    return `${vertical}-center`
  }

  return null
}

export function parseHudVoiceCommand(transcript: string): HudVoiceCommand | null {
  const normalized = normalize(transcript)

  if (!normalized) {
    return null
  }

  const body = normalized.replace(ADDRESS, '')

  // The utterance has to be ABOUT the HUD. "hud" (however the recogniser
  // spells it) must appear, and nothing substantive may follow the command.
  if (!/\b(hud|h u d|heads? up|head up display|overlay)\b/.test(body)) {
    return null
  }

  const rest = body
    .replace(/\b(hud|h u d|heads? up display|heads? up|head up display|overlay)\b/g, ' ')
    .replace(FILLER, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  if (/^(?:me|here|come|come here|here here|come me|come to me|to me|my cursor|cursor|my pointer|pointer|near me|with me)$/.test(rest)) {
    return { kind: 'come-here' }
  }

  if (/^(?:follow|follow me|follow my cursor|follow mouse|follow my mouse|follow pointer|follow my pointer|start following|start follow)$/.test(rest)) {
    return { kind: 'follow', on: true }
  }

  if (/^(?:stay|stay there|stay here|stop following|stop follow|stop|hold|hold still|dont follow|don t follow|do not follow|unfollow)$/.test(rest)) {
    return { kind: 'follow', on: false }
  }

  if (/^(?:hide|close|away|go away|dismiss|bye|exit|off)$/.test(rest)) {
    return { kind: 'hide' }
  }

  const anchor = anchorOf(rest)

  if (anchor && /^[a-z\s-]+$/.test(rest) && rest.split(' ').length <= 4) {
    return { kind: 'place', anchor }
  }

  return null
}
