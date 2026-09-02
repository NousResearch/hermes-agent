/**
 * Numbered pin stack for comment mode. Saving a pin only appends — it never
 * sends a turn. Numbers are assigned 1..N in add order and stay put if a pin
 * is removed, so Comment 3 in the composer still matches marker 3 on the page.
 */

export type AnnotatePinKind = 'area' | 'element'

export interface AnnotateRect {
  height: number
  width: number
  x: number
  y: number
}

export interface AnnotateIdentity {
  css: Record<string, string>
  selector: string
  tag: string
  text: string
}

export interface AnnotatePin {
  id: string
  identity?: AnnotateIdentity
  imageDataUrl: string
  kind: AnnotatePinKind
  note: string
  number: number
  pageTitle: string
  pageUrl: string
  rect: AnnotateRect
}

export interface AnnotateStack {
  nextNumber: number
  pins: AnnotatePin[]
}

export type AnnotatePinDraft = Omit<AnnotatePin, 'id' | 'number'> & { id?: string }

export function emptyAnnotateStack(): AnnotateStack {
  return { nextNumber: 1, pins: [] }
}

export function addAnnotatePin(stack: AnnotateStack, draft: AnnotatePinDraft): AnnotateStack {
  const pin: AnnotatePin = {
    ...draft,
    id: draft.id || `annotate-${stack.nextNumber}`,
    number: stack.nextNumber
  }

  return {
    nextNumber: stack.nextNumber + 1,
    pins: [...stack.pins, pin]
  }
}

export function updateAnnotatePinNote(stack: AnnotateStack, id: string, note: string): AnnotateStack {
  return {
    ...stack,
    pins: stack.pins.map(pin => (pin.id === id ? { ...pin, note } : pin))
  }
}

export function removeAnnotatePin(stack: AnnotateStack, id: string): AnnotateStack {
  return {
    ...stack,
    pins: stack.pins.filter(pin => pin.id !== id)
  }
}

export function clearAnnotateStack(stack: AnnotateStack): AnnotateStack {
  void stack

  return emptyAnnotateStack()
}

/** Clear a flushed batch while keeping comment numbers unique in this page session. */
export function clearAnnotatePins(stack: AnnotateStack): AnnotateStack {
  return { ...stack, pins: [] }
}

export interface AnnotateSession {
  /** In-progress pick that has a crop but no saved note yet. */
  draft: AnnotatePinDraft | null
  mode: boolean
  overlayLive: boolean
  stack: AnnotateStack
}

export function emptyAnnotateSession(): AnnotateSession {
  return { draft: null, mode: false, overlayLive: false, stack: emptyAnnotateStack() }
}

export function beginAnnotateMode(session: AnnotateSession): AnnotateSession {
  return { ...session, mode: true, overlayLive: true }
}

export function endAnnotateMode(session: AnnotateSession): AnnotateSession {
  return { ...session, draft: null, mode: false, overlayLive: false }
}
