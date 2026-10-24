import { useStore } from '@nanostores/react'
import type { WebglAddon } from '@xterm/addon-webgl'
import type { Terminal } from '@xterm/xterm'
import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'

import { $terminalFontSize } from '@/store/terminal-font-size'

import { $terminalFontFamily, applyTerminalFontFamily, resolveTerminalFontFamily } from './terminal-font'
import { redrawAllTerminals } from './terminals'

interface TerminalFontControllerOptions {
  fitRef: RefObject<((visible: boolean) => void) | null>
  termRef: RefObject<Terminal | null>
  webglRef: RefObject<WebglAddon | null>
}

function clearAtlasSafely(webgl: Pick<WebglAddon, 'clearTextureAtlas'> | null): boolean {
  try {
    webgl?.clearTextureAtlas()

    return true
  } catch {
    return false
  }
}

/**
 * Share profile-backed font state across user and agent terminals. The owner
 * flips mountedRef only after term.open(); before then, its mount path reads
 * latestFontFamilyRef and warms the newest value itself.
 */
export function useTerminalFontController({ fitRef, termRef, webglRef }: TerminalFontControllerOptions) {
  const configured = useStore($terminalFontFamily)
  const fontFamily = resolveTerminalFontFamily(configured)
  const latestFontFamilyRef = useRef(fontFamily)
  const mountedRef = useRef(false)
  const generationRef = useRef(0)

  const fontSize = useStore($terminalFontSize)
  const latestFontSizeRef = useRef(fontSize)

  latestFontFamilyRef.current = fontFamily
  latestFontSizeRef.current = fontSize

  useEffect(() => {
    const term = termRef.current

    if (!mountedRef.current || !term || term.options.fontSize === fontSize) {
      return
    }

    term.options.fontSize = fontSize
    fitRef.current?.(true)

    clearAtlasSafely(webglRef.current)
    redrawAllTerminals()
  }, [fitRef, fontSize, termRef, webglRef])

  useEffect(() => {
    const term = termRef.current

    if (!mountedRef.current || !term || term.options.fontFamily === fontFamily) {
      return
    }

    const generation = ++generationRef.current
    let cancelled = false

    void applyTerminalFontFamily({
      clearTextureAtlas: () => webglRef.current?.clearTextureAtlas(),
      fit: () => fitRef.current?.(true),
      fontFamily,
      isCurrent: () => !cancelled && generationRef.current === generation,
      term
    })
    // Font is an atlas-share key: changing one terminal's font rebuilds the
    // shared glyph set, so siblings sharing the atlas must rebuild their
    // models too (see redrawAllTerminals). The caller clears itself inline
    // (applyTerminalFontFamily), so skip it in the fan-out.
    redrawAllTerminals(term)

    return () => {
      cancelled = true
    }
  }, [fitRef, fontFamily, termRef, webglRef])

  return { latestFontFamilyRef, latestFontSizeRef, mountedRef }
}
