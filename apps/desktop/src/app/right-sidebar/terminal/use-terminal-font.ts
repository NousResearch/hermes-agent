import { useStore } from '@nanostores/react'
import type { WebglAddon } from '@xterm/addon-webgl'
import type { Terminal } from '@xterm/xterm'
import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'

import { $terminalFontFamily, applyTerminalFontFamily, resolveTerminalFontFamily } from './terminal-font'

interface TerminalFontControllerOptions {
  fitRef: RefObject<(() => void) | null>
  termRef: RefObject<Terminal | null>
  webglRef: RefObject<WebglAddon | null>
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

  latestFontFamilyRef.current = fontFamily

  useEffect(() => {
    const term = termRef.current

    if (!mountedRef.current || !term || term.options.fontFamily === fontFamily) {
      return
    }

    const generation = ++generationRef.current
    let cancelled = false

    void applyTerminalFontFamily({
      clearTextureAtlas: () => webglRef.current?.clearTextureAtlas(),
      fit: () => fitRef.current?.(),
      fontFamily,
      isCurrent: () => !cancelled && generationRef.current === generation,
      term
    })

    return () => {
      cancelled = true
    }
  }, [fitRef, fontFamily, termRef, webglRef])

  return { latestFontFamilyRef, mountedRef }
}
