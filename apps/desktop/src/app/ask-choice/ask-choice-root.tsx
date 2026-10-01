import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { ErrorBoundary } from '@/components/error-boundary'
import { ThemeProvider } from '@/themes/context'

import { AskChoiceApp } from './ask-choice-app'

/**
 * Boot the on-screen choice dialog window (`?win=askchoice`). A small,
 * transparent, always-on-top, FOCUSABLE card centered on the primary display
 * that asks a short multiple-choice question and reports the click back to main
 * (which persists it for the `ask_choice` tool). Unlike the click-through
 * scanline overlay, the user must click a button, so it's a normal focusable
 * window (see electron/ask-choice-window.ts).
 */
export function mountAskChoice(): void {
  const style = document.createElement('style')
  style.textContent = 'html,body,#root{background:transparent !important;overflow:hidden;}'
  document.head.appendChild(style)

  const root = document.getElementById('root')

  if (!root) {
    return
  }

  createRoot(root).render(
    <StrictMode>
      <ErrorBoundary label="ask-choice">
        <ThemeProvider>
          <AskChoiceApp />
        </ThemeProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}
