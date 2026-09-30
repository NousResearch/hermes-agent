import { useStore } from '@nanostores/react'
import React from 'react'

import { $selectedStoredSessionId } from '@/store/session'

import { WiredPane } from '../wiring'

export function DialoguePane() {
  const sessionId = useStore($selectedStoredSessionId)

  if (!sessionId) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center text-muted-foreground">
        <p className="text-xs">Select a session to start the dialogue</p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-background transition-all">
      {/* Specialized dialogue header - minimal compared to workspace */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card/50">
        <span className="text-[11px] font-bold uppercase tracking-widest opacity-50">Dialogue</span>
        <div className="flex gap-1">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
        </div>
      </div>

      {/* We reuse the chat routing logic but wrap it in a constrained dialogue container */}
      <div className="flex-1 overflow-hidden relative">
        <WiredPane part="chatRoutes" />
      </div>
    </div>
  )
}
