/** Copy for a server-hosted artifact a tool result names. */
export interface HostedArtifactCopy {
  /** Fallback title when the result carries none. */
  untitled: string
  version: (version: number) => string
  /** Badge when a later result already named a newer version. */
  newerVersion: (version: number) => string
  openViewer: string
  /** Shown instead of the live frame (expired ticket, or a build that does not frame). */
  linkOnly: string
  frameTitle: (title: string) => string
  /** Above a message an artifact asks to send; the user must confirm it. */
  messagePrompt: string
  messageSend: string
  messageDismiss: string
}
