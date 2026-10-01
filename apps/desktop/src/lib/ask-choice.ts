// Renderer client for the on-screen choice dialog. The main process owns the
// request/response file bridge; the renderer only mirrors the request it's
// showing and reports the user's decision back. See electron/ask-choice-window.ts.

export interface AskChoiceRequest {
  request_id: string
  question: string
  options: string[]
  ts: number
}
