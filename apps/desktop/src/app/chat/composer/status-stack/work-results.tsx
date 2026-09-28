import { AgentHuddle } from './agent-huddle'

/**
 * The work lane above the composer.
 *
 * It used to paint the finished worker's report here — a model answer with
 * markdown, tables and code, straight onto the composer's doorstep. The report
 * is still the truth the Live agent retells out loud (the announcement queue in
 * `use-realtime-conversation` reads it off the same store), but on screen the
 * user gets the huddle instead: a small animated scene of the agents talking to
 * each other, never the text of what they said.
 */
export function WorkResults({ sessionId }: { sessionId: string }) {
  return <AgentHuddle sessionId={sessionId} />
}
