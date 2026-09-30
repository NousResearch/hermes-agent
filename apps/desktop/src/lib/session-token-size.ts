/**
 * When a conversation's cumulative tokens (everything sent + everything
 * generated, summed over the whole chat) cross a full model context window,
 * every further turn re-reads the entire history: it is the slowest and most
 * expensive the chat will ever be, and the model starts losing its earliest
 * context. That is the point to start a fresh chat, so the sidebar surfaces the
 * token figure — unbidden and in red — once a row crosses this line.
 *
 * 200k ≈ a full Claude context window. Deliberately NOT dollar-derived:
 * per-session cost is 0 on subscription/OAuth auth, and the meaningful signal
 * is "the model is full", which is a token count, not a price.
 */
export const LARGE_CHAT_TOKENS = 200_000

/** True once a chat's total tokens cross {@link LARGE_CHAT_TOKENS}. Guards
 *  against a missing/negative count so it degrades to "not large" rather than
 *  throwing or false-flagging an empty row. */
export function isLargeChat(totalTokens: null | number | undefined): boolean {
  return typeof totalTokens === 'number' && totalTokens >= LARGE_CHAT_TOKENS
}
