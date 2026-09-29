/** Client-side checks for the Discord card. The backend repeats them before saving. */

const SNOWFLAKE = /^\d{15,22}$/

export function discordSnowflake(raw: string): string | null {
  const mention = raw.trim().match(/^<(?:@!?|#)(\d{15,22})>$/)
  const text = mention?.[1] ?? raw.trim()

  return SNOWFLAKE.test(text) ? text : null
}

export function discordTokenAccepted(raw: string): boolean {
  const token = raw.trim()

  return token.length >= 20 && !/^\d+$/.test(token) && token.split('.').length >= 3
}

export function discordMessageReady(raw: string): boolean {
  const text = raw.trim()

  return text.length > 0 && text.length <= 2000
}
