/**
 * Native IX Agency portal login (no webview): drives the admin portal's own
 * email-OTP endpoints directly.
 *
 *   1. POST /api/auth/otp/send   {email}                → {challenge, expiresAt}
 *   2. POST /api/auth/otp/verify {email, code, challenge} → Set-Cookie session
 *
 * The verify call MUST run through the persist:ix-agency-portal session's
 * fetch (session.fetch, credentials 'include') so the httpOnly OTP session
 * cookie lands in the exact cookie jar the main-process auth probe checks —
 * that is what unlocks the native chat / MCP surfaces. Pure logic with an
 * injected fetch so it is unit-testable.
 */

export interface OtpChallenge {
  challenge: string
  expiresAt: number
  /** Present only when the portal runs with OTP_DEV_ECHO=1 (local dev). */
  devCode?: string
}

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const body = (await res.json()) as { error?: string }

    if (body && typeof body.error === 'string' && body.error) {
      return body.error
    }
  } catch {
    // non-JSON error body — use the fallback
  }

  return fallback
}

/** Step 1: request a 6-digit code (portal emails it via SES). */
export async function ixLoginSendOtp(
  portalUrl: string,
  email: string,
  fetchImpl: typeof fetch = fetch
): Promise<OtpChallenge> {
  const trimmed = String(email ?? '').trim()

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmed)) {
    throw new Error('Enter a valid email address.')
  }

  const res = await fetchImpl(new URL('/api/auth/otp/send', portalUrl).toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: trimmed }),
    signal: AbortSignal.timeout(15_000)
  })

  if (!res.ok) {
    throw new Error(await readError(res, `Could not send the code (HTTP ${res.status}).`))
  }

  const body = (await res.json()) as OtpChallenge

  if (!body || typeof body.challenge !== 'string' || !body.challenge) {
    throw new Error('The portal returned no OTP challenge.')
  }

  return body
}

/**
 * Step 2: verify the code. On success the portal answers Set-Cookie with the
 * httpOnly OTP session (and, for scoped admins, the pinned-scope cookie);
 * calling this through session.fetch persists both into the portal partition.
 */
export async function ixLoginVerifyOtp(
  portalUrl: string,
  input: { email: string; code: string; challenge: string },
  fetchImpl: typeof fetch = fetch
): Promise<void> {
  const code = String(input.code ?? '').trim()

  if (!/^\d{6}$/.test(code)) {
    throw new Error('The code is 6 digits.')
  }

  const res = await fetchImpl(new URL('/api/auth/otp/verify', portalUrl).toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // credentials 'include' so the response cookies are persisted into the
    // calling session's cookie store (Electron session.fetch semantics).
    credentials: 'include',
    body: JSON.stringify({ email: String(input.email ?? '').trim(), code, challenge: input.challenge }),
    signal: AbortSignal.timeout(15_000)
  })

  if (!res.ok) {
    throw new Error(await readError(res, `Sign-in failed (HTTP ${res.status}).`))
  }

  const body = (await res.json().catch(() => null)) as null | { ok?: boolean }

  if (!body?.ok) {
    throw new Error('The portal did not confirm the sign-in.')
  }
}
