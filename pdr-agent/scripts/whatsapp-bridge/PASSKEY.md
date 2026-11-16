# WhatsApp passkey pairing

The bridge uses the official `@whiskeysockets/baileys` 7.0.0-rc13 npm release.

`passkey_flow.js` and `passkey_crypto.js` adapt the MIT-licensed handshake from
vinikjkkj's [Baileys PR #2689](https://github.com/WhiskeySockets/Baileys/pull/2689).
That PR was closed unmerged after stale handling. Original commit references and
the license are preserved in `PASSKEY-LICENSE`.

`passkey_socket.js` attaches the handshake to Baileys' public notification event,
query method, authentication state and connection lifecycle. Baileys owns the
notification acknowledgement; the interactive signer runs separately. Closing the
socket cancels pending authentication and removes the added listeners.

A server-sent prologue request derives a pairing handoff proof from the ADV
secret the phone already received during the QR scan (as upstream does), so the
primary normally skips code matching and no confirmation prompt appears. A
request that supersedes an earlier one keeps that QR-era key rather than one
derived from the secret the first request rotated in, which the phone never saw;
whether WhatsApp re-sends the request within one pairing is unverified. The
terminal prompt only runs when no trusted handoff applies (for example when
signing takes longer than the five-minute handoff window). The flow awaits the
owner's answer itself and sends the pairing request only for the attempt that
derived the code: a newer prologue request or a socket close cancels both the
pending browser prompt and the code question, and whatever the interrupted call
then throws counts as that cancellation, not a pairing failure. Ctrl+C at the
code question ends pairing with an error. Other failures
print the error's name and message; assertion and credential buffers are not
printed. The interactive signer loads the real WhatsApp Web origin and validates the returned
challenge and origin. No existing browser profile or browser session is imported.
The temporary browser profile does not inherit Chrome's signed-in password manager.
Availability of a phone or operating-system passkey depends on the browser's
WebAuthn prompt and has not been verified with a real WhatsApp account.

The official Baileys package requires Node 20 or newer. The signer supports that
same floor without depending on `AbortSignal.any`, which arrived in Node 20.3.
A passkey challenge in gateway mode before the connection opens exits the bridge
with code 78 (`PASSKEY_REQUIRED_EXIT`); the adapter reports
`whatsapp_passkey_required` as non-retryable until interactive pairing. A
challenge reaching a paired session (connected, or still logging in after a
reconnect) is logged and ignored, as before passkey support, so a working
gateway is not parked. Dashboard pairing
(`--pair-json`) reports `{event: error, error: passkey_required}`, which the
dashboard shows as an instruction to pair from a local terminal.

Known limit: Baileys keeps rotating QR codes while the browser prompt is open,
and when it runs out of codes it ends the connection, which cancels the prompt
before the signer's own 600-second bound. Real-account timing is unverified.

## Verification

Run from this directory:

```sh
npm ci --ignore-scripts --no-audit --no-fund
node --test *.test.mjs
```

Set `PASSKEY_TEST_BROWSER` to an installed Chrome/Chromium executable to include
the offline WebAuthn integration test. It uses a virtual authenticator and a
test-only intercepted page, verifies a real assertion signature, then decrypts
the actual handshake payload as a simulated primary. Production does not
intercept the page. Tests also cover connection teardown, rejected signing,
server-origin filtering and preserving explicit owner confirmation.

Offline tests do not establish WhatsApp server acceptance. Keep real-account
pairing validation outstanding until observed. This supports interactive CLI
pairing with a graphical browser, not headless SSH or dashboard onboarding.
