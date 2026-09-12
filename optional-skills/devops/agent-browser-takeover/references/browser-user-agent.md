# One browser configuration for agent and takeover

VNC mirrors the existing headed browser; it is not a second browser requiring
separate user-agent settings. Agent clients should reuse the shared server.
The hold process owns the visible tab. Creating another Playwright context
does not recover that tab's cookies or authenticated state.

## Optional user-agent compatibility override

The sample server uses Camoufox `launch_options`, including its generated
fingerprint configuration. Do not assume that a custom launcher which starts
only the browser binary has the same defaults. Check the actual request
User-Agent and JavaScript `navigator.userAgent` before changing anything.

In my POC, a custom launcher exposed a Camoufox product token. A Firefox-style
profile override was followed by a successful page load after an access
restriction. This is one compatibility observation, not proof that the UA
was the sole cause, that all sites work, or that automation is undetectable.

For a custom Firefox-compatible launcher that needs this override, set
`general.useragent.override` in its `firefox_user_prefs` at startup. Construct
it for the actual platform and installed Gecko version. For example, the
shape on Linux is:

```text
Mozilla/5.0 (X11; Linux x86_64; rv:<engine-version>) Gecko/20100101 Firefox/<engine-version>
```

Replace the version placeholder after checking the installed engine. Do not
copy a fixed version indefinitely, claim Chrome for a Firefox engine, or
layer a contradictory override over Camoufox's generated fingerprint.
This is intentionally an optional compatibility recipe, not an unconditional
change to the sample server's generated fingerprint defaults.

All contexts without a conflicting per-context UA use the browser default;
independent `Camoufox()` launches do not inherit a remote server's settings.
Prefer the shared server. Review any independently launched exception for
matching UA, proxy, DNS, and WebRTC settings.

## Apply without destroying the active session

Changing the launcher prepares the next startup; it does not mutate a running
browser. An operator can set the same preference in `about:config` for the
current profile. Do not restart the browser or hold process merely to deploy
this change: temporary contexts and their logins may disappear. Schedule
restart verification separately with the operator.

Preserve the WireGuard-only viewer, loopback browser-control and raw VNC
listeners, and configured proxy/remote-DNS/WebRTC protection. A UA override
is not a reason to disable privacy protections or rotate network identities.

## Verification and rollback

- Compare HTTP User-Agent at an HTTPS echo endpoint with `navigator.userAgent`.
- Check proxy egress and WebRTC protection without logging cookies or tokens.
- Inspect the actual VNC view, not only an HTTP success status.
- Stop at an access restriction or human verification prompt rather than
  repeatedly retrying or treating a UA as guaranteed access.
- Recheck UA/platform/engine consistency after each browser update.
- To revert, remove only the override from startup preferences and reset the
  profile preference if applied manually. Preserve other protection settings.

Record current-profile success separately from startup/restart verification.
