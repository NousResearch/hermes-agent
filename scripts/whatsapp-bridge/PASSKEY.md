# Passkey dependency and verification

The bridge pins a review candidate of Baileys at commit
`e5125258c920bf06448cf9f25fda32671aac5d3b` in `helix4u/Baileys`.
The archive URL is immutable by commit and package-lock.json records its integrity.
It is not an official Baileys release. Restore an official release pin once an
equivalent passkey implementation is available upstream.

The candidate preserves vinikjkkj's original commits from
[Baileys PR #2689](https://github.com/WhiskeySockets/Baileys/pull/2689), which was
closed unmerged after stale handling. Source commit
`8979f800c4c0a0b26fe6430579748c3b861b334b` adds explicit owner confirmation,
socket-lifetime cancellation, notification-origin validation and bridge-facing
error/confirmation events. The subsequent packaging commit includes `lib/`
generated with the repository's locked Yarn 4.9.2 build. It retains the MIT
license and includes `HERMES-PASSKEY.md` with build provenance.

The compiled files make the archive usable with the bridge's CI install command:

```sh
npm ci --ignore-scripts --no-audit --no-fund
node --test *.test.mjs
```

Neither command requires a sibling source checkout or build tools for Baileys.
The package-local `.npmrc` permits direct remote URL dependencies for npm 12;
transitive remote URL dependencies remain disabled. Setup checks the installed
dependency records against this lockfile so an older `node_modules` directory
does not skip the new pin.
For the additional offline browser integration test, set `PASSKEY_TEST_BROWSER`
to a Chrome/Chromium executable before running the tests. That test uses a
virtual authenticator and intercepts the page in the test only. It verifies a
real browser assertion's signature, origin and challenge, then decrypts the
actual handshake's pairing payload as a simulated primary device. Production
signing loads the real WhatsApp Web origin and does not intercept the page.

These tests do not establish WhatsApp server acceptance for a real account.
Keep that distinction in release and support statements. The candidate supports
interactive CLI pairing; it does not add headless or dashboard passkey onboarding.
