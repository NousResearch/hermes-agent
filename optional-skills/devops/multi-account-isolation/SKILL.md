---
name: multi-account-isolation
description: Verify browser profiles are really isolated.
version: 1.0.0
author: antibrow
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Browser Profiles, Isolation, Proxy, WebRTC, Timezone, Verification]
    homepage: https://antibrow.com
prerequisites:
  commands: [node]
  env: [ANTI_DETECT_BROWSER_KEY]
---

# Profile Isolation - verifying it, not assuming it

A profile that *looks* isolated usually is not. The failures are boring and mechanical: a timezone that does not match the exit IP, a WebRTC candidate carrying the real address, a canvas hash that changes on every read, two profiles that ended up on the same persona. This skill is the check list for catching those before they matter.

> **Authorized use only.** This is for identities you own or are authorized to operate: your own accounts, your own test fixtures, your own QA fleet, and your own anti-fraud stack. It is not for accessing systems without authorization, for accounts that are not yours, or for creating fake accounts or engagement. Comply with the terms of the sites you automate and with applicable law - see [Acceptable use](#acceptable-use).

**What this does not claim.** Passing every check below means the browser layer is internally consistent. It does not mean a given site will treat two profiles as unrelated: things entirely outside the browser - a shared payment instrument, a shared contact detail, identical activity patterns - are not something any browser setting reaches. Treat a clean result as "the technical layer is not the problem", not as a guarantee.

For the SDK that creates and launches these profiles, see the **anti-detect-browser** skill.

## Prerequisites

The checks run from Node or Python code executed with the native Hermes `terminal`
tool. Proxy credentials and `ANTI_DETECT_BROWSER_KEY` come from the environment,
never from literals in the script.


## The configuration invariant

One identity gets one of everything. Any cell shared between two identities is a defect to find:

```
identity  →  profile  →  persona  →  proxy  →  timezone
   1      :     1     :     1     :    1    :     1
```

Profiles are unlimited and free on every antibrow plan, so there is never a reason to reuse one. "Log out and log back in as the other identity" inside one profile defeats the entire setup - the cookie jar and `localStorage` are the point.

## Setup under test

```typescript
import { AntiDetectBrowser } from 'anti-detect-browser'

const ab = new AntiDetectBrowser({ key: process.env.ANTI_DETECT_BROWSER_KEY })

const identities = [
  { profile: 'fixture-us-01', proxy: process.env.PROXY_US_1, tags: ['Windows 10', 'Chrome'] },
  { profile: 'fixture-us-02', proxy: process.env.PROXY_US_2, tags: ['Apple Mac', 'Safari'] },
  { profile: 'fixture-de-01', proxy: process.env.PROXY_DE_1, tags: ['Windows 10', 'Edge'] },
]

for (const id of identities) {
  const { browser, page } = await ab.launch({
    profile: id.profile,              // isolated cookies, storage, login state
    proxy: id.proxy,                  // from the environment, one per identity
    fingerprint: { tags: id.tags },   // drawn once, frozen, replayed after
    label: id.profile,                // floating label so windows are tellable apart
  })
  // ... run the checks below, then ...
  await browser.close()
}
```

Python, same on-disk profile format:

```python
import os
from antibrow import launch

with launch(
    profile="fixture-us-01",
    proxy=os.environ["PROXY_US_1"],   # from the environment, never a literal
    geoip=True,            # timezone + WebRTC follow the proxy exit
    label="fixture-us-01",
) as browser:
    page = browser.new_page()
    print(browser.timezone, browser.public_ip)
```

## The checks

Run each profile **through its own proxy**, and assert rather than eyeball.

| # | Check | How | Fails when |
|---|---|---|---|
| 1 | Timezone matches the exit IP | `browser.timezone` vs the country of `browser.public_ip` | `geoip` was disabled, or `timezone` was forced to something the IP contradicts. This is the single most common defect. |
| 2 | WebRTC exposes only the proxy | [browserleaks.com/webrtc](https://browserleaks.com/webrtc) | ICE candidates still carry a local or real public address |
| 3 | Canvas hash is stable across launches | Read it, close, relaunch the same profile, read again | The two reads differ - a value that changes every read is itself an anomaly, and it means the persona is not frozen |
| 4 | Worker and main thread agree | [CreepJS](https://abrahamjuliot.github.io/creepjs/) | UA, `languages`, `hardwareConcurrency`, timezone or GPU differ when re-read inside a Web Worker |
| 5 | One GPU across three interfaces | CreepJS, or read WebGL / WebGL2 / WebGPU directly | `adapter.info.vendor` does not match the unmasked WebGL renderer family |
| 6 | No two profiles share a persona | Diff `browser.persona` across the fleet | Two profiles report the same UA, screen geometry and seeds |
| 7 | No two profiles share an address | Collect `browser.public_ip` for the fleet | Two identities came out of the same exit, or the same /24 |
| 8 | Cookie jars are separate | Inspect `~/.anti-detect-browser/<profile>/` | One profile directory holds state that belongs to another identity |
| 9 | Whole-stack coherence | [whoer.net](https://whoer.net), [pixelscan.net](https://pixelscan.net) | IP, timezone and locale disagree at a glance |
| 10 | Consistency rules in CI | `npx liarjs` ([liarjs.dev](https://liarjs.dev)) | Any of ~40 open-source cross-layer rules fail - this is the one that runs unattended |

Checks 1, 3 and 7 are the ones worth wiring into CI: they are cheap, deterministic, and they catch the defects that actually recur.

## Reading a failure

Work down in this order, cheapest first - a fingerprint is almost never the actual cause:

1. **Profile name reused?** `list_profiles`, or look at `~/.anti-detect-browser/`. Two identities in one directory explains everything else.
2. **Same address twice?** Confirm each `public_ip` is distinct.
3. **Clock disagrees with the address?** Print `browser.timezone` and `browser.public_ip` together.
4. **Persona regenerated?** If the canvas hash moved between launches, the profile is not frozen - check whether `profile_dir` or the cache directory changed under it.
5. **Only then** the fingerprint itself, verified with the suites above rather than assumed.

## What the runtime touches, and how to check it

Any tool that drives logged-in sessions receives cookies and proxy credentials, so it is fair to ask what it does with them. For antibrow:

| Artifact | Where it lives | Who sees it |
|---|---|---|
| Cookies, `localStorage`, login state | `~/.anti-detect-browser/<profile>/` on your disk | Local. Cloud profile sync is a separate paid-plan feature - check whether it is on before assuming a profile stays on the machine |
| Persona (`persona.json`) | same profile directory, written once and frozen | Local |
| Proxy URL and its credentials | passed to the kernel at launch; answered in the network stack (HTTP 407 / SOCKS5 RFC 1929) so no extension holds them | The kernel process and your proxy provider |
| API key | your environment, or `~/.antibrow/license.key` | Exchanged with `antibrow.com` for a short-lived license token, roughly once a day |

The kernel is a closed-source Chromium build - that is the tradeoff for the spoofing living in C++ rather than in an injectable script - so verify behaviour rather than take it on faith:

```bash
python -m antibrow info          # kernels, profiles, license state, cache dir
```

```python
browser.plan.redacted_args()     # exact kernel command line, secrets masked - safe to paste in a bug report
```

Point it at a proxy whose logs you can read, or at a local MITM proxy, and watch what leaves the machine during a launch. Pin the SDK version and check the published hash (`npm view anti-detect-browser@2.2.0 dist.integrity`) so the code you audited is the code that runs. If a deployment must not phone home at all, this is the wrong tool: license verification is compiled into the kernel and there is no offline mode.

## What isolation cannot cover

Worth stating plainly, because a clean check list invites the wrong conclusion:

- **Anything outside the browser.** A shared payment instrument, a shared contact detail, a shared payout destination - no browser setting touches these, and they are the strongest correlators that exist.
- **Activity patterns.** Identical timing, identical content, identical interaction targets. Not a technical property.
- **Identity verification.** A document check is not a fingerprint problem.
- **A platform's own decision.** Nothing here changes how a site chooses to treat an account.

If the ten checks pass and something still looks wrong, the cause is in this list, not in the browser layer.

## Acceptable use

**Intended:** verifying isolation between identities you own; running client accounts with the account holder's authorization; building QA fixtures that emulate distinct devices; testing your own anti-fraud and correlation logic; auditing what a browser runtime does with your credentials.

**Out of scope, and not supported:** accessing any system without authorization; logging into accounts that are not yours; credential stuffing or account takeover; creating fake accounts, reviews or engagement; circumventing an authentication, payment or authorization control; scraping personal data in violation of applicable law; working around a platform's enforcement decision.

Complying with the terms of the platforms being used, and with applicable law, is the operator's responsibility. Report abuse or a security issue via the contact at `https://antibrow.com`.

## Related Skills

- **anti-detect-browser** - the SDK, profiles, personas, proxies and REST API that create the setup being verified here
- **browser-mcp-agent** - MCP server mode, for letting an AI agent drive a single profile itself
