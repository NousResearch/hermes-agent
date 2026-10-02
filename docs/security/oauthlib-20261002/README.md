# OAuthlib security follow-up

This change is stacked on Hermes PR #105 at `dd7a4c94287f72745f4fb3cdf7ebdbd5e17421da`.
It fixes the two remaining oauthlib advisories by locking the transitive package
at 4.0.0, with a dated package-only exception to the 14-day release quarantine.
The release was published on 2026-09-28; the window ends on 2026-10-12.

- [GHSA-hj66-6f7g-4r5v](https://github.com/advisories/GHSA-hj66-6f7g-4r5v): revocation JSONP callback injection.
- [GHSA-xpv3-w29h-x7cv](https://github.com/advisories/GHSA-xpv3-w29h-x7cv): PKCE comparison timing leak.

[PR #105's scanner job](https://github.com/joojalre/hermes-agent-almorshednet/actions/runs/36987068250/job/110774410095)
reported **two Medium findings in oauthlib**, without a remaining PyJWT finding.
Its title/body's older 70-of-71 estimate is not the current scanner result.

## Verification

- Existing Hermes PM was used with an explicit isolated source, output, cache, home and store.
- PM lock consistency check passed. No live Hermes environment was updated.
- Native Windows `scripts/run_tests.ps1` passed 68 existing JWT authentication tests.
- Two new contracts passed through real Google OAuth `Flow` and `requests-oauthlib` APIs:
  authorization-code/PKCE state and token exchange/refresh. Only HTTP transport was replaced.
- A local evidence test passed PyJWT 2.15.0 options-dict immutability and expired-token rejection.
- OSV Scanner 2.6.0, with `--all-vulns`, scanned all five explicit lockfiles and returned exit 0
  with zero findings. [Results](osv-results.json) and [input hashes](verification.json) are saved.
  Hashes describe the working-tree bytes used for that local scan, not Linux checkout bytes.

## Boundaries

CI for this follow-up remains Pending until its exact head completes. The scanner's
`fail-on-vuln` setting is unchanged. Local scan acceptance does not prove merge,
live runtime adoption, or real-account Google authorization.

The original #103 stderr-tail assertion was not reproduced locally. Its test requires
POSIX wrappers and `/proc`; native collection also lacked optional aiohttp. An expanded
base smoke run found an existing Windows executable-mode assertion in the skills-sync
suite. Neither failure was changed by this dependency follow-up.

#82 remains open; current main push CI is intentionally SKIPPED. This follow-up does
not change its branch, close it, or introduce a successful skip sentinel.
