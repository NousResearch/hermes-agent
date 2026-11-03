# Event Containment Order

Open while responding to a security event. Each step is `prepared` until its output is observed, and
the event stays open while any closing step is prepared.

## 1. Leaked secret

| Order | Step | Observed when |
| --- | --- | --- |
| 1 | Record the credential type, its scope, where it was pushed, whether the repository or log is public, and the first push time | the push is located in the history or the log |
| 2 | Issue the replacement and deploy it to every consumer | the service runs on the new credential |
| 3 | Revoke the leaked credential at its issuer | a call with the old credential is rejected; the revoke command's exit status only says the request was accepted |
| 4 | Audit the issuer's access log for the exposure window | the window's calls are listed and attributed |
| 5 | Rewrite history (`git filter-repo`, BFG) and coordinate every clone, only if the value must also leave the repository | the rewritten refs are pushed and forks and caches are named |
| 6 | Add prevention: push protection, a pre-commit secret scan, the ignore rule | the guard rejects a planted test secret |

Rotation comes before any rewrite. A rewrite does not reach clones, forks, CI caches, or a scraper
that read the public push in its first minutes, so a rewrite without a rotation leaves a live
credential in circulation. Per-credential issue-deploy-verify-revoke steps: `credential-rotation.md`.

## 2. CVE or advisory in a dependency

1. Read the advisory: id (CVE, GHSA), affected range, fixed version, vulnerable function or input.
2. Read the installed version from the lockfile, not the manifest range, including every transitive
   path to the package.
3. Reachability: find the repository call sites that reach the vulnerable function or accept the
   vulnerable input, or observe that none do.
4. Severity: start from the advisory score, then adjust. Reachable from untrusted input on a request
   path keeps it; reachable only from trusted input or tooling lowers it; unreachable is a scheduled
   upgrade, not an event. State the adjustment and its evidence.
5. Contain: upgrade to the fixed version, pin or override a transitive version, or disable the feature
   path; a workaround is recorded as a workaround, not a fix.
6. Close only when the fixed version is observed in the lockfile and the build that ships it.

When the fixed version is only in a new major, hand the migration to its own staged upgrade plan and
keep this event open, with the workaround in place, until that fix is observed.

## 3. License question

| License family | Typical obligation | Risk for closed distribution |
| --- | --- | --- |
| MIT, BSD, Apache-2.0, ISC | keep the notice; Apache-2.0 adds the NOTICE file and a patent grant | low |
| MPL-2.0 | modified files of the library stay open | low to moderate |
| LGPL | allow relinking; ship the library dynamically or provide objects | moderate |
| GPL-2.0, GPL-3.0 | the combined work is distributed under the GPL | high when distributed |
| AGPL-3.0 | network use counts as distribution | high for a hosted service |
| No license, custom, or "non-commercial" | no grant by default | needs counsel |

Read the declared license from the package metadata and the repository `LICENSE`, not from a README
badge. The verdict is fits, conflicts, or needs counsel, for the stated distribution model; anything
past the declared terms is a question for counsel.

## 4. Closure

Closed only when every closing step is observed: the rotation and the old credential's rejection for
a secret, the fixed version in the shipped build for a CVE, the recorded decision for a license. Name
each step that is still prepared.
