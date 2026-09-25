# Deferred plugin catalog admissions

These entries are retained as review references and are not discoverable catalog entries.
They were new additions in upstream checkpoints integrated by this fork. Their pinned
revisions could not pass the existing catalog admission checks.
No installed plugin is removed or disabled, and the validator and CI gates are unchanged.

| Plugin | Pinned revision | Admission blocker |
| --- | --- | --- |
| hermes-tailscale | 6dad1d5aac443a196f7ba83736a1f4c9e3a75048 | Lines 1521 and 1532 import a blob module and inject a script element to execute xterm. |
| hermes-terminal | 43658bf3c883178bf035d2163f7f541500d17407 | Lines 326 and 331 import a URL and inject a script element to execute xterm. |
| entropicmem | 437c89b7da625928c41171e88d86f45af41342e1 | Source repository returned 404 on 2026-09-24; CI could not clone the pinned revision, so source validation did not run. |

The original entry YAMLs are preserved beside this document. These are specific
admission blockers, not a claim of malicious behavior. A later admission requires a
reachable source at its reviewed pin and passing `hermes plugins validate`, including
its Desktop surface check. See [catalog admission policy](../../plugin-catalog/README.md).

Evidence: [Desktop admission failure](https://github.com/joojalre/hermes-agent-almorshednet/actions/runs/35629759954/job/106439087915)
and [unavailable entropicmem source](https://github.com/joojalre/hermes-agent-almorshednet/actions/runs/36061033694/job/107839740244).
