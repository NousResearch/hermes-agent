# Deferred plugin catalog admissions

These entries are retained as review references and are not discoverable catalog entries.
They were new additions in the upstream checkpoint integrated by this fork, but did not
pass the existing Desktop admission policy at their pinned revisions on 2026-09-21.
No installed plugin is removed or disabled, and the validator and CI gates are unchanged.

| Plugin | Pinned revision | Admission failure in catalog/desktop/plugin.js |
| --- | --- | --- |
| hermes-tailscale | 6dad1d5aac443a196f7ba83736a1f4c9e3a75048 | Lines 1521 and 1532 import a blob module and inject a script element to execute xterm. |
| hermes-terminal | 43658bf3c883178bf035d2163f7f541500d17407 | Lines 326 and 331 import a URL and inject a script element to execute xterm. |

The original entry YAMLs are preserved beside this document. These are specific policy
failures, not a claim of malicious behavior. A later admission requires a reviewed pin
that uses the supported SDK and passes `hermes plugins validate`, including its Desktop
surface check. See [catalog admission policy](../../plugin-catalog/README.md).

Evidence: [exact-head pinned-source validation](https://github.com/joojalre/hermes-agent-almorshednet/actions/runs/35629759954/job/106432813043).
