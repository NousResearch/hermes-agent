# deferred/copilot-limits — account-specific cap residuals

**Disposition:** account-sensitive caps (e.g. gpt-5.4 891K) per user rule [id=63592]:
ship-verbatim, do NOT generalize, do NOT open a public PR with account-specific values.
The contributable, generalized limit table already shipped in feature PR #49449
(_PROBE_VERIFIED_OVERRIDES). These 2 patches are the account-specific residual lines
held back from that PR.

**Pull:** `git apply deferred/copilot-limits/*.patch` only on an account where those
caps are correct; they are intentionally not generalized for upstream.
