# deferred/private-feature-mixed — residual private LINES of files already in feature PRs

**Disposition:** these 7 patches are the leftover private lines of files whose PUBLIC
content already ships in open feature PRs (e.g. agent/agent_init.py is in #50296,
#50073, #49917, #49184, #48065). The private lines (agy-cli wiring, overlay glue,
the opus-context private test) were deliberately stripped from those public PRs and
cannot be re-published. Per [id=92873] they stay deferred.

**Pull:** `git apply --3way deferred/private-feature-mixed/*.patch` (reference/private
overlay only; not for upstream).
