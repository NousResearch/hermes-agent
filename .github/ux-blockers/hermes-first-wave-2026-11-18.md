# Hermes first-wave UX audit — November 18, 2026

**10 open report-evidenced problem clusters; 7 qualifying merged clusters across 8 product PRs.**

Scope is October 1–November 18 in Bangkok, with one September carry-in. This bounded sample is a lower bound, not a census. Metadata was refreshed on November 18. Current release-blocker status is not assessed by this audit. Qualifying a historical report does not automatically confirm a blocker for a release candidate.

The [structured dataset](hermes-first-wave-2026-11-18.json) retains recovery descriptions, source URLs, current PR heads and merge identities, exclusions and caveats. It is independent of the live release ledger.

## Open reported problems

| Cluster | Problem and reported recovery | Reports | Product fix candidates |
| --- | --- | --- | --- |
| O1 | Windows fixture-copy failure in install/update. Installer reporter recovered only by enabling OS long paths as administrator and rebooting; updater reporter says ordinary retry fails. #151274 is corroborating MSIX/plugin evidence, not a separate core-flow count. | [#150277](https://github.com/NousResearch/hermes-agent/issues/150277), [#151274](https://github.com/NousResearch/hermes-agent/issues/151274), [#156983](https://github.com/NousResearch/hermes-agent/issues/156983) | No audited product-fix PR |
| O2 | Windows Desktop update never relaunches. Reported verified recipe kills processes, reruns CLI update with --no-gateway-restart, starts the scheduled task, and removes a stale pending marker. | [#156217](https://github.com/NousResearch/hermes-agent/issues/156217) | [#156237](https://github.com/NousResearch/hermes-agent/pull/156237) (open) |
| O3 | Interrupted Windows update destroys recoverability. Reporter recovered through manual environment rebuilding and marker cleanup. | [#149520](https://github.com/NousResearch/hermes-agent/issues/149520) | No audited product-fix PR |
| O4 | Windows reboot strands Telegram. Deleting the lock manually and then restarting gateway restored binding; restart alone did not help. | [#157025](https://github.com/NousResearch/hermes-agent/issues/157025) | No audited product-fix PR |
| O5 | Non-ASCII Windows profile blocks first install. Only forced UTF-8 script execution or an ASCII-only LOCALAPPDATA path recovers; ordinary retries repeat failure. | [#151581](https://github.com/NousResearch/hermes-agent/issues/151581), [#151600](https://github.com/NousResearch/hermes-agent/issues/151600) | [#148568](https://github.com/NousResearch/hermes-agent/pull/148568) (open) |
| O6 | Slow version probe treats healthy Desktop install as absent. Later manual relaunch can work; blobless-clone reporter used git fetch or disabled passive update checks. No Desktop Retry repairs the failed path. | [#148498](https://github.com/NousResearch/hermes-agent/issues/148498), [#155862](https://github.com/NousResearch/hermes-agent/issues/155862) | [#150540](https://github.com/NousResearch/hermes-agent/pull/150540) (open) |
| O7 | Smart App Control update leaves gateway down. Reporter manually repointed launch scripts/interpreter; normal follow-up could reapply the broken runtime. | [#148950](https://github.com/NousResearch/hermes-agent/issues/148950) | No audited product-fix PR |
| O8 | Subagent route corrupts returning Desktop Bot Chat. Gateway restart does not help; reporter repaired the route by hand. | [#154994](https://github.com/NousResearch/hermes-agent/issues/154994) | [#155020](https://github.com/NousResearch/hermes-agent/pull/155020) (open) |
| O9 | Windows logon task reports success without gateway. Only a manually added console-owning Startup-folder script works in the report. | [#136390](https://github.com/NousResearch/hermes-agent/issues/136390) | [#136397](https://github.com/NousResearch/hermes-agent/pull/136397) (open) |
| C1 | Source Desktop update. Fully quit Desktop, rerun hermes update in Terminal, reopen; installed commit matched runtime and warning cleared. | [#123737](https://github.com/NousResearch/hermes-agent/issues/123737) | [#128465](https://github.com/NousResearch/hermes-agent/pull/128465) (open) |

## Qualifying merged fixes

| Cluster | Problem and reported recovery | Reports | Product fix candidates |
| --- | --- | --- | --- |
| M1 | Profile OAuth false success. Repeated add/re-login did not persist; fix stores the independent profile grant and checks readback. | [#103694](https://github.com/NousResearch/hermes-agent/issues/103694), [#120177](https://github.com/NousResearch/hermes-agent/issues/120177), [#125501](https://github.com/NousResearch/hermes-agent/issues/125501) | [#134574](https://github.com/NousResearch/hermes-agent/pull/134574) (merged) |
| M2 | Valid Anthropic Console key sent as OAuth. Re-entering the same valid key cannot change misclassification; fix selects x-api-key. | [#133856](https://github.com/NousResearch/hermes-agent/issues/133856) | [#137140](https://github.com/NousResearch/hermes-agent/pull/137140) (merged) |
| M4 | CLI provider re-login advice cannot work. Suggested command fails; fix points to hermes model, which invokes vendor CLI login. |  | [#151957](https://github.com/NousResearch/hermes-agent/pull/151957) (merged) |
| M5 | Windows install caches incomplete Git forever. Fix detects broken cached helper and re-extracts atomically; native Windows PR proof shows HTTPS probe exit 128 before and 0 after. |  | [#145592](https://github.com/NousResearch/hermes-agent/pull/145592) (merged) |
| M6 | Packaged macOS update downloads but never installs. Fix makes update payload pycs owner-writable; repeated normal updates otherwise repeat the abort. | [#138659](https://github.com/NousResearch/hermes-agent/issues/138659) | [#148192](https://github.com/NousResearch/hermes-agent/pull/148192) (merged) |
| M7 | Remote Desktop cannot create any new chat. Fix negotiates authority wire from gateway.ready. Normal retry cannot change protocol. |  | [#148219](https://github.com/NousResearch/hermes-agent/pull/148219) (merged) |
| M9 | Lost turn silently pauses returning conversation. Automation is settled as interrupted; every supported surface gains discard; Desktop reconnect/relaunch surfaces Needs your input and Discard runs queued send exactly once. | [#139532](https://github.com/NousResearch/hermes-agent/issues/139532) | [#141357](https://github.com/NousResearch/hermes-agent/pull/141357) (merged), [#145078](https://github.com/NousResearch/hermes-agent/pull/145078) (merged) |

## Excluded recovery-ambiguous merged candidates

| Cluster | Problem and reported recovery | Reports | Product fix candidates |
| --- | --- | --- | --- |
| M3 | Installed CLI disappears in new login shell. Fix writes the guarded PATH line to the startup file Bash actually reads; prior normal reopen repeated failure. |  | [#131809](https://github.com/NousResearch/hermes-agent/pull/131809) (merged) |
| M8 | Gated remote Desktop rejects correct token on REST. Fix accepts the Desktop header via the session-token provider; re-entering a correct token did not change gate behavior. | [#147387](https://github.com/NousResearch/hermes-agent/issues/147387) | [#149623](https://github.com/NousResearch/hermes-agent/pull/149623) (merged) |

Related update-lock PR [#152895](https://github.com/NousResearch/hermes-agent/pull/152895) is recorded separately for O3. Cross-references, including the policy PR itself, receive no product-fix credit.

The new `just ux-report` command refreshes lifecycle/SHAs for these same records. It does not rerun native journeys, silently update historical classifications or establish that a merged fix reached a release. Prior excluded candidates remain excluded until recovery evidence changes.

Methodology and OpenClaw comparison are in the [pinned original audit](https://github.com/100yenadmin/hermes-agent-for-upstream-PR-only/blob/6ac3291a188161c3f658a0f7f011a9660a2ced7d/ux-release-gate-evidence.md).
