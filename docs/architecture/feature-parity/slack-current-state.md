# Slack Feature Parity and @Hermes Tag — Current-State Reconciliation

Snapshot: `4a5b6dd4512a10c3c18da3e5b9e5c7fb681cbfbb` at `2026-08-20T19:11:08Z`.

## Corrected delivery state

- `candidate_blocked`: **1** — SLK3, owned by #91036 and blocked on #90307
- `gap`: **23**
- `candidate_unwired`: **0**
- `candidate_open`: **0**
- `on_main_unverified`: **0**
- `released`: **0**

The August 14 packets are preserved as evidence, not promoted into repository
delivery. The Slack packet's own release verifier failed closed because the
repository, independent-review, full-CI, and live-workspace receipts were absent.

## Current-main facts

- `plugins/platforms/slack/` contains four files: `__init__.py`, `adapter.py`,
  `block_kit.py`, and `plugin.yaml`.
- `plugins/platforms/slack/adapter.py` is 424,946 bytes.
- No `hermes_state/channel_governance/`, `plugins/platforms/slack/tag_mixin.py`,
  or `gateway/hermes_tag/` implementation is present on the pinned main tree.
- The five decomposition PRs remain open: #79712, #79713, #79714, #79800, and
  #80303. None is merged into the pinned snapshot; #79800 is non-mergeable.
- #80338 remains the flagship @Hermes Tag issue.
- #90978 records a newly confirmed parity failure: Slack's native
  `agent_session_stopped` action is not subscribed, so the Stop button does not
  cancel the running agent; status/title handling still uses legacy Assistant APIs.

## Packet evidence boundary

The Slack overlay contains the generic governance model, persistence,
authorization, dual tool enforcement, budgets, audit, Slack binding, Tag
behavior, native cards, bounded proactivity, history scope checks, management
API, shared Web/Desktop candidates, setup alignment, docs, and local tests.

The horizontal Hermes Tag packet contains an additive kernel and 73 passing
packet tests. Its own handoff says runtime insertion is a separate PR train.
Commit `dcf05b8ff59a81709f044c5aa6c8ce1026bc8d19` is not reachable from the
current fork or upstream repository.

Neither packet has an open implementation authority, a current-main runtime
consumer, an exact merged SHA, full repository CI, two independent exact-SHA
reviews, or a live Slack release receipt. Those rows remain `gap`.

## Publication order

1. Merge #90307, which supplies the external registry and validator.
2. Rebase and merge #91036 so #79772 has one machine-checkable authority.
3. Publish the horizontal kernel as additive HT-00/HT-01 slices with preserved
   provenance and independent review.
4. Revalidate and sequence #79712, #79713, #79714, #79800, and #80303 against
   current main; continue decomposition until both Slack god-files are below
   2,000 lines.
5. Publish governance/runtime slices with real definition, dispatch, provider,
   Slack identity, API, Web, and Desktop consumers.
6. Close #90978 through the canonical cancellation path and current Slack
   Assistant APIs.
7. Advance rows only from exact-head CI, independent review, and live-workspace
   evidence. Packet tests cannot perform that promotion.

## Contract source

The immutable 24-row identity comes from the approved implementation matrix.
Mutable delivery state lives in `slack.json`; semantic changes require a new
append-only registry revision authorized through #79772.
