TH1-1177 — Repair local Hermes plugin compatibility and Desktop connector/update reliability.
Issue: https://linear.app/th1337/issue/TH1-1177/repair-local-hermes-plugin-compatibility-and-desktop-connectorupdate
Branch: dispatch/TH1-1177; delivery remote https://github.com/hteo1337/hermes-agent.git (upstream write denied).
PR: https://github.com/NousResearch/hermes-agent/pull/140155; implementation c163267a360; no merge/deploy.
Done: staging discovery, slow-update truth/deduplication and session availability payload fixed; TypeSafe local safe-key fix backed up.
Verification: 61 distinct Python + 8 Desktop tests pass; final RPC 6/6, staged checks 11/11, eslint and renderer tsc pass; controls red on base.
Review: sole tool-selected gpt-6.1-sol CONFIRMED after one fixed P1; review.txt + PR comment 6225303067 inspected; CI has no reported checks.
Next: await Teo merge/rollout approval; apply staged Toolaria companion patch only with matching host; verify live Desktop after approved rollout.
decided: preserve active chats/guards/secrets/profiles/Jev edits/jailbreak; GoodMem unchanged; authorized HA uninstall preserved.
Open questions: none; report updated at /Users/hteo/.hermes/cache/scratch/connectors-repair-20261016/repair-report.md.
