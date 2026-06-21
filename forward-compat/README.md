# forward-compat/ — the v0.17.0-clean resolved forms, as PR-resident pullable patches

The two feature PRs whose files overlap v0.16.0 -> v0.17.0 upstream drift
(#50056, #48069) cannot pre-resolve that conflict *inside the PR itself*: each PR
is cut off `origin/main`, where the conflict does not exist. The resolved,
v0.17.0-clean form only materializes when the change is replayed onto v0.17.0.

Previously that resolved form lived only as loose fork branches
(`forward-compat/<n>-on-v0.17.0`). The Council noted that "lives on a branch, not
in a PR" is weaker than the goal's "every src delta lives in a separate PR usable
on top of v0.17.0." So the resolved forms are now ALSO committed here as pullable
patches **inside this draft PR (#50111)**.

## The patches

| patch | feature | applies onto v0.17.0 |
|-------|---------|----------------------|
| `50056-on-v0.17.0.patch` | HERMES_SQLITE_DRIVER selection + driver-agnostic Row/error handling | `git apply --check` exit 0 (verified) |
| `48069-on-v0.17.0.patch` | MCP keepalive in-flight race (keep-both with upstream `_keepalive_probe` refactor) | `git apply --check` exit 0 (verified) |
| `50073-on-v0.17.0.patch` | compression oversized-message offload (conversation_loop/agent_init drift) | `git apply --check` exit 0 (verified) |

Each patch is `git diff v0.17.0(2bd1977d8)..forward-compat/<n>-on-v0.17.0` — i.e.
exactly the feature's contribution expressed against a pristine v0.17.0 tree, with
the upstream-drift conflict already resolved.

## How an operator pulls them

```bash
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
git checkout -b on-v0.17.0 "$V017"
git apply forward-compat/50056-on-v0.17.0.patch
git apply forward-compat/48069-on-v0.17.0.patch
git apply forward-compat/50073-on-v0.17.0.patch
# ...plus the 38 clean feature PRs (see ../APPLY-ORDER.md)
```

The companion fork branches (`forward-compat/50056-on-v0.17.0` @ `e55b6481d`,
`forward-compat/48069-on-v0.17.0` @ `bcf9ff2eb`, both with v0.17.0 as ancestor)
remain published as the git-native form for anyone who prefers `git cherry-pick`.
Both forms encode the identical resolution; this directory makes the resolved form
PR-resident so nothing about the v0.17.0 disposition lives outside a pullable PR.
