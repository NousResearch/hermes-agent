# Wave-1 witness cluster map for a TEST-file godfile (tui_test s4-w1a, 2026-08-05)

The 5×2×3 method also decomposes giant TEST files (e.g. `tests/test_tui_gateway_server.py`, 16,186 lines / 504 tests). Wave-1 witnesses map the shard's tests into **observable modules** (`tests/<pkg>/test_<subject>.py`) instead of mixin modules. Proven on tui_test shard s4 (lines 9712-12948, 80 tests → 14 clusters → 14 proposed modules).

## Shard-plan builder (agreement stage, tui_test 2026-08-05) — w2 schema variants + dedup + audit

After w1-w3 land, the plan builder merges the verdicts into `shard-plan.json`. The w2 verdict shapes vary per shard and each needs a handler:

- `canonical_clusters` (or `clusters`) with `tests` as **dicts `{name, span}`** (s1) OR **plain name strings** (s3-s5) — normalize `t["name"] if isinstance(t, dict) else t`.
- **s2 style**: `per_cluster_adjudication` entries carry cluster name-with-count, `canonical_module`, and `w1a`/`w1b` MODULE names but NO test list. Resolve the tests from the w1 maps by module-name match: split `canonical_module` on `/`, strip `.py`, scan both w1 verdicts' clusters for a `proposed_module` containing that stem, return its tests. A cluster whose every test is already claimed by an earlier shard is a seam-reference — mark `seam_reference: True` and subtract from the region count rather than double-claiming.
- **Seam-crosser dedup across shards** (e.g. `test_session_not_running_before_agent_ready_emits_error_event`, body 9709→9772 spanning s3/s4): the plan builder must claim it in the FIRST shard that lists it and treat later listings as references. Track a `_claimed` set during the build; pop it before JSON serialization (sets aren't serializable).
- **Plan-vs-live audit is the acceptance gate**: compare `set(plan tests)` against a live-file regex inventory (`re.findall(r"^(?:    )?def (test_\w+)", src, re.M)` — note the optional 4-space indent for class methods). The contract is **unique-set equality: 0 missing, 0 extra** (e.g. 504 == 504), NOT raw count equality — a raw total of 505 with 504 unique is the documented seam relist, not an error. Plan region test_counts must sum to the unique total after seam-reference subtraction.

## Schema — adapt the source-map keys, keep the top level EXACT

Reference schema (from the campaign's first source witness JSON, e.g. `godfile/w1/s1-w1a.json`):

- Top level (EXACT key set, no extras): `shard, witness, wave, file, clusters, slice_order, cross_region_dependencies, risks, unverifiable`.
  - **PITFALL: additive top-level keys break strict verifiers.** An added `"note"` key tripped an exact-key-set check (`assert set(d) == top`) — remove it and fold the content into `cross_region_dependencies`/`unverifiable` instead.
- Cluster keys (task-required additions for test maps): `name, purpose, tests, line_span, classification, proposed_module, depends_on, shared_state, entanglements, evidence, rationale, observability`.
  - `tests` = list of `{name, line}` (def lines, 1-indexed) — replaces the source map's `methods`.
  - `observability` is the TEST-file-specific field: how the cluster's behavior is observed (server._emit event tuples, JSON-RPC error code/message pairs, result envelopes, env vars, capture lists) — the contract the split must preserve.
  - `rationale` = why these tests form one module (shared harness, one RPC handler, one issue lineage).
  - `classification` = `"move"` for extraction candidates; note partial-extraction families separately in entanglements.
  - `proposed_module` = `tests/tui_gateway/test_<subject>.py` — the target dir already exists with sibling `test_*.py` files; naming convention follows siblings.

## Shard boundaries and seam rules (same as source maps, applied to test defs)

- Compute boundaries from the shard script (`per = total // 5`; s4 = 9712-12948 for a 16,186-line file). Grep the def index with an awk line filter (`grep -nE "^\s*(async\s+)?def test_" file | awk -F: '$1 >= S && $1 <= E'`).
- **Seam-OWNER** (def in the neighbor region, body tail in yours — e.g. def @9709, 3 lines above the 9712 boundary, body to 9772): list it as a MEMBER of its cluster with a seam note; never propose extracting it standalone; tell the neighbor witness to cluster the same def.
- **Seam-CROSSER** (def in yours, body runs into the next region — e.g. def @12927, body to ~12961 past 12948): the cluster must move WHOLE; the neighbor witness must cluster the same family. Verify the crossing with `sed -n '<boundary-20>,<boundary+20>p'` on both sides.
- Counting discipline: 79 in-range defs + 1 seam-owner = 80 attributed tests. Verify with a script that (a) every in-range def is mapped, (b) every mapped line is an in-range def OR the seam def, (c) no duplicates.

## Cross-region family greps — document partial extraction and filename collisions

Some subjects are tested across the WHOLE file, not just your shard. Grep the full file by test-name family (`def test_.*prompt_submit`, `.*compress`, `.*browser_manage`, `.*session_create`, ...) and record in `cross_region_dependencies`:

- **Partial-extraction families** (prompt.submit: compute-host/truncate/history-version tests at 206-9495 in s1-s3, completion family at 11948-12101 in s4, persist/heap-trim at 16073-16116 in s5): your module is a SLICE — either other witnesses emit into the same module or the name must be disambiguated (e.g. `test_prompt_submit_completion.py`). State this in entanglements so the plan doesn't look incomplete.
- **Filename-collision risk** (compress: core `_compress_session_history` tests at 7310-7340 + session.compress suite at 7373-7636 belong to s3; your s4 slice is only the 'here [N]' parsing family): coordinate module naming with the neighbor witness (it should take `test_session_compress.py`, you keep `test_compress_session_history.py`, or vice versa).
- **Whole-family continuation** (browser.manage: 22 tests total, 11 in s4 + 11 in s5): the extraction must be ONE module in one PR or two coordinated moves; the s5 witness MUST cluster the same family.

## Observability + coupling seams specific to TEST splits

- **monkeypatch binding is module-attr**: tests patch `server._AGENT_BUILD_WAIT_SLICE`, `server._apply_model_switch`, `server._compress_session_history`, `server._get_db` etc. ON the server module. Splitting TEST code is safe ONLY if the server module (`tui_gateway.server`) is never renamed — do NOT propose renaming the server module in the same campaign.
- **Autouse fixtures in the god test file must survive the split**: `_neuter_agent_prewarm_timer` (module-level autouse fixture) stubs `server._schedule_agent_build` for every test EXCEPT those marked `@pytest.mark.real_agent_prewarm` (which opt back in to arm the 50ms deferred-build timer). Extracted modules must keep both the marker and the fixture — move it to `tests/tui_gateway/conftest.py` or import it — or the deferred-build tests can't arm the timer deterministically and flake (documented `'tip' == 'cont_tip'` flake class).
- **Parent conftest keeps working via pytest hierarchy**: autouse fixtures in `tests/conftest.py` (e.g. `_isolate_hermes_home` pinning HERMES_HOME) serve extracted subdir modules automatically — note it, no action needed.
- **Module-level test helpers move with their exclusive cluster**: `_partial_compress_agent` + `_PARTIAL_FAKE_HISTORY`/`_PARTIAL_COMPRESSED_HEAD` (compress cluster), `_ImmediateThread` (prompt_submit cluster, sits BETWEEN two other clusters' spans), `_stub_urlopen`/`_stub_urlopen_capture` (browser_manage cluster). Verify exclusivity by reading the spans around them; a helper used by no test outside its span moves with the cluster.
- **Process-global state needs isolation-pattern preservation**: `server._sessions` (global dict — some tests snapshot/clear/restore to defeat sibling leakage, documented flaky under `-j 8`), `tools.approval` registry (process-wide notify registration), `BROWSER_CDP_URL` real process env, `sys.modules['hermes_state']` / `sys.modules['tools.browser_tool']` swaps via `patch.dict`. Splitting into parallel test modules RAISES the leak surface — keep the isolation patterns verbatim (own_key scoping, previous_sessions restore).
- **String patch paths are import-time-bound**: `patch.dict(sys.modules, {'tools.browser_tool': fake})` and `patch('hermes_cli.inventory.build_models_payload', ...)` must keep identical strings in the extracted module or the fake stops binding.
- **Error-code matrices are the observable contract**: session.delete pins literal codes 4006/4023/4007/5036; keep the exact-dict row equality assertions (e.g. active_list rows) intact — the desktop GUI consumes those field names.

## Verification

- Write the JSON early (persist-early doctrine), then run a throwaway integrity script against the live file: JSON parses; top-level + cluster keys equal the reference sets exactly; every listed `{name, line}` exists at that line with the exact def name; all in-range defs mapped + seam def mapped + no extras/duplicates; every test line inside its cluster's `line_span`; `slice_order` covers all cluster names; proposed modules unique; identity fields (shard/witness/wave) correct; meta sections non-empty.
- When the harness demands fresh passing verification evidence for the changed path and re-flags it: re-run the tempfile-bootstrap verifier (`tempfile.mkstemp(prefix='hermes-verify-', suffix='.py')` in the user Temp dir, write, `subprocess.run([sys.executable, path])`, `os.unlink` in `finally`) as the LAST action of the turn, against the exact current bytes. If you edited the artifact (e.g. removed the `note` key) after the first verify, the earlier evidence is stale — re-verify after the final edit.
- The repo test file is the ground truth — read it read-only; only the campaign JSON path and the temp dir are written.
