# Phase 3 history copy repair

## Outcome and scope

**Both reproduced P2 copy/restore blockers repaired with actual RED/GREEN.** The corrected source/tests are frozen and the bounded selections have been rerun. Ready for independent review, **not self-approved, merged or activated**. Generic Hermes worker, not BUREAU/CLIENT.

Checkout: `C:/Users/sibag/hermes-phase3-authority-history`; branch `feat/phase3-authority-history`; HEAD unchanged at `09109fec98016ffd7fef8622223073d296c02fa4`.

Read QUALITY-REVIEW in full, original BRIEF, latest RETRY-RESULT, FINISH-API-COVERAGE and the unchanged `run_isolated.py` before execution. Read repository development/testing guidance. Report saved early, then updated with completed execution. No previous report, freeze, log or review was overwritten.

## Exactly what changed

1. **`hermes_cli/profiles.py` — clone-all destination ownership.** Replace the final `shutil.move(staged, profile_dir)` with `shutil.copytree(staged, profile_dir, symlinks=True)` using default exclusive destination creation. An existing/racing directory cannot become a container or be reused. FileExistsError propagates before stripping, metadata initialization or registration. The actual staged authority check remains before placement. Copying needs no cross-filesystem rename. No second `exists()` check is presented as a fix.
2. **`hermes_cli/backup.py` — ZIP destination-scope preflight.** Strip the detected wrapper, normalize through the same resolved destination/containment boundary as extraction, then classify the destination-relative path before mapping selected boards to generated safe staging names. External-provider members retain their separate extraction boundary. Escaping paths remain blocked/reported by extraction. Standard board sidecars (`-wal`, `-shm`, `-journal`) remain grouped with their member; target authority refusal remains before overlay.
3. **`hermes_cli/kanban_history.py` — necessary shared location helper only.** Add `is_standard_kanban_database`, used by ZIP preflight and existing tree candidate checking. It recognizes root `kanban.db`, `kanban/**/kanban.db`, recursively nested `profiles/<name>` roots, and host-native case normalization. No schema, journal, writer, reservation, UDF or history-performance changes.
4. **New `tests/hermes_cli/test_kanban_history_copy_qualityfix.py`.** Dedicated real candidate/filesystem/SQLite tests; previous five history test files remain unchanged.

The destination collision test calls real `create_profile`; after real staged validation it creates a competing directory with config, `.env` and a deliberately non-PID runtime sentinel. It asserts every competing file's bytes and file set remain identical, no nested clone, no registration, and FileExistsError. Only competing-creator timing and process-registration boundaries are injected; no real service or worker starts.

ZIP matrix accepts ordinary SQLite, unrelated `authority_roles` schemas and actual enrolled test fixtures **outside** standard locations (skills, plugins, a nested-profile skill directory). It refuses enrolled and malformed boards at root, named board, profile root and nested-profile named board locations, including `.hermes/` wrappers and an in-root `skills/../kanban.db` normalization case. Additional compatibility cases prove authority present only in a real WAL is refused with and without wrappers, escaping archive members preserve an outside disposable sentinel, and clone uses real copy I/O when rename is simulated to fail EXDEV.

## Actual RED/GREEN and frozen execution

All artifacts below are new under `.phase3-evidence/`:

| Log | Actual result |
|---|---|
| `qualityfix-red.log` | **13 failed, 26 passed**, exit 1, before production edits. Collision mutates/nests competing target; twelve unrelated-authority-table/enrolled nonstandard ZIP variants are wrongly refused. |
| `qualityfix-green.log` | Same original 39 cases: **39 passed**, exit 0. |
| `qualityfix-green-defenses.log` | Original cases plus four defense/compatibility cases: **43 passed**, exit 0. |
| `qualityfix-frozen-history.log` | All five existing history files plus dedicated copy repair file: **164 passed**, exit 0. |
| `qualityfix-frozen-adjacent.log` | Existing bounded boards/init/transaction/dashboard/backup/quick-restore selection: **96 passed, 1 skipped**, exit 0. |
| `qualityfix-frozen-profiles.log` | Existing `clone_all or import` selection: **21 passed, 1 failed, 133 deselected**, exit 1. |
| `qualityfix-frozen-imports.log` | Existing origin smoke test: **1 passed**, exit 0. |

Nonduplicated frozen total: **282 passed, 1 failed, 1 skipped, 133 deselected. NOT ALL GREEN.** Focused RED/GREEN runs are not added to that total. No test timeout occurred.

The unchanged Windows failure is `TestExportImport.test_export_default_handles_broken_symlinks`, `test_profiles.py:1439`: WinError 1314 while creating the fixture symlink, before production export. The unchanged skip is `test_backup.py:846`, “POSIX file permissions only.” No blanket skip, privilege change or assertion disabling.

### Exact commands

Executed from the checkout with outer `python -B`; the installed venv's Python 3.11.4 is used read-only for runtime children. Exact resolved child command, source origins, actual tempfile root and exit are also recorded in every log.

```text
python -B .phase3-evidence/qualityfix-verify.py baseline
python -B .phase3-evidence/qualityfix-verify.py red-freeze
python -B .phase3-evidence/qualityfix-run.py qualityfix-red tests/hermes_cli/test_kanban_history_copy_qualityfix.py -q --tb=short
python -B .phase3-evidence/qualityfix-run.py qualityfix-green tests/hermes_cli/test_kanban_history_copy_qualityfix.py -q --tb=short
python -B .phase3-evidence/qualityfix-run.py qualityfix-green-defenses tests/hermes_cli/test_kanban_history_copy_qualityfix.py -q --tb=short
python -B .phase3-evidence/qualityfix-verify.py freeze
python -B .phase3-evidence/qualityfix-run.py qualityfix-frozen-history tests/hermes_cli/test_kanban_authority_history.py tests/hermes_cli/test_kanban_history_pass2.py tests/hermes_cli/test_kanban_history_strict.py tests/hermes_cli/test_kanban_history_lifecycle.py tests/hermes_cli/test_kanban_history_profile_copy.py tests/hermes_cli/test_kanban_history_copy_qualityfix.py -q --tb=short -rs
python -B .phase3-evidence/qualityfix-run.py qualityfix-frozen-profiles tests/hermes_cli/test_profiles.py -q --tb=short -k 'clone_all or import'
python -B .phase3-evidence/qualityfix-run.py qualityfix-frozen-imports .phase3-evidence/finish_imports_test.py -q --tb=short
python -B .phase3-evidence/qualityfix-verify.py verification
```

Adjacent selection was replayed exactly (no old launcher run/log overwritten):

```bash
python -B -c "import ast,pathlib,subprocess,sys; e=pathlib.Path('.phase3-evidence'); args=ast.literal_eval((e/'retry-frozen-adjacent.log').read_text().splitlines()[0].removeprefix('COMMAND: '))[4:]; raise SystemExit(subprocess.call([sys.executable,'-B',str(e/'qualityfix-run.py'),'qualityfix-frozen-adjacent',*args]))"
```

The fully expanded adjacent node list is in `qualityfix-frozen-adjacent.log`. Four frozen commands ran independently in parallel, each with its own fresh state; source editing had stopped.

## Isolation, freeze and manifest

New support files: `.phase3-evidence/qualityfix-run.py` and `qualityfix-verify.py`. The runtime wrapper loads the **unchanged reviewed launcher** with `runpy`, overrides only its environment function's STATE global in memory, and uses that allowlisted environment. Each label gets a previously nonexistent `.phase3-evidence/state/qualityfix-<label>/` directory and an exclusively created log. It duplicates the launcher child origin/temp checks and pytest flags, adds explicit profiles/backup/history origins, and bounds each child to 180 seconds. No live environment credentials or board/profile selectors are inherited. HOME, USERPROFILE, HERMES_HOME, APPDATA, LOCALAPPDATA, TEMP, TMP and TMPDIR are all fresh worktree-local paths; Windows essentials retained; bytecode/user-site/plugin auto-loading disabled. This is cooperative containment, not an OS sandbox or CI/full-suite equivalence.

New static manifests: `qualityfix-baseline.json`, `qualityfix-red-freeze.json`, `qualityfix-freeze.json`, `qualityfix-verification.json`. Baseline checked every previous retry source input against current bytes before the repair. Preservation map has **117 existing paths**; only the three scoped production files may differ. New tests/report/evidence are listed separately. Final verification requires all **32 explicit source/test/support hashes** to equal the frozen map, Python compilation without runtime imports/pyc, `git diff --check`, empty index, expected HEAD/branch and exactly the three installed baseline hashes. The manifest includes the complete nonignored modified/untracked path listing, all new preceding evidence hashes and this report's hash. This is an explicit bounded input/preservation map, not exhaustive transitive-import/repository attestation.

Freeze SHA-256 (`qualityfix-freeze.json`):

```text
0019bf4fbc92d16cd2bca284039b5964c486d221173708adc604a818dd1f5600
```

Repair source/test SHA-256:

```text
hermes_cli/profiles.py
ac21fc6fca21216266a190fe1148d2f5f7f27be8d8aeef4771639aad195e8b08
hermes_cli/backup.py
4164d7955511b69d73dd7a3ed7205f9b12b97b2d4c360e9bd2b6535509d94cb2
hermes_cli/kanban_history.py
501aad6097cfd836237eb3e1ff938a58567d81f7f932d60176b0fc1ba3bb5187
tests/hermes_cli/test_kanban_history_copy_qualityfix.py
95f1ec48232883fa01cec8d754f8c1f4acf65e540dab7cb966e74e289ebbb6b3
```

Exactly three installed files compared read-only against `installed-before.json`, at `C:/Users/sibag/AppData/Local/hermes/hermes-agent`: **hermes_cli/kanban_db.py, hermes_cli/kanban.py, hermes_constants.py**. No whole-install or live-state attestation, and no claim that installed profiles.py/backup.py were part of that three-file baseline.

## Remaining gates and limitations

- Independent review of this corrected frozen candidate is still required. Existing historical review verdicts are not rewritten as acceptance. No full repository suite, pinned lint/security gate, deployment or activation approval.
- Exclusive destination creation is not an atomic whole-tree publication guarantee. Copying a validated stage incurs a second copy, and a subsequent I/O failure may leave an owned partial destination, as ordinary copytree does. No rollback/cleanup/recovery redesign. Cross-device compatibility is exercised with a rename-failure boundary and real copy I/O, not a physical second mounted volume.
- Existing arbitrary-copy/custom-location/symlink-alias/online-copy/rollback continuity and surviving execution-tree exclusions remain unsupported. ZIP sidecar grouping retains the original exact-member convention; no new arbitrary archive-alias coherence guarantee. No performance/UDF changes or claims.
- No real task workers, gateways/services, PID signals, sends, installs, staging, commits, pushes, merges, live board/profile/config writes or installed-source modifications. Disposable SQLite contender/self-exit probes in the preexisting history suite remain test processes, not workers. All authored state stayed in this checkout.
- If independent verification is interrupted or discovers drift: use final manifest/logs to locate the exact frozen inputs, investigate mismatches before rerunning, and use new unique evidence labels/state roots; never overwrite these artifacts. Existing known Windows failure/POSIX skip remain explicit gates, not reasons to conceal new failures.
