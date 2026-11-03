# Repository Governance D1.0/D1.1 — Base and Binding Spike Decision Record

- **Date:** 2026-08-11
- **Candidate revision:** v3
- **Scope:** D1.0 isolated base + D1.1 SQLite/Darwin binding spike only
- **Status:** candidate awaiting independent review
- **Live activation:** not authorized
- **D1.2 or later:** not authorized
- **D2 tool seam:** not authorized
- **Commit/push/merge:** not authorized

Candidate v3 supersedes frozen candidates v1 and v2. Candidate v1 used
`PRAGMA trusted_schema=OFF` instead of the v0.11-mandated
`sqlite3_db_config(SQLITE_DBCONFIG_TRUSTED_SCHEMA,0,...)` operation and did
not exercise `loadExtensionEnabled` or `recursiveTriggers`. Candidate v2
fixed those exact API gaps but retained an overbroad D1.4 ordering-closure
claim and a non-replayable live-source status fingerprint. Exact v1/v2 bytes
remain in the external candidate archive; neither grants authority.

## 1. D1.0 isolated source identity

| Field | Exact value |
|---|---|
| Isolated workspace | `/Users/ykliu/Projects/hermes-agent-repo-governance-d1` |
| Source type | independent local clone with no hardlinks |
| Checkout mode | detached HEAD |
| Git remotes | none |
| Base commit | `af250d84948179834820a62bfd870c0df6f264a1` |
| Base tree | `6865279a06e3bbab8dbf91b330dc2455e402b204` |
| Base rationale | exact parity with the current live runtime commit; no fetch or tracking-ref promotion |
| Live source | `/Users/ykliu/.hermes/hermes-agent` |
| Fresh isolation probe workspace | `/Users/ykliu/Projects/hermes-agent-repo-governance-d1-isolation-probe-v3` |
| Replayable producer | `scripts/d1_isolation_probe.py`; SHA-256 `a86ad03d552b34ca4146b74547d55e3baa9b0cb5486ac5c60be5039d4ea1256d` |
| Producer receipt | `d1-isolation-probe-v3-receipt/receipt.json`; SHA-256 `cf623f528f2cc0b76916afb0641b001ccdb1c20abf249e11849e0ae9842c2882` |
| Git version | `git version 2.50.1 (Apple Git-155)` |
| Before/after result | all eight independently hashed source-state fields byte-equal |

The fresh producer captured exact raw stdout before and after for HEAD, tree,
NUL-separated index stages, NUL-separated porcelain status including all
untracked paths, tracked binary diff, staged binary diff, NUL-separated
untracked path set and recursive submodule status. It then made a second
`--no-local --no-hardlinks` clone, removed its origin, detached at the exact
base and proved: clean status, zero remotes, no object alternates and no
regular file with `st_nlink != 1`. All eight source-state fields matched.

No fetch, push, commit, merge, observed live source content/index/status
change, runtime restart, profile/config edit, database enrollment, GitHub
mutation or deployment occurred.

The current live install remains a dirty and diverged operational checkout. This spike did not normalize, clean, stash, reset, copy, commit or otherwise alter it.

## 2. Binding feasibility evidence

### 2.1 Live stdlib binding

```text
Python                         3.11.15
sqlite3 module                  2.6.0
SQLite runtime                  3.53.1
SQLite source ID                2026-05-05 10:34:17 c88b22011a54b4f6fbd149e9f8e4de77658ce58143a1af0e3785e4e6475127e9
Connection.setconfig           absent
SQLITE_DBCONFIG_DEFENSIVE      absent
```

**Disposition:** rejected for the reviewed evaluator contract. The Python 3.11 stdlib API cannot set and exactly read back `SQLITE_DBCONFIG_DEFENSIVE`.

### 2.2 Exact APSW prototype candidate

Installed only into the disposable workspace virtual environment:

```text
apsw==3.53.1.0
SQLite runtime=3.53.1
SQLite source ID=2026-05-05 10:34:17 c88b22011a54b4f6fbd149e9f8e4de77658ce58143a1af0e3785e4e6475127e9
```

Observed API behavior:

```text
config(SQLITE_DBCONFIG_DEFENSIVE, -1) -> 0
config(SQLITE_DBCONFIG_DEFENSIVE, 1)  -> 1
config(SQLITE_DBCONFIG_DEFENSIVE, -1) -> 1
```

Installed candidate identity:

```text
apsw/__init__.cpython-311-darwin.so
SHA-256 679dec646bf89e76bb3b636dfc6ffd80695196f84f842ab5795abe4ea2d297b8

apsw-3.53.1.0.dist-info/METADATA
SHA-256 836e9d2f3cdd6d84dc00449822debf5ca23ae13bb7e1f66b83f707eadf74edc7

apsw-3.53.1.0.dist-info/RECORD
SHA-256 9eb3a4b381978ac468abefe9936c4554f682a831322d4b47fc0c5fc8f9bd8c5f
```

These installed-file hashes identify this disposable probe only; a future dependency decision must bind the original wheel artifact and its provenance before installation.

`otool -L` reported only `/usr/lib/libSystem.B.dylib` for the installed APSW
extension and no dynamic `libsqlite3`, which is consistent with a bundled
SQLite candidate. This is not a substitute for original-wheel provenance,
per-platform artifact review or a future exact packaging decision.

APSW and the stdlib report the same SQLite version and source ID but different compile-option sets. For example:

```text
stdlib: MAX_ATTACHED=10
APSW:   MAX_ATTACHED=125
```

**Required consequence:** a future D1 store/evaluator may not verify with stdlib SQLite and operate with APSW, or vice versa. Verification and operational connections must use one exact frozen binding/library/profile. Same SQLite source ID alone is insufficient identity.

**Prototype recommendation:** use exact-pinned `apsw==3.53.1.0` for the next separately approved D1 slices, subject to:

1. independent supply-chain/license/wheel review;
2. exact wheel/binary hash capture for every supported platform;
3. full compile-options/profile freeze;
4. explicit packaging decision in `pyproject.toml` and lock regeneration only after separate approval;
5. fail-closed behavior when APSW or its exact profile is unavailable.

This record does not approve APSW as a production dependency and does not modify project metadata.

### 2.3 Rejected/held alternatives

| Option | D1.1 disposition | Reason |
|---|---|---|
| Python 3.11 stdlib only | reject | defensive-mode API/readback absent |
| CPython-private connection-pointer extraction | reject | unsupported ABI, memory-safety and portability risk |
| use stdlib for verification plus APSW for operation | reject | distinct compile-option/evaluator identities |
| direct system `libsqlite3` helper | hold | API exists, but a second library/connection model increases implementation and packaging complexity |
| upgrade minimum Python solely for `Connection.setconfig` | hold | broad product/runtime change beyond D1.1; does not by itself freeze SQLite library identity |
| exact APSW binding | recommend for disposable D1 prototype | exposes required API and can bind one evaluator identity |

## 3. Darwin witness capability evidence

Observed on Darwin arm64:

```text
os.O_NOFOLLOW                 present
os.O_DIRECTORY                present
libc.openat                   present
fcntl.F_FULLFSYNC             present (51)
libc.acl_get_fd_np            present
libc.acl_free                 present
ACL_TYPE_EXTENDED             0x00000100 (Darwin SDK sys/acl.h)
```

A disposable witness capability test proved only:

1. final-component root open with `O_DIRECTORY|O_NOFOLLOW`;
2. descriptor-relative witness creation with `O_NOFOLLOW` and mode `0600`;
3. successful `F_FULLFSYNC`;
4. exact same-descriptor byte readback;
5. descriptor/path device+inode equality;
6. regular-file, mode, link-count, block-accounting and `st_flags` checks;
7. no-extended-ACL disposition as `acl_get_fd_np(...) == NULL` with `errno=ENOENT`;
8. symlink witness rejection with `ELOOP`.

A future production implementation must distinguish Darwin's normal
NULL/`ENOENT` no-ACL result from unsupported/indeterminate query failures.
Any other ACL failure remains fail closed. The current Python probe hardcodes
the SDK-observed `ACL_TYPE_EXTENDED` value and does not enumerate a returned
ACL object; this is not a production contract. A later separately approved
D1.5 should use a minimal Darwin helper compiled against system headers for
ACL enumeration and closed dispositions.

This D1.1 probe does **not** prove the full v0.8 witness protocol: traversal
from `/` through every ancestry component, ancestor owner/writeability checks,
root UID/flags/ACL, normal `O_APPEND` without `O_CREAT`, the `flock(LOCK_EX)`
lock domain, repeated descriptor/path/genesis checks, directory fsync after
bootstrap metadata changes, failure injection or full-chain validation remain
for separately authorized D1.5.

## 4. Preliminary connection-ordering feasibility evidence

The v3 spike set and read back all six v0.11 connection-local fields before
transaction start using their exact mandated APIs:

```text
synchronous=FULL
foreign_keys=ON
SQLITE_DBCONFIG_TRUSTED_SCHEMA=0
recursive_triggers=OFF
SQLITE_DBCONFIG_DEFENSIVE=1
SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION=0
```

It then entered `BEGIN IMMEDIATE` and repeated readback. Inside the transaction:

- attempting to change `synchronous` raised `apsw.SQLError`;
- attempting `foreign_keys=OFF` was ineffective and readback remained enabled.

This proves the local API ordering primitive:

```text
set connection-local fields
-> exact readback
-> BEGIN IMMEDIATE
-> repeat the same six connection-local readbacks inside transaction
```

`LOW-ELEVENTH-01` remains open. This spike does not reconstruct or compare the
full evaluator profile, schema manifest, DB anchor, witness chain or full
projection before and inside the transaction. Those checks, persistent-byte
non-mutation proof, read-only-verification-first ordering and fail-closed
mismatch paths belong to D1.4, which remains separately unauthorized and
unimplemented.

## 5. Test evidence

Candidate test:

```text
tests/repo_governance/test_d1_binding_spike.py
SHA-256 69195e088771fafd8270f9b0b39d5801a01cd364f6cb071820d56b7659e2dfb3
```

Executed command:

```text
.d1-venv/bin/python -m pytest -q tests/repo_governance/test_d1_binding_spike.py --tb=short
```

Fresh result:

```text
6 passed in 0.03s
```

Covered behaviors:

1. stdlib defensive-mode insufficiency;
2. APSW exact version/source ID and all three required `db_config` readbacks;
3. distinct stdlib/APSW compile-option identities, which motivates the future
   prohibition but does not implement its runtime gate;
4. all six v0.11 connection-local settings use their exact APIs, are read
   back before `BEGIN IMMEDIATE`, and are read back again inside it;
5. final-component/descriptor-relative Darwin
   `O_NOFOLLOW`/`F_FULLFSYNC`/ACL/readback primitives;
6. `O_NOFOLLOW` symlink rejection.

A preliminary command with `-n 0` did not run tests because the disposable venv intentionally lacked `pytest-xdist`; it exited at argument parsing. The successful command above is the test evidence.

Fresh bounded regression evidence:

```text
.d1-venv/bin/python -m pytest -q \
  tests/test_hermes_state.py::TestApplyWalProbe \
  --tb=short -p no:cacheprovider
10 passed in 0.24s
```

Fresh lint evidence:

```text
.d1-venv/bin/ruff check tests/repo_governance/test_d1_binding_spike.py
All checks passed!
```

## 6. Gate result and next boundary

```text
D1_0_ISOLATED_BASE=PASS
D1_1_BINDING_FEASIBILITY=PASS_CANDIDATE
LIVE_INSTALL_CHANGED=NO
LIVE_RUNTIME_OR_CONFIG_CHANGED=NO
LIVE_DB_OR_WITNESS_CREATED=NO
REAL_REPOSITORY_ENROLLED=NO
GIT_REMOTE_PRESENT_IN_ISOLATED_WORKSPACE=NO
COMMIT_PUSH_MERGE_PERFORMED=NO
D1_2_OR_LATER_AUTHORIZED=NO
D2_TOOL_SEAM_AUTHORIZED=NO
NEXT_REQUIRED_ACTION=INDEPENDENT_D1_0_D1_1_REVIEW
```

Only an independent review with no HIGH or MEDIUM finding may close D1.0/D1.1. A PASS still does not authorize D1.2, dependency metadata changes, runtime integration, live bootstrap, D2, commit, push, merge or deployment.