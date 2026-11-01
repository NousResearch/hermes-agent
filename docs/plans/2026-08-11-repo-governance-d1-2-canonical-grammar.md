# Repository Governance D1.2 — Canonical Byte Grammar and Golden Vectors

- **Date:** 2026-08-11
- **Candidate revision:** v3
- **Status:** candidate awaiting independent exact-byte review
- **Scope:** D1.2 pure canonical byte functions and executable vectors only
- **D1.3 or later:** not authorized
- **Live/runtime/config/DB/witness integration:** not authorized
- **Commit/push/merge/GitHub/deployment:** not authorized

## 1. Owner ATP and exact base

The owner explicitly selected the D1.2-only implementation option after the
D1.0/D1.1 feasibility gate closed. The authorization is bound to:

```text
workspace  /Users/ykliu/Projects/hermes-agent-repo-governance-d1
HEAD       af250d84948179834820a62bfd870c0df6f264a1
tree       6865279a06e3bbab8dbf91b330dc2455e402b204
mode       detached HEAD
remotes    none
```

Authorized writes are limited to D1.2 pure source, tests, vectors, this decision
record and external audit artifacts. Existing D1.1 candidate files remain
unaltered historical prerequisites.

## 2. Normative requirement mapping

| Requirement | Governing source | D1.2 evidence |
|---|---|---|
| RFC 8785 canonical UTF-8 and named preimages | v0.2 §11 | `canonical_json_bytes`, strict byte-round-trip parser and domain-separated preimage/digest builders |
| canonical unsigned/signed decimal strings and ranges | v0.9 §6.2 | `require_unsigned_decimal`, `require_signed64_decimal`, boundary and rejection vectors |
| NFC UTF-8 field grammar without silent normalization | v0.9 §6.1/§7.2 and v0.10 exact-source exceptions | explicit `require_nfc_text`; serializer preserves exact scalar bytes and rejects surrogates |
| complete witness `/0.9` outer/payload/install identity grammar | v0.9 §6–§7 | exact field sets, literals, decimal/digest/base64url/UUIDv7 formats and state-specific nullability |
| witness framing | v0.9 §7.1 | exact canonical JSON plus one final `0x0A`; malformed/multiple/missing boundary fails closed |
| witness digest/signature preimages | v0.9 §7.3 | exact `/0.9` digest domain and raw-32-byte signature preimage; no live key |
| deterministic golden bytes | implementation-readiness D1.2 and v0.9 §8 | independent producer plus genesis/prepared/committed/aborted exact payload/frame/preimage vectors |
| duplicate/missing/unknown/noncanonical input | v0.2 §11, v0.9 §7.3 | strict parser and closed-object rejection under one `CanonicalEncodingError` class |
| version-prefix changes | implementation-readiness D1.2 | fixed `/0.2` versus `/0.3` digest vectors and witness schema-version negative |

Normative precedence remains `v0.11 > v0.10 > ... > v0.1`.

## 3. Closed canonical value profile

`canonical_json_bytes` intentionally implements the RFC 8785 subset used by
these closed governance artifacts:

```text
null
boolean
Unicode scalar string
array of closed values
string-keyed object of closed values
```

Raw JSON number tokens and Python integer/float values are rejected. Normative
governance integers are canonical decimal **strings** under v0.9 §6.2, avoiding
binary64 ambiguity. This is a narrower, truthful profile; it does not claim to
be a general arbitrary-number RFC 8785 package.

Object properties sort by UTF-16 code units as RFC 8785 requires. Strings use
minimal JSON escaping and exact UTF-8. Generic serialization never silently
normalizes Unicode. Field-specific NFC requirements are explicit validators.
Invalid UTF-8, surrogate code points, duplicate keys, whitespace, alternate key
order, BOM, trailing bytes, raw numbers and recursive in-memory values all fail
closed through `CanonicalEncodingError`.

## 4. Witness `/0.9` pure grammar

The pure witness helpers validate and encode:

- the exact three-field outer record;
- the exact nineteen-field payload;
- the exact eleven-field genesis installation identity;
- canonical unsigned decimal ranges and signed-64 helper grammar;
- lowercase 64-hex digests;
- lowercase canonical UUIDv7 transaction IDs;
- canonical unpadded base64url encoding of exactly 64 signature bytes;
- genesis/prepared/terminal nullability and literal rules;
- genesis installation-identity digest recomputation;
- record-digest recomputation;
- one canonical frame terminated by exactly one LF.

The signature preimage builder uses raw 32-byte record digest bytes, not hex
text. It does not sign or verify with a live key.

## 5. Golden vector producer

The external producer is deliberately independent from the candidate module:

```text
/Users/ykliu/.hermes/profiles/dev/artifacts/repo-governance/scripts/
  generate_d1_2_golden.py
```

It uses only Python stdlib JSON/SHA-256 and imports no `repo_governance` code.
All vector keys are ASCII, so its `sort_keys=True` order equals RFC 8785 UTF-16
order for this fixture. A fresh producer replay was byte-equal to:

```text
tests/repo_governance/vectors/d1_canonical_golden.json
SHA-256 465094439dda61dd91d1f4b28becf0906671b7623b405584abedb644a0365dc3
```

The vector contains exact canonical payload hex, record digest, signature
preimage hex, outer record and frame hex for:

1. genesis;
2. prepared;
3. committed terminal;
4. aborted terminal.

The vector signature is a syntactically canonical 64-zero-byte placeholder. It
is not presented as a valid cryptographic signature or authority.

## 6. TDD evidence

Observed RED→GREEN steps included:

1. missing `repo_governance` module;
2. missing signed/unsigned decimal validators;
3. missing NFC validator;
4. missing strict parser/closed-object validator;
5. missing domain/preimage builders;
6. missing witness encoder/parser;
7. root `/` incorrectly rejected despite the normative exception;
8. unhashable `recordType` leaked `TypeError` instead of the single fail-closed error;
9. recursive container leaked `RecursionError` instead of the single fail-closed error.
10. deeply nested canonical input leaked `RecursionError` from `json.loads`;
11. 5000-digit canonical-shaped decimal strings leaked Python's integer-conversion `ValueError`.
12. caller-supplied unsigned `maximum` could widen the normative uint64 range
    and an extreme bound could leak the same integer-conversion `ValueError`.

Each failing behavior was corrected before advancing.

Fresh private-`TMPDIR`, bytecode-disabled results:

```text
D1.2 canonical grammar       63 passed in 0.35s
D1.1 prerequisite regression 6 passed in 0.06s
bounded WAL regression       10 passed in 0.32s
Ruff --no-cache              All checks passed
vector producer replay       byte-equal
```

## 7. Independent review findings and candidate closures

The v1 independent exact-byte review returned `HIGH=0 / MEDIUM=2 / LOW=0 /
REVISE`. v1 remains an immutable historical candidate and does not close D1.2.

### MEDIUM-01 — hostile input exception leakage

Root causes:

1. `json.loads` can raise `RecursionError` before the parsed object reaches the
   serializer's existing recursion normalization;
2. Python 3.11 limits decimal-to-integer conversion to 4300 digits by default,
   so converting before checking the normative 64-bit range leaked `ValueError`.

v2 fixes:

- `parse_canonical_json_bytes` translates bounded parser `RecursionError` to
  `CanonicalEncodingError`;
- decimal validators compare canonical digit strings against the small bound
  lexicographically before safe integer conversion;
- four hostile-input regressions cover 2000-level JSON/frame nesting and
  5000-digit signed/unsigned inputs.

### MEDIUM-02 — ambiguous residue claim

The isolated workspace contains `.d1-venv`, the pre-existing D1.1 disposable
APSW/pytest/Ruff toolchain documented by the closed prerequisite. Its installed
packages necessarily contain `__pycache__` and `.pyc` files. The v1 phrase
`bytecodeResidueUnderWorkspace: 0` failed to exclude that known prerequisite
and was therefore false as written.

v2 does not delete or relabel the D1.1 toolchain. It explicitly excludes only
`.d1-venv` from the D1.2 residue claim, removes workspace test/lint caches
outside that venv, runs pytest with `PYTHONDONTWRITEBYTECODE=1` and
`-p no:cacheprovider`, runs Ruff with `--no-cache`, and verifies:

```text
ignored_test_lint_residue_outside_d1_venv=0
```

This is a scoped, reproducible claim rather than a whole-workspace zero claim.

### MEDIUM-NEW-01 — widening caller-supplied unsigned bound

The v2 independent exact-byte review returned `HIGH=0 / MEDIUM=1 / LOW=0 /
REVISE`. It confirmed both v1 findings closed, but found that the public
`maximum` parameter could widen the normative uint64 range. An extreme Python
integer bound could also raise `ValueError` during `str(maximum)`.

v3 makes the optional bound narrowing-only. Before any string conversion,
`maximum` must be an exact Python `int` satisfying:

```text
0 <= maximum <= UINT64_MAX
```

Regressions verify a legitimate narrowed bound, rejection above that bound,
rejection of `UINT64_MAX + 1`, rejection of `2**100`, and rejection of a
5000-digit integer bound through `CanonicalEncodingError` only.

## 8. Deliberately excluded behavior

D1.2 does not implement or authorize:

1. cryptographic signature creation or verification;
2. live keys, key loading, rotation, compromise handling or trust decisions;
3. witness file creation, append, fsync, descriptor/path validation or locking;
4. cross-record chain traversal, prepared-to-terminal copy equality, unresolved-transaction policy, reconciliation or recovery;
5. SQLite schema/evaluator/anchor/full-state behavior;
6. repository identity resolution;
7. mutation decisions, leases, intents, receipts, tool dispatch or side effects;
8. a general-purpose RFC 8785 binary64 number implementation;
9. D1.3+, live installation, runtime/config/plugin changes, real repository enrollment, Git/GitHub mutation or deployment.

Items 1–4 remain D1.5 or later. SQLite evaluator work remains D1.4. Repository
identity remains D1.3. No later slice starts from this candidate without a
separate owner ATP.

## 9. Candidate gate

```text
D1_2_PURE_GRAMMAR=PASS_CANDIDATE
D1_2_INDEPENDENT_REVIEW=PENDING
D1_2_GATE_CLOSED=NO
D1_3_OR_LATER_AUTHORIZED=NO
LIVE_INSTALL_CHANGED=NO
LIVE_RUNTIME_OR_CONFIG_CHANGED=NO
LIVE_DB_OR_WITNESS_CREATED=NO
REAL_REPOSITORY_ENROLLED=NO
COMMIT_PUSH_MERGE_PERFORMED=NO
D2_TOOL_SEAM_AUTHORIZED=NO
NEXT_REQUIRED_ACTION=INDEPENDENT_EXACT_BYTE_REVIEW
```
