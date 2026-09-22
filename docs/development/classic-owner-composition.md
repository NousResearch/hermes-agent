# Classic export owner composition (#104198)

This branch is the owner-only review unit for classic export production, lifecycle, and exact-byte reads. It deliberately does not duplicate Files, hosted Output, or generic runtime implementation. The branch is based on the existing #106742 runtime commit and is not expected to run alone.

## Immutable public inputs

| Layer | Public source | Tested commit |
|---|---|---|
| Runtime base | #106742, `NousResearch/hermes-agent:feat/unified-gateway-runtime` | `485d5f6848c2598ca00fa7704e50e8ae66d0983a` |
| Files | #98072 | `804ce9124f6669568585a331aeb4ef1b5cce5e29` |
| Hosted Output | #99159 | `0caf39c3c5e91eb5e0911b72fba120616d0b41e0` |
| Transaction callbacks | #111216 | `bd629002a4b8fea665404f233ddb4727082b9587` |
| Classic exports | #104198 owner checkout | the commit checked out when the recipe is run |

The script fetches the lower commits from the public NousResearch PR refs and refuses a moved ref. For #104198, run it from a checkout of the published owner commit; the PR body can pin that final SHA without creating a self-hash in this branch.

## Dependency direction and intermediate states

All four lower pins are hard dependencies of the **complete #104198 classic export journey** exercised below: #106742 supplies the review base, #98072 supplies Files custody and sharing, #99159 supplies hosted Output, and #111216 supplies the optional transaction-callback contracts consumed by classic settlement. They are not bundled into the #104198 review interval.

The dependency direction is one-way. Each lower layer remains usable before #104198 lands: #98072 and #99159 have no classic consumer requirement, and #111216 keeps both callbacks optional when no owner passes one. The classic-specific marker, read/discard routes, scope handling, and callback arguments exist only in this owner. There is no compatibility shim or undocumented glue step.

#99159 and #98072 have overlapping history. The tested composition starts from the exact #99159 tree, which already carries the integrated Files contracts, and records the exact #98072 head with an `ours` merge that must leave the tree unchanged. It does **not** replay the Files delta. It then cherry-picks the three commits ending at #111216 exactly once. Their authors, author dates, commit messages, and source trailers are retained.

Eleven of the fourteen reviewed owner path changes apply directly to #106742 and are ordinary source/test changes in this branch. Exactly three owner-owned modifications cannot be represented directly on that base: `gateway/hosted_room_artifacts.py` and `tools/hosted_room_artifact.py` do not exist there, while the insertion context in `gateway/session_finite.py` is supplied by hosted Output. Those three focused hunks are retained in `docs/development/patches/classic-owner-dependent-integration.patch`. The patch contains only #104198 hunks, not lower-layer whole-file postimages.

## Compose

From the #104198 owner checkout:

```sh
scripts/compose_classic_owner.sh /absolute/path/to/classic-integration HEAD
```

The destination must not exist. The script creates a disposable Git checkout, verifies all four public pins, composes each lower layer once, cherry-picks only the owner commits above `485d5f6848c2598ca00fa7704e50e8ae66d0983a`, applies the focused dependent patch, and prints the resulting commit and tree.

## Portable Python setup

Hermes declares its build backend and dependencies in `pyproject.toml` and locks them in `uv.lock`. A maintainer can create an external runner rather than placing a virtual environment in the checkout:

```sh
RUNNER="$HOME/.hermes/venvs/hermes-classic-104198"
uv venv "$RUNNER" --python 3.11
uv pip install --python "$RUNNER/bin/python" -e '.[all,dev]'
```

Run the causal test from the composed checkout with isolated state:

```sh
mkdir -p .test-env/home .test-env/hermes .test-env/tmp .test-env/pycache

env \
  HOME="$PWD/.test-env/home" \
  HERMES_HOME="$PWD/.test-env/hermes" \
  TMP="$PWD/.test-env/tmp" TEMP="$PWD/.test-env/tmp" \
  PYTHONPYCACHEPREFIX="$PWD/.test-env/pycache" \
  HERMES_WRITE_SAFE_ROOT=/opt/data \
  HERMES_PYTHON="$RUNNER/bin/python" \
  HERMES_TEST_WORKERS=2 HERMES_TEST_FILE_RETRIES=0 \
  taskset -c 0,1 scripts/run_tests.sh \
  tests/gateway/test_classic_current_export.py \
  -k test_registered_classic_admission_shares_and_reads_exact_bytes -q
```

A backend-only syntax/import/build check can use the same runner; no Desktop build is part of this owner:

```sh
PYTHONPYCACHEPREFIX="$PWD/.test-env/pycache" "$RUNNER/bin/python" -m compileall -q \
  gateway/classic_output_exports.py \
  gateway/hosted_room_artifacts_classic.py \
  gateway/session_classic_exports.py \
  gateway/session_classic_output.py

HOME="$PWD/.test-env/home" HERMES_HOME="$PWD/.test-env/hermes" \
PYTHONPYCACHEPREFIX="$PWD/.test-env/pycache" "$RUNNER/bin/python" -c \
  'import gateway.classic_output_exports, gateway.hosted_room_artifacts_classic, gateway.session_classic_exports, gateway.session_classic_output'

BUILD_RUNNER="${TMPDIR:-/tmp}/hermes-classic-104198-editable"
uv venv "$BUILD_RUNNER" --python 3.11
uv pip install --python "$BUILD_RUNNER/bin/python" --no-deps -e .
```

The editable install is the supported setuptools backend path declared by this repository. `setup.py` intentionally rejects wheel and sdist publication outside the separately configured Nix build, so `uv build --wheel` is not a valid backend gate here.

The focused test is the proof for this final composition. It performs a fresh canonical submission, real registered `share_group_file` execution, terminal settlement, exact retained-byte read, generation and principal refusal, replay refusal, and retirement refusal without provider or native/network application calls. Native/Windows CI remains separate.
