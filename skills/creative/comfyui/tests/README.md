# ComfyUI Skill Tests

Pytest suite covering the skill's scripts. Pure-stdlib unit tests run
without any setup; cloud integration tests need a Comfy Cloud API key.

## Running

```bash
# Unit tests only (no network required) — runs in <1s
python3 -m pytest tests/ -c tests/pytest.ini -o addopts="-p no:xdist"

# Including cloud integration tests
COMFY_CLOUD_API_KEY="comfyui-..." python3 -m pytest tests/ \
  -c tests/pytest.ini -o addopts="-p no:xdist"

# Just cloud tests
COMFY_CLOUD_API_KEY="comfyui-..." python3 -m pytest tests/test_cloud_integration.py \
  -c tests/pytest.ini -o addopts="-p no:xdist" -v
```

The `-c` and `-o` overrides isolate this suite from any parent
`pyproject.toml` pytest config (e.g. the `-n auto` from a parent repo).

## Test files

| File | Coverage |
|------|----------|
| `test_common.py` | Cloud detection, URL routing, format validation, embeddings, paths, seeds, model-list parsing, folder aliases |
| `test_extract_schema.py` | Connection tracing, positive/negative prompt detection, dedup logic, embedding deps |
| `test_run_workflow.py` | Param injection (incl. -1 seed, link refusal), output download walk, runner construction |
| `test_check_deps.py` | Model-name fuzzy matching, install command suggestions |
| `test_cloud_integration.py` | Live cloud API contract tests (auto-skipped without API key) |

## Adding tests

When you change a script:

1. Add a unit test if the change is pure logic (cloud detection, parsing, etc.)
2. Add a cloud integration test if the change depends on cloud API behavior
   (use `pytestmark = pytest.mark.cloud` so it auto-skips without a key)
3. Workflow fixtures live in `conftest.py` (`sd15_workflow`, `flux_workflow`,
   `video_workflow`)

## Why the explicit `-c` / `-o`?

The parent hermes-agent repo's `pyproject.toml` enables `pytest-xdist` by
default (`-n auto`). This suite is small enough that parallelism isn't
worth the complexity, and pytest-xdist isn't always installed in the user's
environment. The `-c tests/pytest.ini -o addopts="-p no:xdist"` flags make
the suite run identically regardless of the parent project's config.
