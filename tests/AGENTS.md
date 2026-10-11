# tests/ — how Hermes tests are written and run

Applies on top of the root `AGENTS.md`. Read it before adding or changing any test.

## Running tests

**ALWAYS use `scripts/run_tests.sh`**, never bare `pytest`. It enforces CI parity: credential
vars unset, `TZ=UTC`, `LANG=C.UTF-8`, `HERMES_HOME` → temp dir, and per-file subprocess
isolation via `scripts/run_tests_parallel.py` (no xdist; workers scale with CPU count) so
module-level dicts/ContextVars cannot leak between files. Direct `pytest` on a big machine
with API keys set has caused repeated "works locally, fails in CI" incidents (and the reverse).

Prepare a test interpreter with the checkout's bootstrapped Python:

```bash
python -m pm.build_env --source . --out .venv --group dev --group test
```

This is a fresh build, not an in-place sync. If the disposable output exists,
stop its processes and intentionally remove it before regeneration. The runner
clears `PYTHONPATH`, so PM shell activation alone does not supply pytest. For a
fresh output outside the checkout, set `HERMES_PYTHON` to its interpreter.

```bash
scripts/run_tests.sh                                    # full suite
scripts/run_tests.sh tests/gateway/                     # one directory
scripts/run_tests.sh tests/agent/test_foo.py -k test_x  # runner is file-granular; -k narrows
scripts/run_tests.sh -v --tb=long                       # pytest flags pass through
```

- **Flake policy:** a failing FILE is retried once in a fresh subprocess (`--file-retries`;
  `HERMES_TEST_FILE_RETRIES=0` disables); a worker killed by signal or the file timeout is never
  retried (relaunching a runaway doubles the damage). Pass-on-retry is green but printed under `⚠ FLAKY`
  with both outputs — a bug to fix, not noise. Timing tests must not assume a quiet runner:
  wall-clock bounds ≥ 2s, event-based sync, no `assert not _wait_until(...)` races.
- **Placement mirrors the source tree.** A test lives in `tests/<top-level source dir>/` (`tests/hermes_cli/`,
  `tests/agent/`, `tests/hermes_state/`, `tests/gateway/relay/`, ...); installer/updater script tests
  under `tests/scripts/{install,desktop_update}/`. Only tests of root-level modules (`batch_runner`,
  `utils`, `hermes_constants`, packaging) sit directly in `tests/`. No issue numbers in filenames —
  cite the issue in the module docstring (`test_89315_x.py` → `test_x.py`, "Regression for #89315").
- **Placement (CI lanes):** `scripts/ci/classify_changes.py` picks jobs by changed files. A Python test
  asserting about `package.json`, `package-lock.json`, `tsconfig.json`, or `.ts/.tsx/.js/
  .mjs/.cjs` sources will not run on a JS-only PR (green on PR, red on `main` where the
  classifier fails open). Such tests belong in the vitest suite, not `tests/*.py`.
- **Tests must not write to `~/.hermes/`.** The autouse `_isolate_hermes_home` fixture in
  `tests/conftest.py` redirects `HERMES_HOME`; never hardcode `~/.hermes/` in tests. Profile
  tests also mock `Path.home()` so `_get_profiles_root()` / `_get_default_hermes_home()` stay
  in the temp dir (pattern: `tests/hermes_cli/test_profiles.py`):
  ```python
  @pytest.fixture
  def profile_env(tmp_path, monkeypatch):
      home = tmp_path / ".hermes"; home.mkdir()
      monkeypatch.setattr(Path, "home", lambda: tmp_path)
      monkeypatch.setenv("HERMES_HOME", str(home))
      return home
  ```
  Tests that `patch.object(Path, "home", ...)` must ALSO set `HERMES_HOME` — code reads the
  env var, not `Path.home()/.hermes`.

## Don't fake the host OS

Behaviour that genuinely differs per host is tested ON that host with `@pytest.mark.platforms("linux")`
/ `platforms("macos")` / `platforms("windows")`, never by patching `sys.platform`. Host-independent things stay
unmarked: pure functions that take the platform as data (`hidden_windows_child_options(opts,
is_windows=True)`) and declaration/packaging invariants ("pyproject declares `tzdata` with a
`sys_platform == 'win32'` marker"). Setting a module-level `IS_WINDOWS` flag and calling
`windows_detach_flags()` IS a fake. The line: **if the test needs the interpreter to believe it
is on another OS to pass, it belongs on that OS.** A test that walks several platforms in
sequence is split — host-native arm on Linux, other arms as their own marked tests.

One marker per test, with any number of spec strings (any-of semantics) plus
optional arch filters. To gate on several OSes, pass several specs to ONE
marker — never stack several `platforms()` decorators on one test (the
conftest rejects that at collection):

```python
@pytest.mark.platforms("linux", "macos")  # ONE marker, two specs: runs on either
def test_posix_signal_path(): ...
```

Other single-marker forms (each is a complete marker on its own):
`platforms("windows")` (native Windows only), `platforms("not macos")`
(anywhere except macOS), `platforms("windows", arch="arm64")` (native Windows
on arm64), `platforms("posix")` (Linux or macOS).

Specs: `linux`, `macos`, `windows`, `posix`, `any`, and `not <spec>`.
The historic `linux_only` / `macos_only` / `windows_only` markers have been
fully replaced — `platforms` is the only host-gating marker in the tree.

**Live Windows process-topology E2E: the `wine2e` lane.** For claims about
real Windows process behavior that mocks cannot reproduce (venv-holder
scans, process-tree parentage, launcher/worker chains, detach semantics),
there is an on-demand workflow `windows-venv-e2e.yml` that runs
`tests/hermes_cli/test_venv_holder_windows_live.py` on a real
`windows-latest` runner — spawning actual processes and driving the real
detection code, no mocked psutil. It fires ONLY on pushes to `wine2e/**`
branches (inert on PRs and main; costs nothing on normal work). The proven
workflow: write probes that pin CORRECT behavior, push to a `wine2e/`
branch to reproduce the bugs live on unfixed code, build the fix, iterate
until the lane is green, then open the PR — the live receipt on the exact
head is the Windows proof reviewers ask for. Extend the live suite when
touching that subsystem; assert against the gateway ANCESTOR found by
argv, not the direct parent (the venv shim makes every spawn a
launcher/worker chain).

**Use the marker, never a bare `skipif`.** `scripts/ci/list_os_marked_tests.py`
decides which files an OS lane imports by resolving the quoted specs inside
`platforms(...)` (`"posix"` reaches the macOS lane, `"not linux"` reaches
both others), and the lane then selects with `-m platforms` while the
conftest's per-test host skips do the actual gating. A test gated with
`@pytest.mark.skipif(sys.platform != "win32")` therefore runs on no host at
all, silently — it is never imported by the lane that would run it, and the
full-suite lanes skip it. `skipif(sys.platform == "win32")` becomes
`platforms("posix")`; a non-host condition (`os.geteuid() == 0`) stays a
separate `skipif` beside the marker. A misspelt spec is a collection error,
not a skip. Don't stack a module-level `pytestmark =
platforms(...)` on a file whose tests carry their own host marker — the
conftest hard-rejects tests carrying two `platforms()` markers (a test
skipped on every host, reported green everywhere).
Equally, don't `pytest.skip()` the non-host rows of a `@parametrize` over
platforms — split it into one marked test per OS, or only the host's row ever
executes.

## Don't write change-detector tests

A change-detector fails whenever data *expected to change* is updated — model catalogs,
`_config_version`, enumeration counts, hardcoded model lists. It adds no coverage and taxes
every routine update. Don't: `assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]`,
`assert DEFAULT_CONFIG["_config_version"] == 21`, `assert len(models) == 8`. Do: `assert
"gemini" in _PROVIDER_MODELS and len(_PROVIDER_MODELS["gemini"]) >= 1` (plumbing works);
`assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]` (migration reaches
latest); `assert not (set(moonshot_models) & coding_plan_only_models)` (no leak); every
catalog model has a context-length entry (relationship). If it reads like a snapshot, delete
it; if it reads like a contract between two pieces of data, keep it. Reviewers reject new
change-detectors; authors convert them before re-review.

## Never read source code in tests

A test that reads a `.py`/`.ts`/`.tsx` file's text tests the *shape of the source*, not
behavior — banned outright. It passes when the implementation is subtly broken (regex matches
a mis-wired call site) and fails on correct refactors; it can't run against bundled/minified
artifacts; it blocks structural cleanup; it gives false confidence. Don't
`fs.readFileSync('main.ts')` + `assert.match(source, /spawn\(...hiddenWindowsChildOptions/)`.
Do extract the logic into a pure/DI-testable function and call it:
```ts
export function hiddenWindowsChildOptions(options = {}, isWindows = process.platform === 'win32') {
  if (!isWindows || 'windowsHide' in options) return options
  return { ...options, windowsHide: true }
}
```
If the logic lives inline in a god-file and extraction feels disruptive, that is the signal to
extract, not to regex around it.
