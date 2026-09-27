# tests/ — testing rules in detail

Applies on top of the root `AGENTS.md` § Testing (runner, interpreter prep, placement, flake
policy). The detailed rule sets below moved here from the root file (Sept 2026) to keep the
root under the 32k subdirectory-hint truncation ceiling
(`agent/subdirectory_hints.py::_MAX_HINT_CHARS`): a root file over the ceiling is truncated
head+tail when an agent first touches the repo root, silently dropping the middle sections
from its context.

## Home isolation

Tests must not write to `~/.hermes/`: The autouse `_isolate_hermes_home` fixture in
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

