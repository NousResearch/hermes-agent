"""Unit identity in the code-health ratchet: moves between files, repeated names, TypeScript.

Real git repos, the pinned ruff and the pinned TypeScript, like test_code_health.py. Each
positive case sits next to the control that must keep failing (copies never inherit debt).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.code_health.ts_measure import pinned_typescript, resolve_typescript
from tests.scripts.test_code_health import _LEGACY, _SWALLOW, REPO, _commit, _git, _repo, _verdict

_EDITED = _LEGACY.replace("return 7\n", "return 77\n")  # one constant changed, CC still 22
# Same name and CC as _LEGACY, unrelated body: a new function reusing a deleted name.
_UNRELATED = "def legacy(x):\n" + "".join(
    f"    while x > {i}:\n        x = x // 2 - {i}\n" for i in range(21)) + "    return x\n"
_MODULE_SWALLOW = "try:\n    import json\nexcept Exception:\n    pass\n"


def _base(tmp_path: Path, files: dict[str, str | None]) -> tuple[Path, str]:
    repo, _ = _repo(tmp_path)
    return repo, _commit(repo, files)


# --- M1: module-level code moved to another file keeps its existing hits -----------------

_M1_BASE: dict[str, str | None] = {"pkg/a.py": _LEGACY + "\n\n" + _SWALLOW + "\n\n" + _MODULE_SWALLOW}


@pytest.mark.parametrize("files, blocks", [
    # A0 (control): the block stays, a function is added
    ({"pkg/a.py": _M1_BASE["pkg/a.py"] + "\n\ndef added():\n    return 1\n"}, False),
    # A1: the same block moved unchanged to a new module
    ({"pkg/a.py": _LEGACY + "\n\n" + _SWALLOW, "pkg/a_compat.py": _MODULE_SWALLOW}, False),
    # moved out of a deleted file
    ({"pkg/a.py": None, "pkg/a_legacy.py": _LEGACY + "\n\n" + _SWALLOW,
      "pkg/a_compat.py": "X = 1\n\n" + _MODULE_SWALLOW}, False),
    # a copy (the origin keeps it) is new
    ({"pkg/a_compat.py": _MODULE_SWALLOW}, True),
    # one departed occurrence pays for one arrival, never two
    ({"pkg/a.py": _LEGACY + "\n\n" + _SWALLOW, "pkg/a_compat.py": _MODULE_SWALLOW + "\n" + _MODULE_SWALLOW}, True),
    # a module-level hit never pays for one inside a function elsewhere
    ({"pkg/a.py": _LEGACY + "\n\n" + _SWALLOW,
      "pkg/a_compat.py": "def load():\n" + "".join("    " + ln + "\n" for ln in _MODULE_SWALLOW.splitlines())}, True),
])
def test_module_level_hits_follow_code_moved_to_another_file(tmp_path, capsys, files, blocks):
    repo, base = _base(tmp_path, _M1_BASE)
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out


# --- M2: a function moved to another file AND edited keeps its cap ----------------------

_LEGACY_SW = _LEGACY.replace("def legacy(x):\n", "def legacy(x):\n" + _SWALLOW.split("\n", 1)[1])


@pytest.mark.parametrize("extra_base, files, blocks", [
    # P1: moved with one constant changed (CC still 22)
    ({}, {"pkg/a.py": _SWALLOW, "pkg/a_legacy.py": _EDITED}, False),
    # P2 (control): the same edit in place
    ({}, {"pkg/a.py": _EDITED + "\n\n" + _SWALLOW}, False),
    # moved, edited and grown: the cap is the old value, not a pass
    ({}, {"pkg/a.py": _SWALLOW, "pkg/a_legacy.py": _EDITED + "    if x == 99:\n        return 99\n"}, True),
    # a copy (the origin still has it) is new code
    ({}, {"pkg/a_legacy.py": _EDITED}, True),
    # an unrelated function that reuses the deleted name is new code
    ({}, {"pkg/a.py": _SWALLOW, "pkg/a_legacy.py": _UNRELATED}, True),
    # two arrivals with that name: neither is provably the moved one
    ({}, {"pkg/a.py": _SWALLOW, "pkg/b.py": _EDITED, "pkg/c.py": _EDITED}, True),
    # the moved+edited function keeps its existing swallow, but not a second one
    ({"pkg/a.py": _LEGACY_SW + "\n\n" + _SWALLOW},
     {"pkg/a.py": _SWALLOW, "pkg/a_legacy.py": _LEGACY_SW.replace("return 7\n", "return 77\n")}, False),
    ({"pkg/a.py": _LEGACY_SW + "\n\n" + _SWALLOW},
     {"pkg/a.py": _SWALLOW, "pkg/a_legacy.py": _LEGACY_SW.replace("return 7\n", "return 77\n")
      + "    try:\n        pass\n    except Exception:\n        pass\n"}, True),
])
def test_function_moved_and_edited_keeps_its_cap(tmp_path, capsys, extra_base, files, blocks):
    repo, base = _base(tmp_path, extra_base) if extra_base else _repo(tmp_path)
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out


# --- B3: repeated names pair by body, not by ordinal (banditburai) ------------------------

_DECORATOR = "def method(name):\n    return lambda f: f\n"


def _handler(route: str, branches: int, fmt: str = "    if x == {i}:\n        return {i}\n") -> str:
    return f'\n\n@method("{route}")\ndef _(x):\n' + "".join(fmt.format(i=i) for i in range(branches)) + "    return x\n"


@pytest.mark.parametrize("files, blocks", [
    # a new CC-22 `_` above the old one (trimmed to 20) is new code
    ({"pkg/h.py": _DECORATOR + _handler("new", 21, '    if x == "s{i}":\n        return "s{i}" * 2\n')
      + _handler("old", 19)}, True),
    # a small `_` above the old one, whose CC-22 body only changes a literal, keeps its cap
    ({"pkg/h.py": _DECORATOR + _handler("new", 0) + _handler("old", 21).replace("return 7\n", "return 77\n")}, False),
    # control: the old handler grows; its cap is 22 wherever it sits
    ({"pkg/h.py": _DECORATOR + _handler("new", 0) + _handler("old", 22)}, True),
])
def test_repeated_python_names_pair_by_body(tmp_path, capsys, files, blocks):
    repo, base = _base(tmp_path, {"pkg/h.py": _DECORATOR + _handler("old", 21)})
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out


# --- TypeScript identity ----------------------------------------------------------------

def _ts_base(tmp_path: Path, files: dict[str, str | None]) -> tuple[Path, str]:
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    repo, _ = _repo(tmp_path)
    # The repo's pin; the measurer resolves (or installs) it into the user cache.
    lock = {"packages": {"node_modules/typescript": {"version": pinned_typescript(REPO)}}}
    return repo, _commit(repo, {**files, "package-lock.json": json.dumps(lock)})


def _ifs(n: int, k: int = 0, indent: str = "    ") -> str:
    return "".join(f"{indent}if (x === {i + k}) return {i + k}\n" for i in range(n))


def _obj(name: str, branches: int, k: int = 0) -> str:
    return (f"export const {name} = {{\n  f(x: number): number {{\n{_ifs(branches, k, '    ')}"
            f"    return -1\n  }},\n}}\n")


@pytest.mark.parametrize("files, blocks", [
    # V9a: objects swapped; a.f CC 31 -> 20, b.f CC 3 -> 23: b.f must not inherit a.f's cap
    ({"web/o.ts": _obj("b", 22, 100) + _obj("a", 19)}, True),
    # V9b (control): the same edits in the original order
    ({"web/o.ts": _obj("a", 19) + _obj("b", 22, 100)}, True),
    # control: swapped, only a.f trimmed
    ({"web/o.ts": _obj("b", 2, 100) + _obj("a", 19)}, False),
])
def test_ts_object_literal_methods_are_named_by_their_object(tmp_path, capsys, files, blocks):
    repo, base = _ts_base(tmp_path, {"web/o.ts": _obj("a", 30) + _obj("b", 2, 100)})
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out


_SIGS = "export function old(x: number): number\nexport function old(x: string): number\n"
_NEW_SIG = "export function old(x: boolean): number\n"
_IMPL = "export function old(x: any): number {\n" + _ifs(21) + "  return -1\n}\n"
_IMPL_EDITED = _IMPL.replace("return 7\n", "return 77\n")


@pytest.mark.parametrize("files, blocks", [
    ({"web/v.ts": _SIGS + _NEW_SIG + _IMPL_EDITED}, False),  # V10a: new overload + edited impl
    ({"web/v.ts": _SIGS + _IMPL_EDITED}, False),  # V10b (control): no new overload
    ({"web/v.ts": _SIGS + _NEW_SIG + _IMPL}, False),  # V10c (control): impl untouched
    ({"web/v.ts": _SIGS + _NEW_SIG + _IMPL.replace("  return -1", "  if (x === 99) return 99\n  return -1")}, True),
])
def test_ts_overload_signatures_are_not_units(tmp_path, capsys, files, blocks):
    repo, base = _ts_base(tmp_path, {"web/v.ts": _SIGS + _IMPL})
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out


def test_ts_syntax_error_fails_closed(tmp_path, capsys):
    repo, base = _ts_base(tmp_path, {"web/ok.ts": "export const x = 1\n"})
    code, out = _verdict(repo, base, {"web/bad.ts": "export function f(x: number {\n  return x\n}\n"}, capsys)
    assert code == 1 and "MEASURE" in out and "does not parse" in out, out
    code, out = _verdict(repo, base, {"web/bad.ts": "export function f(x: number) {\n  return x\n}\n"}, capsys)
    assert code == 0, out


_TS_LEGACY = "export function legacy(x: number): number {\n" + _ifs(21, indent="  ") + "  return -1\n}\n"


@pytest.mark.parametrize("files, blocks", [
    # renamed, plus one comment line: comments are not code, so it is the same function
    ({"web/l.ts": _TS_LEGACY.replace("legacy", "walk").replace("  return -1", "  // fell through\n  return -1")}, False),
    ({"web/l.ts": _TS_LEGACY.replace("legacy", "walk")}, False),  # control: rename alone
    ({"web/l.ts": _TS_LEGACY + _TS_LEGACY.replace("legacy", "copied")}, True),  # control: a copy
])
def test_ts_identity_ignores_comments(tmp_path, capsys, files, blocks):
    repo, base = _ts_base(tmp_path, {"web/l.ts": _TS_LEGACY})
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out


_RUN_DECL = "declare function run(cb: (x: number) => number): void\n"
_SMALL_CB = "run((x: number) => {\n  return x + 1\n})\n"
_BIG_CB = "run((x: number) => {\n" + _ifs(21, indent="  ") + "  return -1\n})\n"


@pytest.mark.parametrize("files, blocks", [
    # a new callback above shifts every `<anon>#n`; the edited CC-22 callback keeps its cap
    ({"web/c.ts": _RUN_DECL + "run((x: number) => {\n  return x * 2\n})\n" + _SMALL_CB
      + _BIG_CB.replace("return 7\n", "return 77\n")}, False),
    # a new CC-22 callback above, the old one trimmed to 20: the new one is new code
    ({"web/c.ts": _RUN_DECL + "run((x: number) => {\n" + "".join(
        f"  while (x > {i}) x = x / 2 - {i}\n" for i in range(21)) + "  return x\n})\n"
      + _SMALL_CB + _BIG_CB.replace(_ifs(2, 19, "  "), "")}, True),
])
def test_ts_anonymous_callbacks_pair_by_body(tmp_path, capsys, files, blocks):
    repo, base = _ts_base(tmp_path, {"web/c.ts": _RUN_DECL + _SMALL_CB + _BIG_CB})
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out


# --- m5: the TypeScript install honours the repo's .npmrc and fails as a RuntimeError -----

def test_typescript_install_uses_repo_npmrc_and_fails_cleanly(tmp_path, monkeypatch):
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("npm is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "package-lock.json").write_text(
        json.dumps({"packages": {"node_modules/typescript": {"version": pinned_typescript(REPO)}}}),
        encoding="utf-8")
    # An unreachable registry and an empty npm cache: only the repo's .npmrc can say so.
    (repo / ".npmrc").write_text(
        f"registry=http://127.0.0.1:9/\nfetch-retries=0\ncache={tmp_path / 'npm-cache'}\nmin-release-age=14\n",
        encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="typescript"):
        resolve_typescript(repo)
    prefix = next((tmp_path / "cache" / "hermes-code-health").iterdir())
    got = subprocess.run([npm, "config", "get", "min-release-age", "--prefix", str(prefix)],
                         cwd=prefix, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    assert got == "14"
