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
from tests.scripts.test_code_health import REPO, _commit, _git, _repo, _verdict


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
