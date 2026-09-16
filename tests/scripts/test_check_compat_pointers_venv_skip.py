"""scripts/check_compat_pointers.py must not walk or scan virtualenvs (regression for #112584).

With an environment inside the checkout the scanner treated the environment's packages as
first-party source: it read and AST-parsed them, and an installed package whose source mentions a
compat-pointer name was reported as an in-tree violation (the reporter's build failed on
``.venv/lib/python3.11/site-packages/pip/__init__.py``). These tests build environments on disk and
assert on what the scanner enters, reads, and reports — no source-shape assertions.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_compat_pointers.py"
FACADE = "acp_adapter.auth"  # a real facade/name pair from compat_manifest.json
POINTER = "has_provider"
VIOLATION = f"from {FACADE} import {POINTER}\n"
CLEAN = "import json\n\n\ndef helper():\n    return json.dumps({})\n"


def _load(root: Path):
    """Load the scanner by path and point it at ``root`` instead of the repo."""
    spec = importlib.util.spec_from_file_location("check_compat_pointers_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ROOT = root
    mod.MANIFEST = root / "compat_manifest.json"
    mod.MANIFEST.write_text(
        json.dumps({"entries": [{"facade": FACADE, "name": POINTER, "kind": "import", "target": "x"}]}),
        encoding="utf-8",
    )
    return mod


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _scan(mod, monkeypatch):
    """Run the real traversal + scan, recording every directory entered and file read."""
    entered: list[Path] = []
    read: list[Path] = []
    real_scandir, real_read_text = os.scandir, Path.read_text

    def spy_scandir(path="."):
        entered.append(Path(path))
        return real_scandir(path)

    def spy_read_text(self, *args, **kwargs):
        read.append(Path(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", spy_scandir)
    monkeypatch.setattr(Path, "read_text", spy_read_text)
    yielded = [Path(p) for p in mod._py_files()]
    rc = mod.main()
    return yielded, entered, read, rc


def _rel(paths, root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in paths}


def _venv(root: Path, name: str, *, package: str = "pip", marker: bool = True) -> Path:
    """An on-disk environment whose site-packages holds a pointer-using package."""
    env = root / name
    if marker:
        _write(env / "pyvenv.cfg", "home = /usr/bin\n")
        _write(env / "bin" / "activate", "")
    store = env / "lib" / "python3.11" / "site-packages"
    _write(store / package / "__init__.py", VIOLATION)
    _write(store / package / "_vendor" / "mod.py", CLEAN)
    return env


def test_virtualenv_is_pruned_never_read_and_never_reported(tmp_path, monkeypatch, capsys):
    env = _venv(tmp_path, ".venv")
    _write(tmp_path / "hermes_cli" / "pointer_user.py", VIOLATION)
    _write(tmp_path / "hermes_cli" / "clean.py", CLEAN)
    mod = _load(tmp_path)

    yielded, entered, read, rc = _scan(mod, monkeypatch)

    # The environment's whole subtree is never entered, so its files are never read or parsed.
    assert not [d for d in entered if d == env or env in d.parents]
    assert not [r for r in read if env in r.parents]
    assert not [p for p in yielded if env in p.parents]

    # ...while first-party files are still walked, scanned and reported.
    assert _rel(yielded, tmp_path) == {"hermes_cli/clean.py", "hermes_cli/pointer_user.py"}
    assert rc == 1
    out = capsys.readouterr().out
    assert f"hermes_cli/pointer_user.py:1: from {FACADE} import {POINTER}" in out
    assert ".venv" not in out
    assert "1 site(s)" in out


def test_environment_named_env_is_skipped(tmp_path, monkeypatch, capsys):
    env = _venv(tmp_path, "env")
    _write(tmp_path / "app.py", CLEAN)
    mod = _load(tmp_path)

    yielded, entered, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"app.py"}
    assert not [d for d in entered if d == env or env in d.parents]
    assert rc == 0
    assert "env/" not in capsys.readouterr().out


def test_first_party_package_named_env_is_still_scanned(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "env" / "settings.py", VIOLATION)
    _write(tmp_path / "env" / "__init__.py", CLEAN)
    mod = _load(tmp_path)

    yielded, _, _, rc = _scan(mod, monkeypatch)

    # No venv markers, so `env/` is ordinary source and its violation must still fail the build.
    assert _rel(yielded, tmp_path) == {"env/__init__.py", "env/settings.py"}
    assert rc == 1
    assert f"env/settings.py:1: from {FACADE} import {POINTER}" in capsys.readouterr().out


def test_conda_environment_and_bare_package_store_are_skipped(tmp_path, monkeypatch, capsys):
    conda_env = tmp_path / "miniconda" / "envs" / "py311"
    _write(conda_env / "conda-meta" / "history", "")
    _write(conda_env / "lib" / "python3.11" / "site-packages" / "numpy" / "__init__.py", VIOLATION)
    _write(tmp_path / "vendor" / "lib" / "python3.12" / "site-packages" / "pkg" / "mod.py", VIOLATION)
    _write(tmp_path / "dist-packages" / "legacy" / "mod.py", VIOLATION)
    _write(tmp_path / "app.py", CLEAN)
    mod = _load(tmp_path)

    yielded, entered, read, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"app.py"}
    assert not [d for d in entered if conda_env in d.parents or d == conda_env]
    assert not [r for r in read if r.suffix == ".py" and ("packages" in r.parts or "conda-meta" in r.parts)]
    assert rc == 0
    assert "✅" in capsys.readouterr().out


def test_environment_with_a_nonstandard_name_is_skipped_via_the_pyvenv_marker(tmp_path, monkeypatch):
    env = _venv(tmp_path, "my-work-env")
    _write(tmp_path / "tools" / "thing.py", CLEAN)
    mod = _load(tmp_path)

    yielded, entered, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"tools/thing.py"}
    assert not [d for d in entered if d == env or env in d.parents]
    assert rc == 0


def test_virtualenv_nested_below_a_first_party_package_is_skipped(tmp_path, monkeypatch):
    env = _venv(tmp_path, "packages/desktop/.venv")
    _write(tmp_path / "packages" / "desktop" / "main.py", CLEAN)
    mod = _load(tmp_path)

    yielded, entered, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"packages/desktop/main.py"}
    assert not [d for d in entered if d == env or env in d.parents]
    assert rc == 0


def test_environment_without_markers_is_still_skipped_by_name(tmp_path, monkeypatch):
    env = _venv(tmp_path, "venv", marker=False)
    _write(tmp_path / "src" / "app.py", CLEAN)
    mod = _load(tmp_path)

    yielded, entered, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"src/app.py"}
    assert not [d for d in entered if d == env or env in d.parents]
    assert rc == 0


def test_clean_project_scans_every_first_party_file(tmp_path, monkeypatch):
    files = {
        "app.py",
        "src/app.py",
        "src/nested/deep/module.py",
        "tests/test_app.py",
        "scripts/generate.py",
        "scripts/test_contract.py",
    }
    for rel in files:
        _write(tmp_path / rel, CLEAN)
    mod = _load(tmp_path)

    yielded, _, read, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == files
    assert {r.name for r in read if r.suffix == ".py"} >= {Path(f).name for f in files}
    assert rc == 0


def test_scanner_own_files_stay_excluded(tmp_path, monkeypatch):
    _write(tmp_path / "scripts" / "check_compat_pointers.py", VIOLATION)
    _write(tmp_path / "tests" / "test_compat_manifest_targets.py", VIOLATION)
    _write(tmp_path / "app.py", CLEAN)
    mod = _load(tmp_path)

    yielded, _, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"app.py"}
    assert rc == 0


def test_symlinked_tree_is_not_followed(tmp_path, monkeypatch):
    """A symlink out of the repo must not pull foreign source into the scan (no followlinks)."""
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    _write(outside / "contains.py", VIOLATION)
    _write(tmp_path / "app.py", CLEAN)
    (tmp_path / "src-linked").symlink_to(outside, target_is_directory=True)
    mod = _load(tmp_path)

    yielded, entered, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"app.py"}
    assert not [d for d in entered if d == outside or outside in d.parents]
    assert rc == 0


def test_nested_exclusions_are_pruned_at_any_depth(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "packages" / "web" / "node_modules" / "dep" / "index.py", VIOLATION)
    _write(tmp_path / "packages" / "web" / ".git" / "hook.py", VIOLATION)
    _write(tmp_path / "packages" / "web" / "main.py", CLEAN)
    mod = _load(tmp_path)

    yielded, _, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {"packages/web/main.py"}
    assert rc == 0


def test_nested_dirs_sharing_a_root_excluded_name_are_still_scanned(tmp_path, monkeypatch, capsys):
    """`skills/` at the repo root is out of scope; `tests/skills/` is test source and must be scanned."""
    _write(tmp_path / "tests" / "skills" / "test_skills.py", VIOLATION)
    _write(tmp_path / "tests" / "evals" / "test_evals.py", VIOLATION)
    _write(tmp_path / "website" / "build_docs.py", VIOLATION)
    _write(tmp_path / "app.py", CLEAN)
    mod = _load(tmp_path)

    yielded, _, _, rc = _scan(mod, monkeypatch)

    assert _rel(yielded, tmp_path) == {
        "app.py",
        "tests/skills/test_skills.py",
        "tests/evals/test_evals.py",
    }
    assert rc == 1
    out = capsys.readouterr().out
    assert "tests/skills/test_skills.py:1" in out
    assert "website/build_docs.py" not in out


def test_prune_reason_classifies_environments_and_leaves_source_alone(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _write(tmp_path / "src" / "pkg" / "mod.py", CLEAN)
    _write(tmp_path / "node_modules" / "x.py", CLEAN)
    _write(tmp_path / ".git" / "hooks" / "x.py", CLEAN)
    _venv(tmp_path, ".venv")
    _write(tmp_path / "env_no_marker" / "mod.py", CLEAN)
    _write(tmp_path / "dockerfile_env" / "mod.py", CLEAN)
    _write(tmp_path / "site-packages" / "pkg.py", CLEAN)
    _write(tmp_path / "dist-packages" / "pkg.py", CLEAN)
    # An ambiguous `env` is only an environment when it carries evidence. POSIX layout...
    _write(tmp_path / "posix" / "env" / "bin" / "activate", "")
    _write(tmp_path / "posix" / "env" / "lib" / "python3.11" / "site-packages" / "pkg.py", CLEAN)
    # ...and Windows layout, where there is no `bin/activate` to find but there is a package store.
    _write(tmp_path / "windows" / "env" / "Scripts" / "Lib" / "site-packages" / "pkg.py", CLEAN)
    conda = tmp_path / "condaenv"
    _write(conda / "conda-meta" / "history", "")
    _write(conda / "lib" / "python3.11" / "site-packages" / "pkg.py", CLEAN)

    assert mod._prune_reason(tmp_path / ".venv") == "pyvenv.cfg"
    assert mod._prune_reason(tmp_path / "posix" / "env") == "env/ with an activate script"
    assert mod._prune_reason(tmp_path / "windows" / "env") == "env/ with a package store"
    assert mod._prune_reason(tmp_path / "site-packages") == "site-packages"
    assert mod._prune_reason(tmp_path / "dist-packages") == "dist-packages"
    assert mod._prune_reason(conda) == "conda-meta"
    assert mod._prune_reason(tmp_path / "node_modules") == "excluded dir node_modules"
    assert mod._prune_reason(tmp_path / ".git") == "excluded dir .git"
    assert mod._prune_reason(tmp_path / "skills") == "excluded root dir skills"
    assert mod._prune_reason(tmp_path / "tests" / "skills") is None
    assert mod._prune_reason(tmp_path / "tests" / "evals") is None
    assert mod._prune_reason(tmp_path / "packages" / "web" / "node_modules") == "excluded dir node_modules"
    assert mod._prune_reason(tmp_path / "src") is None
    assert mod._prune_reason(tmp_path / "env_no_marker") is None
    assert mod._prune_reason(tmp_path / "dockerfile_env") is None
    assert mod._prune_reason(tmp_path / "tests") is None
