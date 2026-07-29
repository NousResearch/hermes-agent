"""Tests for the launch-time stale-bytecode sweep (checkout fingerprint guard).

Bug class: the checkout's ``.py`` files change (``hermes update``, manual
``git pull``, ZIP update) while ``__pycache__`` retains bytecode compiled
from the previous revision; the next process to import trusts the stale
``.pyc`` and dies with ``cannot import name ...`` (#6207, #60242).

The launch-time guard compares the current checkout fingerprint against the
last-validated stamp and sweeps ``__pycache__`` once when they diverge —
covering paths no update-time clear can reach (manual pulls, pre-hardening
updaters).
"""

from pathlib import Path

from hermes_cli import main as hermes_main


def _make_repo(tmp_path: Path, sha: str = "a" * 40) -> Path:
    """Minimal git checkout layout that _read_git_revision_fingerprint groks."""
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")
    return repo


def _make_pycache(repo: Path, subdir: str = "hermes_cli") -> Path:
    cache = repo / subdir / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "main.cpython-311.pyc").write_bytes(b"stale")
    return cache


def test_sweep_clears_pycache_when_checkout_changed(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path, sha="b" * 40)
    cache = _make_pycache(repo)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    # Stamp records a different (older) fingerprint.
    (repo / hermes_main._BYTECODE_FINGERPRINT_FILE).write_text(
        "git:refs/heads/main:" + "a" * 40, encoding="utf-8"
    )

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert not cache.exists()
    # Stamp updated to the current fingerprint.
    recorded = (repo / hermes_main._BYTECODE_FINGERPRINT_FILE).read_text(encoding="utf-8")
    assert recorded.strip().endswith("b" * 40)


def test_sweep_first_launch_clears_and_records(monkeypatch, tmp_path):
    """No stamp yet (first launch with the guard) → sweep once, record."""
    repo = _make_repo(tmp_path, sha="d" * 40)
    cache = _make_pycache(repo)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert not cache.exists()
    assert (repo / hermes_main._BYTECODE_FINGERPRINT_FILE).exists()


def test_record_bytecode_fingerprint_writes_atomically(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path, sha="f" * 40)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    hermes_main._record_bytecode_fingerprint()

    stamp = repo / hermes_main._BYTECODE_FINGERPRINT_FILE
    assert stamp.read_text(encoding="utf-8").endswith("f" * 40)
    assert not stamp.with_name(stamp.name + ".tmp").exists()


def test_sweep_skips_venv_and_git_dirs(monkeypatch, tmp_path):
    """The underlying clear must not touch venv/node_modules bytecode."""
    repo = _make_repo(tmp_path, sha="9" * 40)
    repo_cache = _make_pycache(repo, "hermes_cli")
    venv_cache = repo / "venv" / "lib" / "__pycache__"
    venv_cache.mkdir(parents=True)
    (venv_cache / "x.pyc").write_bytes(b"keep")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)

    hermes_main._sweep_stale_bytecode_if_checkout_changed()

    assert not repo_cache.exists()
    assert venv_cache.exists()

# ---------------------------------------------------------------------------
# Plugin-update sibling site: __pycache__ under ~/.hermes/plugins/<name>
# ---------------------------------------------------------------------------

def test_clear_plugin_bytecode_removes_nested_caches(tmp_path):
    from hermes_cli import plugins_cmd

    plugin = tmp_path / "myplugin"
    top = plugin / "__pycache__"
    nested = plugin / "sub" / "__pycache__"
    top.mkdir(parents=True)
    nested.mkdir(parents=True)
    (top / "a.pyc").write_bytes(b"stale")
    (nested / "b.pyc").write_bytes(b"stale")

    removed = plugins_cmd._clear_plugin_bytecode(plugin)

    assert removed == 2
    assert not top.exists()
    assert not nested.exists()


def test_clear_plugin_bytecode_never_raises_on_missing_dir(tmp_path):
    from hermes_cli import plugins_cmd

    assert plugins_cmd._clear_plugin_bytecode(tmp_path / "nope") == 0
