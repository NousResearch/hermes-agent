"""A python entry's site-packages is a mutable layer, never digest damage (#138689).

Users `pip install` into `~/.hermes/tools/python-*/lib/python3.x/site-packages`
after publication, and a desktop bundle preinstalls its gateway deps there.
A recorded digest that counts those bytes sees drift on the next explicit
check and "repairs" the entry by re-realizing the bare upstream archive —
wiping both layers. These tests pin the exclusion and the lenient verify.
"""
from pathlib import Path

from pm import paths
from pm.install import _entry_verified
from pm.lock import Facts
from pm.registry import get_package
from pm.store import Store, current_target, tree_digest
from tests.pm._fixtures import served as served  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
from tests.pm.test_pm_core import pm_env as core_env  # the imported fixture binds into this module


def _python_entry(store_root: Path, target: str) -> tuple[Path, dict]:
    """A minimal staged python entry: the binary verify() probes, plus deps."""
    entry = store_root / get_package("python").store_entry("3.14.7+20260901", target)
    (entry / "bin").mkdir(parents=True)
    (entry / "bin" / ("python.exe" if target.startswith("win32") else "python3")).write_text("# shim")
    site = entry / "lib" / "python3.14" / "site-packages"
    site.mkdir(parents=True)
    (site / "yaml").mkdir()
    (site / "yaml" / "__init__.py").write_text("")
    return entry, {"entry": entry.name}


def test_tree_digest_excludes_named_dirs(tmp_path):
    root = tmp_path / "tree"
    site = root / "lib" / "python3.14" / "site-packages"
    site.mkdir(parents=True)
    (site / "user_pkg.py").write_text("bundled")
    (root / "stdlib.py").write_text("published")
    baseline = tree_digest(root, ("site-packages",))

    (site / "user_pkg.py").write_text("pip installed over it")
    assert tree_digest(root, ("site-packages",)) == baseline
    (site / "new_pkg.py").write_text("pip install .")
    assert tree_digest(root, ("site-packages",)) == baseline

    (root / "stdlib.py").write_text("tampered")
    assert tree_digest(root, ("site-packages",)) != baseline


def test_python_digest_drift_keeps_the_entry_verified(tmp_path):
    """pip installs must not mark the entry for destructive re-realization."""
    target = current_target()
    store = Store(tmp_path)
    entry, fact = _python_entry(tmp_path, target)
    package = get_package("python")

    # A digest recorded by an older hermes counts site-packages.
    fact["digest"] = tree_digest(entry)
    assert _entry_verified(package, fact, store, target)
    (entry / "lib" / "python3.14" / "site-packages" / "spacy_model").mkdir()
    assert _entry_verified(package, fact, store, target)

    # A digest recorded under the skip stays matching, drift or not.
    fact["digest"] = tree_digest(entry, package.digest_skip_dirs)
    (entry / "lib" / "python3.14" / "site-packages" / "spacy_model" / "meta.json").write_text("{}")
    assert _entry_verified(package, fact, store, target)


def test_digest_drift_still_fails_without_a_skip(tmp_path, monkeypatch):
    """Packages without a mutable layer keep the strict digest contract."""
    target = current_target()
    store = Store(tmp_path)
    entry, fact = _python_entry(tmp_path, target)
    package = get_package("python")
    monkeypatch.setattr(package, "digest_skip_dirs", ())  # stand-in for immutable packages

    fact["digest"] = tree_digest(entry)
    assert _entry_verified(package, fact, store, target)
    (entry / "lib" / "python3.14" / "site-packages" / "new.py").write_text("drift")
    assert not _entry_verified(package, fact, store, target)


def test_refresh_digests_records_python_without_site_packages(tmp_path):
    """Bundle facts must record the skip spelling so a user pip install
    after the first boot never reads as drift."""
    target = current_target()
    entry, _ = _python_entry(tmp_path, target)
    facts_path = tmp_path / "facts.json"
    facts = Facts(facts_path)
    facts.record(
        "python", "3.14.7+20260901", entry.name, {}, tmp_path,
        target=target, artifacts=["a" * 64], digest=tree_digest(entry),
    )
    refreshed = facts.refresh_digests(tmp_path)
    assert refreshed == 1
    assert facts.get("python")["digest"] == tree_digest(
        entry, get_package("python").digest_skip_dirs)


def test_doctor_tolerates_drift_in_a_packages_mutable_layer(core_env, monkeypatch, capsys):
    """A write into a digest_skip_dirs dir is runtime state, not tampering:
    doctor reports it softly and exits clean — flagging it sends the user
    into the repair that re-realizes the bare upstream archive (#138689)."""
    from pm.cli import cmd_doctor
    from pm.install import ensure

    ensure("faketool", base_env={})
    monkeypatch.setattr(get_package("faketool"), "digest_skip_dirs",
                        ("site-packages",))
    facts_path = paths.facts_path()
    fact = Facts(facts_path).get("faketool")
    entry = paths.store_root() / fact["entry"]
    (entry / "site-packages").mkdir()
    (entry / "site-packages" / "user_pkg.py").write_text("pip installed")
    # Re-record the digest an older hermes would have: counting the layer.
    Facts(facts_path).record(
        "faketool", fact["version"], fact["entry"], fact.get("env", {}), paths.store_root(),
        target=fact["target"], artifacts=fact["artifacts"], digest=tree_digest(entry))
    assert cmd_doctor(None) == 0
    assert "mutable layer changed" in capsys.readouterr().out
