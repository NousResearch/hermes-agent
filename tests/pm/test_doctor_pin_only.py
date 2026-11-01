"""Doctor distinguishes external pins from locally realized packages."""
from pm import paths, registry
from pm.cli import cmd_doctor
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.store import current_target, tree_digest

import pytest


class ExternalPin(Package):
    name = "external-pin"
    pin_only = True

    def verify(self, entry, target):
        raise AssertionError("an external pin has no local bytes to verify")


class LocalPackage(Package):
    name = "local-package"

    def verify(self, entry, target):
        return "" if (entry / "payload").is_file() else "missing payload"


@pytest.mark.parametrize("local_state", ["absent", "healthy", "tampered", "legacy"])
def test_doctor_skips_external_pin_but_checks_local_packages(tmp_path, monkeypatch, capsys, local_state):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    monkeypatch.setitem(registry._packages, ExternalPin.name, ExternalPin())
    monkeypatch.setitem(registry._packages, LocalPackage.name, LocalPackage())
    lock = Lockfile(lock_path)
    target = current_target()
    lock.set_pin(ExternalPin.name, "sha256:external", {
        target: {"url": "docker://example.invalid/image@sha256:external"},
    })
    digest = "a" * 64
    lock.set_pin(LocalPackage.name, "1.0", {
        target: {"url": "https://example.invalid/local.tar.gz", "sha256": digest},
    })
    lock.save()
    if local_state != "absent":
        entry = runtime / "local-entry"
        entry.mkdir(parents=True)
        (entry / "payload").write_text("original")
        facts = Facts(paths.facts_path())
        identity = {} if local_state == "legacy" else {
            "target": target, "artifacts": [digest], "digest": tree_digest(entry),
        }
        facts.record(LocalPackage.name, "1.0", entry.name, {}, runtime, **identity)
        if local_state == "tampered":
            (entry / "payload").write_text("changed")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    error = None
    try:
        result = cmd_doctor(None)
    except KeyError as exc:
        error = exc
    assert error is None, f"doctor interpreted an external pin as a local archive: {error}"
    assert result == (0 if local_state == "healthy" else 1)
    output = capsys.readouterr().out
    assert "external-pin: pin only" in output
    expected = {"absent": "not installed", "healthy": "local-package 1.0",
                "tampered": "realized bytes do not match", "legacy": "legacy fact"}
    assert expected[local_state] in output
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
