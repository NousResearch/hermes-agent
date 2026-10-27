"""A cross-target pin failure must identify the target that failed (#145416)."""

from email.message import Message
import urllib.error

from pm import cli, registry
from pm.lock import Lockfile
from pm.package import Package
from pm.update import Resolved


class CrossTargetPackage(Package):
    name = "fixture-pin"

    def missing_reason(self, target):
        return None

    def fetch_urls(self, version, target):
        return [f"https://fixture.example/{target}/tool-{version}.tgz"]

    def known_sha256(self, version, url):
        if "linux-arm64" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        return "0" * 64


def test_failed_cross_target_pin_names_target_and_keeps_lock(tmp_path, monkeypatch, capsys):
    from pm import receipt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    package = CrossTargetPackage()
    monkeypatch.setitem(registry._packages, package.name, package)
    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin(package.name, "1.0", {})
    lock.save()
    before = lock.path.read_bytes()
    decision = Resolved(package.name, "1.0", "semver", version="2.0",
                        per_target={"linux-x64": "2.0", "linux-arm64": "2.0"})

    assert cli._apply_pins([decision], lock) == 1

    output = capsys.readouterr().out
    assert "target linux-arm64" in output
    assert "https://fixture.example/linux-arm64/tool-2.0.tgz" in output
    assert "404" in output
    assert lock.path.read_bytes() == before
    latest = receipt.latest()
    assert latest is not None
    assert "target linux-arm64" in str(latest["steps"])
