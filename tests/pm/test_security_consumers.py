"""Security consumers cross the real PM worker/HTTP/publication boundary."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

import pm
from pm import paths
from tests.pm._range_server import RangeHandler, url
from tests.pm._range_server import dl_server as dl_server


@pytest.fixture
def consumer_store(tmp_path, monkeypatch, dl_server):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    # Only the already-prepared interpreter is supplied: the worker, downloader,
    # extractor, probes, publication and read-only selector are all real.
    monkeypatch.setattr("pm.runtime.runtime_python", lambda **kwargs: Path(sys.executable))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("consumer bypassed PM acquisition"))
    return dl_server


def pin(server, name, version="1", *, bad_hash=False, payload=None, signature=None, link=False):
    payload = payload or (
        f"#!{sys.executable}\nimport json,sys\n"
        f"print({name + ' fixture ' + version!r})\n"
    ).encode()
    buf = io.BytesIO()
    ext = ".zip" if name == "bws" else ".tar.gz"
    if ext == ".zip":
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name, payload)
    else:
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("payload" if link else name)
            info.size, info.mode = len(payload), 0o755
            tf.addfile(info, io.BytesIO(payload))
            if link:
                info = tarfile.TarInfo(name)
                info.type, info.linkname = tarfile.SYMTYPE, "payload"
                tf.addfile(info)
    archive = f"/{name}-{version}{ext}"
    RangeHandler.payloads[archive] = buf.getvalue()
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    artifacts = [{"url": url(server, archive), "sha256": "0" * 64 if bad_hash else digest}]
    if name != "bws":
        checksum_path = f"/{name}/{version}/checksums.txt"
        checksum = f"{digest}  {archive[1:]}\n".encode()
        RangeHandler.payloads[checksum_path] = checksum
        artifacts.append({"url": url(server, checksum_path), "sha256": hashlib.sha256(checksum).hexdigest()})
        if signature is not None:
            files = ({"checksums.txt.sig": signature, "checksums.txt.pem": b"certificate"}
                     if name == "tirith" else {"checksums.txt.asc": signature, "public-key.asc": b"public-key"})
            for filename, raw in files.items():
                path = f"/{name}/{version}/{filename}"
                RangeHandler.payloads[path] = raw
                artifacts.append({"url": url(server, path), "sha256": hashlib.sha256(raw).hexdigest()})
    lock = pm.Lockfile(paths.lockfile_path())
    lock.set_pin(name, version, {pm.current_target(): artifacts})
    lock.save()


def consumer(name):
    if name == "bws":
        from agent.secret_sources.bitwarden import find_bws, install_bws
        return find_bws, install_bws
    if name == "iron-proxy":
        from agent.proxy_sources.iron_proxy import find_iron_proxy, install_iron_proxy
        return find_iron_proxy, install_iron_proxy
    from tools.tirith_security import ensure_installed
    return (lambda **kw: ensure_installed(),
            lambda **kw: ensure_installed(explicit=True))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("name", ["bws", "iron-proxy", "tirith"])
def test_consumer_lifecycle(consumer_store, monkeypatch, tmp_path, name):
    find, install = consumer(name)

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("TIRITH_ENABLED", "true")
    pin(consumer_store, name)
    if name != "tirith":
        external = tmp_path / name
        external.write_text(f"#!{sys.executable}\nprint('external')\n")
        external.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert find(install_if_missing=True) == external
        assert subprocess.check_output([external], text=True).strip() == "external"
        assert pm.installed_package(name) is None
        monkeypatch.setenv("PATH", "")
    assert find(install_if_missing=True) is None  # lazy refusal, no HTTP
    assert not RangeHandler.ranges_seen
    binary = Path(install())
    assert binary == pm.installed_package(name).binary == Path(find())
    assert subprocess.check_output([binary], text=True).strip() == f"{name} fixture 1"
    requests = list(RangeHandler.ranges_seen)
    assert Path(install()) == binary
    assert RangeHandler.ranges_seen == requests
    binary.write_text("damaged executable")
    assert Path(install(force=True)) == binary
    assert subprocess.check_output([binary], text=True).strip() == f"{name} fixture 1"
    pin(consumer_store, name, "2", bad_hash=True)
    with pytest.raises(pm.InstallError, match="[Hh]ash|[Cc]hecksum|sha256"):
        install(force=True)
    assert pm.installed_package(name, allow_outdated=True).binary == binary
    assert subprocess.check_output([binary], text=True).strip() == f"{name} fixture 1"
    pin(consumer_store, name, "2")
    updated = install(force=True)
    assert subprocess.check_output([updated], text=True).strip() == f"{name} fixture 2"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("name,verifier", [("tirith", "cosign"), ("iron-proxy", "gpg")])
def test_signature_rejection_preserves_previous_selection(consumer_store, tmp_path, monkeypatch, name, verifier):
    """A real child verifier rejects after PM hashes pass; publication must not happen."""
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "verifier.jsonl"
    executable = commands / verifier
    executable.write_text(
        f"#!{sys.executable}\nimport json,pathlib,sys\n"
        f"with open({str(log)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "args=sys.argv[1:]\n"
        "flag='--signature' if '--signature' in args else '--verify'\n"
        "if flag not in args: sys.exit(0)\n"
        "sys.exit(0 if pathlib.Path(args[args.index(flag)+1]).read_bytes() == b'accept' else 1)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(commands))
    monkeypatch.setenv("TIRITH_ENABLED", "true")
    _, install = consumer(name)
    pin(consumer_store, name, signature=b"accept")
    old = Path(install())
    pin(consumer_store, name, "2", signature=b"reject")
    with pytest.raises(RuntimeError, match="cosign_verification_failed|GPG signature verification"):
        install(force=True)
    assert pm.installed_package(name, allow_outdated=True).binary == old
    assert subprocess.check_output([old], text=True).strip() == f"{name} fixture 1"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    if verifier == "cosign":
        args = calls[0]
        assert args[args.index("--certificate-identity-regexp") + 1] == (
            r"^https://github.com/sheeki03/tirith/\.github/workflows/release\.yml@refs/tags/v"
        )
        assert args[args.index("--certificate-oidc-issuer") + 1] == "https://token.actions.githubusercontent.com"
    else:
        assert "--import" in calls[0] and "--verify" in calls[1]
        assert not Path(calls[0][calls[0].index("--homedir") + 1]).exists()


@pytest.mark.platforms("posix")
def test_tirith_opt_in_background_and_explicit_override(consumer_store, tmp_path, monkeypatch):
    from tools import tirith_security as tirith
    import threading

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("TIRITH_ENABLED", "false")
    pin(consumer_store, "tirith")
    assert tirith.ensure_installed(explicit=True) is None
    assert not RangeHandler.ranges_seen
    monkeypatch.setenv("TIRITH_ENABLED", "true")
    monkeypatch.setenv("TIRITH_BIN", str(tmp_path / "missing"))
    assert tirith.ensure_installed(explicit=True) is None
    assert not RangeHandler.ranges_seen
    external = tmp_path / "external-tirith"
    external.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps({{'summary':'external'}}))\nsys.exit(1)\n")
    external.chmod(0o755)
    monkeypatch.setenv("TIRITH_BIN", str(external))
    assert tirith.check_command_security("echo hello")["summary"] == "external"
    assert not RangeHandler.ranges_seen
    monkeypatch.delenv("TIRITH_BIN")
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS")
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text("security:\n  allow_lazy_installs: true\n")
    # Hold the real worker's HTTP request so returning before completion is
    # event-proven, not an assertion against a replaced Thread constructor.
    entered, release = threading.Event(), threading.Event()
    real_get = RangeHandler.do_GET
    def blocked_get(handler):
        entered.set()
        assert release.wait(10)
        real_get(handler)
    monkeypatch.setattr(RangeHandler, "do_GET", blocked_get)
    try:
        assert tirith.ensure_installed() is None
        assert entered.wait(10)
        assert pm.installed_package("tirith") is None
    finally:
        release.set()
        for thread in tirith._install_threads.values():
            thread.join(10)
            assert not thread.is_alive()
    assert Path(tirith.ensure_installed()) == pm.installed_package("tirith").binary


@pytest.mark.platforms("posix")
def test_managed_consumers_run_their_business_protocol(consumer_store, monkeypatch):
    from agent.secret_sources.bitwarden import fetch_bitwarden_secrets, install_bws
    from agent.proxy_sources.iron_proxy import install_iron_proxy, iron_proxy_version
    from tools.tirith_security import check_command_security, ensure_installed

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("TIRITH_ENABLED", "true")
    monkeypatch.setenv("MY_PRIVATE_TOKEN", "not-for-proxy")
    pin(consumer_store, "bws", payload=(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "if '--version' in sys.argv: print('bws fixture')\n"
        "else:\n"
        " assert sys.argv[1:] == ['secret','list','project','--output','json']\n"
        " print(json.dumps([{'key':'FIXTURE_VALUE','value':os.environ['BWS_ACCESS_TOKEN']}]))\n"
    ).encode())
    install_bws()
    assert fetch_bitwarden_secrets(access_token="local-token", project_id="project", use_cache=False) == (
        {"FIXTURE_VALUE": "local-token"}, []
    )
    pin(consumer_store, "iron-proxy", payload=(
        f"#!{sys.executable}\nimport os,sys\n"
        "assert 'MY_PRIVATE_TOKEN' not in os.environ\n"
        "assert sys.argv[1:] == ['--version']\nprint('private proxy fixture')\n"
    ).encode())
    proxy = install_iron_proxy()
    assert iron_proxy_version(proxy) == "private proxy fixture"
    pin(consumer_store, "tirith", payload=(
        f"#!{sys.executable}\nimport json,sys\n"
        "if '--version' in sys.argv: print('tirith fixture')\n"
        "else:\n"
        " assert sys.argv[1:] == ['check','--json','--non-interactive','--shell','posix','--','echo hello']\n"
        " print(json.dumps({'summary':'managed scan','findings':[]}))\n sys.exit(2)\n"
    ).encode())
    ensure_installed(explicit=True)
    assert check_command_security("echo hello") == {"action": "warn", "summary": "managed scan", "findings": []}


@pytest.mark.platforms("posix")
def test_nonregular_security_binary_is_never_published(consumer_store, monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("TIRITH_ENABLED", "true")
    _, install = consumer("tirith")
    pin(consumer_store, "tirith")
    old = install()
    pin(consumer_store, "tirith", "2", link=True)
    with pytest.raises(pm.InstallError, match="not a regular file"):
        install()
    assert str(pm.installed_package("tirith", allow_outdated=True).binary) == old


@pytest.mark.platforms("posix")
def test_interrupted_consumer_download_can_retry_without_losing_selection(consumer_store, monkeypatch):
    monkeypatch.setenv("PATH", "")
    _, install = consumer("bws")
    pin(consumer_store, "bws")
    old = install()
    pin(consumer_store, "bws", "2")
    RangeHandler.abort_after = 0
    try:
        with pytest.raises(pm.InstallError):
            install(force=True)
    finally:
        RangeHandler.abort_after = None
    assert pm.installed_package("bws", allow_outdated=True).binary == old
    assert subprocess.check_output([old], text=True).strip() == "bws fixture 1"
    updated = install(force=True)
    assert subprocess.check_output([updated], text=True).strip() == "bws fixture 2"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("name", ["bws", "iron-proxy"])
def test_external_executable_does_not_require_pm_platform_support(tmp_path, monkeypatch, name):
    external = tmp_path / name
    external.write_text(f"#!{sys.executable}\nprint('external')\n")
    external.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    def unsupported(_name):
        raise RuntimeError("unsupported architecture: external tool does not need PM")
    monkeypatch.setattr(pm, "installed_package", unsupported)
    find, _ = consumer(name)
    assert find(install_if_missing=True) == external
    assert subprocess.check_output([external], text=True).strip() == "external"


@pytest.mark.platforms("posix")
def test_tirith_failed_cold_scans_make_one_attempt_then_explicit_can_retry(consumer_store, monkeypatch):
    from tools import tirith_security as tirith
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("TIRITH_ENABLED", "true")
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS")
    monkeypatch.setattr(tirith, "_crash_count", 0)
    monkeypatch.setattr(tirith, "_circuit_open", False)
    pin(consumer_store, "tirith", bad_hash=True)
    requests = []
    real_get = RangeHandler.do_GET
    def record(handler):
        requests.append(handler.path)
        real_get(handler)
    monkeypatch.setattr(RangeHandler, "do_GET", record)
    tirith.check_command_security("echo hello")
    first_attempt = list(requests)
    assert first_attempt
    tirith.check_command_security("echo hello")
    tirith.check_command_security("echo hello")
    assert requests == first_attempt
    pin(consumer_store, "tirith")
    assert tirith.ensure_installed(explicit=True)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("name", ["tirith", "iron-proxy"])
def test_locked_provenance_is_required_even_without_a_verifier(consumer_store, monkeypatch, name):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("TIRITH_ENABLED", "true")
    _, install = consumer(name)
    pin(consumer_store, name, signature=b"pinned-signature")
    old = Path(install())  # absent verifier permits hash-verified installation
    pin(consumer_store, name, "2", signature=b"pinned-signature")
    suffix = "sig" if name == "tirith" else "asc"
    missing = f"/{name}/2/checksums.txt.{suffix}"
    raw = RangeHandler.payloads.pop(missing)
    with pytest.raises(pm.InstallError):
        install()
    assert pm.installed_package(name, allow_outdated=True).binary == old
    RangeHandler.payloads[missing] = raw
    assert subprocess.check_output([install()], text=True).strip() == f"{name} fixture 2"