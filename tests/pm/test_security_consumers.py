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
from tests.pm._range_server import dl_server as dl_server  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


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
            files = {"checksums.txt.asc": signature, "public-key.asc": b"public-key"}
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
    from agent.proxy_sources.iron_proxy import find_iron_proxy, install_iron_proxy
    return find_iron_proxy, install_iron_proxy


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("name", ["bws", "iron-proxy"])
def test_consumer_lifecycle(consumer_store, monkeypatch, tmp_path, name):
    find, install = consumer(name)

    monkeypatch.setenv("PATH", "")
    pin(consumer_store, name)
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
@pytest.mark.parametrize("name,verifier", [("iron-proxy", "gpg")])
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
    _, install = consumer(name)
    pin(consumer_store, name, signature=b"accept")
    old = Path(install())
    pin(consumer_store, name, "2", signature=b"reject")
    with pytest.raises(RuntimeError, match="GPG signature verification"):
        install(force=True)
    assert pm.installed_package(name, allow_outdated=True).binary == old
    assert subprocess.check_output([old], text=True).strip() == f"{name} fixture 1"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert "--import" in calls[0] and "--verify" in calls[1]
    assert not Path(calls[0][calls[0].index("--homedir") + 1]).exists()


@pytest.mark.platforms("posix")
def test_managed_consumers_run_their_business_protocol(consumer_store, monkeypatch):
    from agent.secret_sources.bitwarden import fetch_bitwarden_secrets, install_bws
    from agent.proxy_sources.iron_proxy import install_iron_proxy, iron_proxy_version

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("MY_PRIVATE_TOKEN", "not-for-proxy")
    pin(consumer_store, "bws", payload=(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "if '--version' in sys.argv: print('bws fixture')\n"
        "else:\n"
        " assert sys.argv[1:] == ['secret','list','project','--output','json']\n"
        " assert os.environ['BWS_SERVER_URL'] == 'https://vault.example.invalid'\n"
        " print(json.dumps([{'key':'FIXTURE_VALUE','value':os.environ['BWS_ACCESS_TOKEN']}]))\n"
    ).encode())
    install_bws()
    assert fetch_bitwarden_secrets(access_token="local-token", project_id="project", use_cache=False,
                                  server_url="https://vault.example.invalid") == (
        {"FIXTURE_VALUE": "local-token"}, []
    )
    pin(consumer_store, "iron-proxy", payload=(
        f"#!{sys.executable}\nimport os,sys\n"
        "assert 'MY_PRIVATE_TOKEN' not in os.environ\n"
        "assert sys.argv[1:] == ['--version']\nprint('private proxy fixture')\n"
    ).encode())
    proxy = install_iron_proxy()
    assert iron_proxy_version(proxy) == "private proxy fixture"


@pytest.mark.platforms("posix")
def test_nonregular_security_binary_is_never_published(consumer_store, monkeypatch):
    monkeypatch.setenv("PATH", "")
    _, install = consumer("iron-proxy")
    pin(consumer_store, "iron-proxy")
    old = install()
    pin(consumer_store, "iron-proxy", "2", link=True)
    with pytest.raises(pm.InstallError, match="not a regular file"):
        install()
    assert str(pm.installed_package("iron-proxy", allow_outdated=True).binary) == str(old)


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
@pytest.mark.parametrize("name", ["iron-proxy"])
def test_locked_provenance_is_required_even_without_a_verifier(consumer_store, monkeypatch, name):
    monkeypatch.setenv("PATH", "")
    _, install = consumer(name)
    pin(consumer_store, name, signature=b"pinned-signature")
    old = Path(install())  # absent verifier permits hash-verified installation
    pin(consumer_store, name, "2", signature=b"pinned-signature")
    missing = f"/{name}/2/checksums.txt.asc"
    raw = RangeHandler.payloads.pop(missing)
    with pytest.raises(pm.InstallError):
        install()
    assert pm.installed_package(name, allow_outdated=True).binary == old
    RangeHandler.payloads[missing] = raw
    assert subprocess.check_output([install()], text=True).strip() == f"{name} fixture 2"