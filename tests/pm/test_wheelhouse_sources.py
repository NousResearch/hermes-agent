"""Native wheel delivery is optional, hash-locked, and never masks corruption."""

import hashlib
import json
import shutil
import tomllib

import pytest

from pm.downloader import HashError, Source
from pm.package import InstallError
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


def _project(tmp_path, pins):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[tool.hermes.win-arm64-wheels]\n" + "\n".join(
            f'{name} = {{ filename = "{filename}", sha256 = "{sha}" }}'
            for name, (filename, sha) in pins.items()) + "\n", encoding="utf-8")
    (project / "uv.lock").write_text("\n".join(
        f'[[package]]\nname = "{name}"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'sdist = {{ url = "https://example.invalid/{name}-1.0.tar.gz", '
        f'hash = "sha256:{hashlib.sha256(name.encode()).hexdigest()}" }}\n'
        for name in pins), encoding="utf-8")
    return project


def _source(server):
    def source(filename, sha, dest):
        return Source(url(server, "/release/" + filename), dest, sha,
                      (url(server, "/r2/" + filename),))
    return source


def test_missing_native_wheel_uses_locked_registry_sdist(tmp_path, dl_server):
    from pm.wheel_sources import stage_wheels

    first, second = "first-1.0-cp314-abi3-win_arm64.whl", "second-1.0-cp314-abi3-win_arm64.whl"
    body = b"verified native wheel"
    project = _project(tmp_path, {
        "first": (first, hashlib.sha256(body).hexdigest()),
        "second": (second, hashlib.sha256(b"not published").hexdigest()),
    })
    RangeHandler.payloads["/r2/" + first] = body
    wheels = tmp_path / "wheels"
    result = stage_wheels(project, project / "uv.lock", wheels, _source(dl_server), target="win32-arm64")
    assert result.available == ("first",) and result.missing == ("second",)
    assert (wheels / first).read_bytes() == body
    assert RangeHandler.requests_seen.index("/release/" + first) < RangeHandler.requests_seen.index("/r2/" + first)
    assert RangeHandler.requests_seen.index("/release/" + second) < RangeHandler.requests_seen.index("/r2/" + second)


def test_corrupt_native_wheel_does_not_try_another_url(tmp_path, dl_server):
    from pm.wheel_sources import stage_wheels

    filename = "first-1.0-cp314-abi3-win_arm64.whl"
    project = _project(tmp_path, {"first": (filename, hashlib.sha256(b"reviewed").hexdigest())})
    RangeHandler.payloads["/release/" + filename] = b"tampered"
    RangeHandler.payloads["/r2/" + filename] = b"reviewed"
    with pytest.raises(HashError):
        stage_wheels(project, project / "uv.lock", tmp_path / "wheels", _source(dl_server), target="win32-arm64")
    assert not any(path.startswith("/r2/") for path in RangeHandler.requests_seen)


def test_repair_rebinds_verified_wheels_without_current_inputs(tmp_path):
    from pm.wheel_sources import relocate_wheels

    filename = "first-1.0-cp314-abi3-win_arm64.whl"
    body = b"locked wheel"
    digest = hashlib.sha256(body).hexdigest()
    recorded = _project(tmp_path, {
        "first": (filename, digest),
        "second": ("second-1.0-cp314-abi3-win_arm64.whl", hashlib.sha256(b"absent").hexdigest()),
    })
    directory = recorded / "wheels"
    directory.mkdir()
    (directory / filename).write_bytes(body)
    original_lock = (recorded / "uv.lock").read_text(encoding="utf-8")
    registry_row = original_lock[original_lock.index('[[package]]\nname = "second"'):]

    def escaped_path(path):
        value = str(path)
        # Different TOML spellings can decode to the same filesystem path.
        return '"' + '\\u' + f'{ord(value[0]):04x}' + json.dumps(value)[2:]

    (recorded / "uv.lock").write_text(
        '[[package]]\nname = "first"\nversion = "1.0"\n'
        f'source = {{ registry = {escaped_path(directory)} }}\n'
        f'wheels = [{{ path = {escaped_path(directory / filename)} }}]\n'
        + registry_row, encoding="utf-8")

    repair = tmp_path / "repair"
    shutil.copytree(recorded, repair)
    assert relocate_wheels(repair, recorded, target="win32-arm64") == ("second",)
    rows = tomllib.loads((repair / "uv.lock").read_text(encoding="utf-8"))["package"]
    assert rows[0]["source"]["registry"] == str(repair / "wheels")
    assert rows[0]["wheels"][0]["path"] == str(repair / "wheels" / filename)
    assert rows[1]["source"]["registry"] == "https://pypi.org/simple"

    corrupt = tmp_path / "corrupt-repair"
    shutil.copytree(recorded, corrupt)
    (corrupt / "wheels" / filename).write_bytes(b"tampered")
    with pytest.raises(HashError):
        relocate_wheels(corrupt, recorded, target="win32-arm64")


def test_private_resolution_preserves_wheel_and_sdist_pins(tmp_path):
    from pm.wheel_sources import WheelSelection, verify_selected_wheels

    filename = "first-1.0-cp314-abi3-win_arm64.whl"
    digest = hashlib.sha256(b"native wheel").hexdigest()
    recorded = _project(tmp_path, {
        "first": (filename, digest),
        "second": ("second-1.0-cp314-abi3-win_arm64.whl", "a" * 64),
    })
    resolved = tmp_path / "resolved"
    resolved.mkdir()
    shutil.copy2(recorded / "pyproject.toml", resolved / "pyproject.toml")
    directory = resolved / "wheels"
    directory.mkdir()
    (directory / filename).write_bytes(b"native wheel")
    source = (recorded / "uv.lock").read_text(encoding="utf-8")
    second = source[source.index('[[package]]\nname = "second"'):]
    private = ('[[package]]\nname = "first"\nversion = "1.0"\n'
               f'source = {{ registry = "{directory}" }}\n'
               f'wheels = [{{ path = "{directory / filename}" }}]\n'
               + second)
    lock = resolved / "uv.lock"
    lock.write_text(private, encoding="utf-8")
    selection = WheelSelection(("first",), ("second",), {"first": "1.0", "second": "1.0"})
    verify_selected_wheels(recorded / "uv.lock", lock, directory, selection)
    wrong_sdist = hashlib.sha256(b"second").hexdigest()
    lock.write_text(private.replace(wrong_sdist, "0" * 64), encoding="utf-8")
    with pytest.raises(InstallError, match="source distribution"):
        verify_selected_wheels(recorded / "uv.lock", lock, directory, selection)
    lock.write_text(private, encoding="utf-8")
    (directory / filename).write_bytes(b"tampered")
    with pytest.raises(HashError, match="native wheel"):
        verify_selected_wheels(recorded / "uv.lock", lock, directory, selection)
