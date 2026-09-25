"""Complete desktop dependency preparation and its job-local consume contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.releases.versioning import parse_attempt_ref


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=source, text=True, encoding="utf-8").rstrip("\r\n")


def require_source(source: Path, commit: str) -> None:
    if git(source, "rev-parse", "HEAD") != commit:
        raise ValueError("preparation source checkout changed revision; prepare again")
    status = git(source, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError(f"desktop builds require a clean source checkout at the admitted revision:\n{status}")


def fingerprint(path: Path) -> str:
    if path.is_dir():
        from pm.store import tree_digest
        return tree_digest(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_owned(request: BuildRequest, paths: list[Path]) -> None:
    require_output_roots(request.source)
    roots = (request.work.resolve(), request.cache.resolve(),
             (request.source / "apps/desktop/build").resolve())
    for path in paths:
        if not path.is_absolute() or not any(path.resolve().is_relative_to(root) for root in roots):
            raise ValueError(f"preparation input is outside owned roots: {path}")


def require_output_roots(source: Path) -> None:
    from pm.filesystem import is_junction

    for relative in (".build", "apps", "apps/desktop", "apps/desktop/build", "apps/desktop/dist", "apps/desktop/release"):
        path = source / relative
        if path.is_symlink() or (path.exists() and is_junction(path)):
            raise ValueError(f"desktop output root must not be a symlink or junction: {path}")


def selected_tool_inputs(request: BuildRequest) -> list[Path]:
    from pm.build_operations import verified_tools
    from pm.lock import Lockfile

    selection = verified_tools(["python", "node", "npm"], source_store=request.cache / "tools",
                               target=request.target, lock=Lockfile(request.source / "pm/lock.json"))
    return [entry.path for entry in selection.entries.values()]


def validate_channel_request(value: dict) -> dict:
    """Add native packaging constraints to the shared release protocol."""
    from hermes_cli.release_channels import validate_request
    from scripts.releases.bundle_env import validate

    validate_request(value)
    validate(value["bundleEnv"])
    identity = value["identity"]
    for field in ("cliName", "windowsExecutableName", "appNamePascal", "artifactNamePascal"):
        if re.fullmatch(r"(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])", identity[field].split(".")[0]):
            raise ValueError(f"reserved channel native identity {field}")
    if identity["windowsExecutableName"].endswith((".", " ")):
        raise ValueError("invalid channel native identity windowsExecutableName")
    if not value["publicBase"].startswith("https://"):
        raise ValueError("channel publicBase must be HTTPS for packaged clients")
    return deepcopy(value)


@dataclass(frozen=True)
class BuildRequest:
    source: Path
    work: Path
    cache: Path
    commit: str
    tag: str | None
    version: str
    variant: str
    target: str
    bundle_env: dict[str, str | None]
    channel_request: dict | None = None
    release_epoch: int | None = None

    @classmethod
    def create(cls, source: Path, *, tag: str | None, commit: str | None, variant: str,
               work: Path, cache: Path, bundle_env: dict[str, str | None],
               channel_request: dict | None = None, release_commit: str | None = None) -> BuildRequest:
        from pm.store import current_target
        from scripts.bundles.desktop import release_version
        from scripts.releases.bundle_env import validate
        from scripts.releases.commit_build import require_commit, version_at
        from scripts.termux.deb_version import channel_for_tag

        if channel_request is not None:
            channel_request = validate_channel_request(channel_request)
            if variant != "bundled":
                raise ValueError("channel builds currently support only the bundled variant")
            if tag or commit not in (None, channel_request["commit"]):
                raise ValueError("channel request conflicts with tag or commit selection")
            if bundle_env and bundle_env != channel_request["bundleEnv"]:
                raise ValueError("bundle defaults conflict with channel request")
            commit = channel_request["commit"]
            bundle_env = channel_request["bundleEnv"]
        if variant == "store" and (commit or not tag or channel_for_tag(tag) != "stable"):
            raise ValueError("Store packaging requires a stable release tag")
        if variant not in {"bundled", "store", "light"}:
            raise ValueError("invalid desktop variant")
        if bool(tag) == bool(commit):
            raise ValueError("exactly one of --tag or --commit is required")
        bundle_env = validate(bundle_env)
        if bundle_env and tag:
            raise ValueError("Bundle environment defaults require a commit build")
        source, work, cache = source.resolve(), work.resolve(), cache.resolve()
        require_output_roots(source)
        if work == cache or work.is_relative_to(cache) or cache.is_relative_to(work):
            raise ValueError("preparation work and cache must be separate directories")
        for destination in (work, cache):
            if source == destination or source.is_relative_to(destination):
                raise ValueError("preparation output must not contain the source checkout")
        if commit:
            commit = require_commit(commit)
            require_source(source, commit)
            version = version_at(source, commit)
        else:
            assert tag is not None  # The exclusive selection was checked above.
            version = release_version(source, tag)
            commit = require_commit(release_commit) if release_commit else \
                git(source, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
        release_epoch = None
        if tag:
            canary = re.fullmatch(r"v\d+\.\d+\.\d+\+canary\.(20\d{6}T\d{6}Z)", tag)
            if canary:
                release_epoch = int(datetime.strptime(canary.group(1), "%Y%m%dT%H%M%SZ")
                                    .replace(tzinfo=timezone.utc).timestamp())
            else:
                claim_tag = os.environ.get("RELEASE_CLAIM_TAG", "")
                # The payload version stays plain; the claim must name the same
                # version as an attempt ref, never the checkout.
                parsed = parse_attempt_ref(claim_tag)
                if parsed is None or parsed[0] != version:
                    raise ValueError("stable preparation requires its exact claim tag")
                claim_object = os.environ.get("RELEASE_CLAIM_OBJECT", "")
                if not re.fullmatch(r"[a-f0-9]{40}", claim_object) or \
                        git(source, "rev-parse", f"refs/tags/{claim_tag}") != claim_object:
                    raise ValueError("stable preparation requires its exact claim tag object")
                metadata = git(source, "cat-file", "-p", claim_object)
                tagger = re.search(r"^tagger .* (\d+) [+-]\d{4}$", metadata, re.MULTILINE)
                if tagger is None:
                    raise ValueError("stable claim has no immutable tagger timestamp")
                release_epoch = int(tagger.group(1))
        require_source(source, commit)
        if channel_request is not None:
            if version != channel_request["sourceVersion"]:
                raise ValueError("channel sourceVersion differs from checkout project version")
            version = channel_request["version"]
        return cls(source, work, cache, commit, tag, version, variant, current_target(), bundle_env,
                   channel_request, release_epoch)

    def data(self) -> dict:
        return {**asdict(self), "source": str(self.source), "work": str(self.work), "cache": str(self.cache)}

    @classmethod
    def from_data(cls, data: dict) -> BuildRequest:
        request = cls(**{**deepcopy(data), **{name: Path(data[name]) for name in ("source", "work", "cache")}})
        request.validate_channel()
        return request

    def validate_channel(self) -> None:
        if self.channel_request is None:
            return
        channel = validate_channel_request(self.channel_request)
        if (self.variant != "bundled" or self.tag is not None or self.commit != channel["commit"]
                or self.version != channel["version"] or self.bundle_env != channel["bundleEnv"]):
            raise ValueError("prepared build differs from admitted channel request")

    def identity_digest(self) -> str:
        return hashlib.sha256(json.dumps(self.data(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def workspaces(self) -> list[str]:
        return ["apps/desktop"] + ([] if self.variant == "light" else ["ui-tui", "web"])


@dataclass(frozen=True)
class PreparedDesktop:
    request: BuildRequest
    python: Path
    node: Path
    icon_python: Path
    native: Path
    packager: Path
    payload: Path | None
    digests: dict[str, str]
    native_toolchain: str
    request_digest: str | None = None

    @classmethod
    def record(cls, request: BuildRequest, *, python: Path, node: Path, icon_python: Path,
               native: Path, packager: Path, payload: Path | None, native_toolchain: str) -> PreparedDesktop:
        files = [python, node, icon_python.parent.parent, native, packager]
        if payload is not None:
            files.append(payload)
        require_owned(request, files)
        files.extend(selected_tool_inputs(request))
        return cls(request, python, node, icon_python, native, packager, payload,
                   {str(path): fingerprint(path) for path in files}, native_toolchain, request.identity_digest())

    def write(self, path: Path) -> None:
        from pm.lock import _write
        _write(path, {"schema": 1, "request": self.request.data(),
                      **{name: str(value) if value is not None else None for name, value in (
                          ("python", self.python), ("node", self.node), ("icon_python", self.icon_python),
                          ("native", self.native), ("packager", self.packager), ("payload", self.payload))},
                      "digests": self.digests, "native_toolchain": self.native_toolchain,
                      "request_digest": self.request_digest})

    @classmethod
    def load(cls, path: Path) -> PreparedDesktop:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if data.pop("schema") != 1:
                raise ValueError("unsupported preparation schema")
            request = BuildRequest.from_data(data.pop("request"))
            for name in ("python", "node", "icon_python", "native", "packager", "payload"):
                if data[name] is not None:
                    data[name] = Path(data[name])
            return cls(request=request, **data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"desktop preparation is missing or invalid: {path}; run preparation again") from exc

    def validate(self) -> None:
        from pm.store import current_target
        self.request.validate_channel()
        if (self.request_digest is not None or self.request.channel_request is not None) and self.request_digest != self.request.identity_digest():
            raise ValueError("preparation request identity changed; prepare again")
        require_source(self.request.source, self.request.commit)
        if self.request.target != current_target():
            raise ValueError("preparation target differs from this host")
        paths = [self.python, self.node, self.icon_python.parent.parent, self.native, self.packager]
        paths.extend(selected_tool_inputs(self.request))
        if self.payload is not None:
            paths.append(self.payload)
        require_owned(self.request, paths)
        if set(self.digests) != {str(path) for path in paths}:
            raise ValueError("preparation input inventory changed")
        for path in paths:
            if not path.is_absolute() or not path.exists() or fingerprint(path) != self.digests[str(path)]:
                raise ValueError(f"preparation input changed or is missing: {path}; prepare again")


def prepare(request: BuildRequest) -> Path:
    from scripts.bundles.desktop_inputs import build_lock

    request.validate_channel()
    if request.channel_request is not None and not request.target.startswith(("darwin-", "win32-")):
        raise ValueError("channel builds require a supported native macOS or Windows target")
    require_source(request.source, request.commit)
    with build_lock(request.source):
        return _prepare(request)


def _prepare(request: BuildRequest) -> Path:
    from scripts.bundles.desktop_toolchain import run_preparation
    from pm.lock import _write
    from hermes_cli.runtime_state import _lock

    require_source(request.source, request.commit)
    owner = request.work / ".desktop-preparation"
    if request.work.exists():
        if not owner.is_file() or owner.read_text(encoding="utf-8-sig") != str(request.source):
            raise ValueError(f"preparation work directory is not owned by this checkout: {request.work}")
    else:
        request.work.mkdir(parents=True)
        owner.write_text(str(request.source), encoding="utf-8")
    with (request.work / ".lock").open("a+b") as lock:
        if not _lock(lock.fileno(), wait=False):
            raise ValueError("another desktop preparation is using this work directory")
        result = request.work / "prepared.json"
        result.unlink(missing_ok=True)
        request_file = request.work / "request.json"
        _write(request_file, request.data())
        status = run_preparation(request.source, request.work, request.cache, request_file)
        if status:
            result.unlink(missing_ok=True)
            raise RuntimeError(f"desktop dependency preparation failed (exit {status}); see provider output above")
        PreparedDesktop.load(result).validate()
        return result


def prepare_in_worker(request: BuildRequest) -> Path:
    from scripts.bundles.desktop import run
    from scripts.bundles.desktop_toolchain import prepare_tools
    from scripts.build.icon_environment import prepare_icon_environment
    from scripts.bundles.native import prepare_native

    require_source(request.source, request.commit)
    from scripts.bundles.desktop_inputs import identity_environment
    request.validate_channel()
    python, node, env = prepare_tools(request.source, request.work, request.cache, os.environ)
    env = identity_environment(request, request.variant, env)
    native_toolchain = Path(env["UV_CACHE_DIR"]).name
    run([str(node), "scripts/build/node-deps.mjs", "--source", str(request.source), "--reuse",
         "--native-toolchain", native_toolchain,
         *[arg for name in request.workspaces() for arg in ("--workspace", name)]], cwd=request.source, env=env)
    icon_environment = request.work / "icon-environment"
    if icon_environment.exists():
        if icon_environment.is_symlink():
            raise ValueError("icon environment must be preparation-owned, not a symlink")
        shutil.rmtree(icon_environment)
    icon_python = prepare_icon_environment(request.source, icon_environment, request.cache / "python/build")
    native = request.source / "apps/desktop/build/native-deps"
    run([str(node), "apps/desktop/scripts/stage-native-deps.mjs", "--source", str(request.source),
         "--out", str(native), "--native-toolchain", native_toolchain], cwd=request.source, env=env)
    packager = request.work / "packager"
    dmg_args = []
    if request.target.startswith("darwin-"):
        from apps.desktop.scripts.prepare_dmgbuild import prepare_dmgbuild
        dmg = prepare_dmgbuild(request.cache / "tools", request.cache / "python/build")
        dmg_args = ["--dmgbuild", str(dmg)]
    run([str(node), "apps/desktop/scripts/prepare-packaging-tools.mjs", "--source", str(request.source),
         "--out", str(packager), "--cache", str(request.cache / "packager"), "--target", request.target, *dmg_args],
        cwd=request.source, env=env)
    run([str(node), "apps/desktop/scripts/probe-prepared-native.mjs", "--source", str(request.source),
         "--native-deps", str(native), "--packaging", str(packager / "prepared.json"),
         "--out", str(request.work / "native-probe"), "--native-toolchain", native_toolchain],
        cwd=request.source, env=env)
    payload = None
    if request.variant != "light":
        payload = prepare_native(out=request.source / "apps/desktop/build/agent-payload", ref=request.commit,
                                 source=request.source, cache=Path(env["UV_CACHE_DIR"]),
                                 tools=request.cache / "tools", env=env)
    require_source(request.source, request.commit)
    prepared = PreparedDesktop.record(request, python=python, node=node, icon_python=icon_python,
                                      native=native, packager=packager / "prepared.json", payload=payload,
                                      native_toolchain=native_toolchain)
    result = request.work / "prepared.json"
    prepared.write(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", required=True)
    args = parser.parse_args()
    request = BuildRequest.from_data(json.loads(args.request.read_text(encoding="utf-8-sig")))
    prepare_in_worker(request)


if __name__ == "__main__":
    main()
