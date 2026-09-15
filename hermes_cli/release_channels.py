"""R2 channel wire protocol. Names are data; no local channel registry exists."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_SEQUENCE = 2**32 - 1
MAX_METADATA = 4 * 1024 * 1024

_POLICIES = ("preview", "stable-release", "canary-release", "source-branch")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


class ChannelError(ValueError):
    """Invalid, unavailable, or untrusted channel metadata (never a fallback)."""


class ChannelNotFound(ChannelError):
    pass


def _match(pattern: str, value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(pattern, value, re.ASCII) is None:
        raise ChannelError(f"Invalid {label}")
    return value


def validate_name(value: object) -> str:
    name = _match(r"[a-z0-9]+(?:-[a-z0-9]+)*", value, "channel name")
    if len(name) > 32 or name in _RESERVED:
        raise ChannelError("Invalid channel name")
    return name


def validate_repository(value: object) -> str:
    return _match(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value, "repository")


def require_sha256(value: object) -> str:
    return _match(r"[a-f0-9]{64}", value, "SHA256")


def require_commit(value: object) -> str:
    return _match(r"[a-f0-9]{40}", value, "commit")


def build_prefix(build_id: object) -> str:
    return "releases/channel-builds/" + _match(r"[a-f0-9]{32}", build_id, "build ID") + "/"


def channel_key(name: object) -> str:
    return f"releases/channels/{validate_name(name)}.json"


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def decode_json(body: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ChannelError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(body, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise ChannelError("Invalid channel JSON") from exc
    if not isinstance(value, dict):
        raise ChannelError("Channel metadata must be an object")
    return value


def public_base(value: str) -> str:
    parsed = urlsplit(value)
    if (not parsed.hostname or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(c in value for c in "%\\;")
            or any(ord(c) <= 32 for c in value)
            or any(p in (".", "..") for p in parsed.path.split("/"))
            or not (parsed.scheme == "https" or (parsed.scheme == "http"
                    and parsed.hostname in ("127.0.0.1", "localhost", "::1")))):
        raise ChannelError("Invalid channel archive authority")
    return value.rstrip("/")


def artifact_key(value: object) -> str:
    key = _match(r"[A-Za-z0-9_./+-]+", value, "artifact key")
    if not key.startswith("releases/") or any(p in ("", ".", "..") for p in key.split("/")):
        raise ChannelError("Invalid artifact key")
    if any(p.split(".")[0].lower() in _RESERVED or p.endswith(".") for p in key.split("/")):
        raise ChannelError("Invalid artifact key")
    return key


def package_versions(sequence: int) -> tuple[str, str]:
    _integer(sequence, "sequence", 1, MAX_SEQUENCE)
    return f"0.0.{sequence}", f"0.{sequence // 65536}.{sequence % 65536}.0"


def _integer(value: object, label: str, minimum: int = 1, maximum: int = MAX_SEQUENCE + 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ChannelError(f"Invalid {label}")
    return value


def _schema(value: object) -> dict:
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1:
        raise ChannelError("Unsupported channel schema")
    return value


def validate_identity(value: object) -> dict:
    if not isinstance(value, dict):
        raise ChannelError("Invalid channel identity")
    _match(r"[a-f0-9]{16}", value.get("token"), "identity token")
    _match(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}", value.get("displayName"), "identity display name")
    for field in ("appNamePascal", "artifactNamePascal"):
        _match(r"[A-Za-z][A-Za-z0-9]{0,63}", value.get(field), field)
    _match(r"[a-z][a-z0-9.-]{2,127}", value.get("appId"), "app ID")
    _match(r"[A-Za-z][A-Za-z0-9.-]{2,49}", value.get("msixAppIdWithOrg"), "MSIX identity")
    _match(r"[a-z][a-z0-9-]{0,63}", value.get("cliName"), "CLI name")
    _match(r"[A-Za-z][A-Za-z0-9 ._-]{0,79}", value.get("windowsExecutableName"), "Windows executable")
    return value


def validate_request(value: object, *, repository: str | None = None,
                     base_url: str | None = None, policy: str = "preview") -> dict:
    request = _schema(value)
    build_prefix(request.get("buildId"))
    validate_name(request.get("channel"))
    actual_repo = validate_repository(request.get("repository"))
    if repository is not None and actual_repo.casefold() != repository.casefold():
        raise ChannelError("Channel repository authority mismatch")
    require_commit(request.get("commit"))
    if "controllerCommit" in request:
        require_commit(request["controllerCommit"])
    _match(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", request.get("sourceVersion"), "source version")
    versions = package_versions(request.get("sequence"))
    if policy == "preview":
        if (request.get("version"), request.get("windowsVersion")) != versions:
            raise ChannelError("Channel package version does not match sequence")
    else:
        _match(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-canary\.[0-9]+)?", request.get("releaseTag"), "legacy release tag")
        if request.get("version") != request["releaseTag"][1:]:
            raise ChannelError("Legacy package version mismatch")
        # Canary revisions encode minutes since the stable baseline; only stable
        # packages reserve revision zero.
        pattern = r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" if policy == "canary-release" else r"[0-9]+\.[0-9]+\.[0-9]+\.0"
        quad = _match(pattern, request.get("windowsVersion"), "Windows version")
        if any(int(part) > 65535 for part in quad.split(".")):
            raise ChannelError("Invalid Windows version")
    validate_identity(request.get("identity"))
    values = request.get("bundleEnv")
    if not isinstance(values, dict):
        raise ChannelError("Invalid bundle environment")
    for key, item in values.items():
        _match(r"[A-Za-z_][A-Za-z0-9_]*", key, "bundle environment name")
        if item is not None and (not isinstance(item, str) or "\0" in item):
            raise ChannelError("Invalid bundle environment value")
    base = request.get("publicBase")
    if not isinstance(base, str) or public_base(base) != base:
        raise ChannelError("Invalid request archive authority")
    if base_url is not None and base != public_base(base_url):
        raise ChannelError("Request archive authority mismatch")
    return request


def _head(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ChannelError("Invalid channel head")
    prefix = build_prefix(value.get("buildId"))
    _integer(value.get("sequence"), "head sequence", maximum=MAX_SEQUENCE)
    if value.get("manifestKey") != prefix + "build.json":
        raise ChannelError("Invalid channel manifest key")
    require_sha256(value.get("sha256"))
    return value


def validate_record(value: object, *, name: str | None = None, repository: str | None = None) -> dict:
    record = _schema(value)
    if not {"name", "repository", "policy", "state", "revision", "nextSequence", "identity", "head"} <= record.keys():
        raise ChannelError("Missing channel record fields")
    validate_name(record.get("name"))
    if name is not None and record["name"] != name:
        raise ChannelError("Channel name does not match object key")
    actual_repo = validate_repository(record.get("repository"))
    if repository is not None and actual_repo.casefold() != repository.casefold():
        raise ChannelError("Channel repository authority mismatch")
    if record.get("policy") not in _POLICIES or record.get("state") not in ("active", "retired"):
        raise ChannelError("Invalid channel policy or state")
    _integer(record.get("revision"), "channel revision", maximum=2**53 - 1)
    _integer(record.get("nextSequence"), "next sequence")
    head = _head(record.get("head"))
    if head is not None and head["sequence"] >= record["nextSequence"]:
        raise ChannelError("Channel head exceeds allocation high-water")
    if record["policy"] == "source-branch":
        delivery = record.get("delivery")
        if record.get("identity") is not None or head is not None or not isinstance(delivery, dict) or delivery.get("kind") != "source-branch":
            raise ChannelError("Source branch cannot carry native artifacts")
        branch = _match(r"[A-Za-z0-9][A-Za-z0-9_./-]*", delivery.get("branch"), "source branch")
        if ".." in branch or "//" in branch or branch.endswith(("/", ".", ".lock")):
            raise ChannelError("Invalid source branch")
    else:
        validate_identity(record.get("identity"))
    if record["state"] == "retired":
        validate_name(record.get("destination"))
        _match(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", record.get("minimumVersion"), "retirement minimum version")
        if _head(record.get("destinationHead")) is None:
            raise ChannelError("Retirement requires a pinned destination head")
        if type(record.get("receiverProtocol")) is not int or record["receiverProtocol"] != 1:
            raise ChannelError("Unsupported retirement receiver protocol")
        receiver = record.get("receiver")
        if not isinstance(receiver, dict) or receiver.get("kind") not in ("in-place", "discontinued"):
            raise ChannelError("Invalid retirement receiver kind")
        if _head(record.get("lastHead")) != head:
            raise ChannelError("Retirement last head mismatch")
    return record


@dataclass(frozen=True)
class ChannelResolution:
    requested: dict
    terminal: dict
    manifest: dict | None



class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ChannelError("Channel archive redirects are not permitted")


class ChannelReader:
    def __init__(self, base_url: str, repository: str | None = None, opener=None):
        self.base_url = public_base(base_url)
        self.repository = validate_repository(repository) if repository is not None else None
        self.opener = opener or build_opener(_NoRedirect()).open

    def read_bytes(self, key: str, sha256: str | None = None) -> bytes:
        url = self.base_url + "/" + artifact_key(key)
        if sha256 is not None:
            require_sha256(sha256)
        try:
            with self.opener(Request(url, headers={"Cache-Control": "no-cache"}), timeout=30) as response:
                if response.geturl() != url:
                    raise ChannelError("Channel archive redirects are not permitted")
                body = response.read(MAX_METADATA + 1)
        except HTTPError as exc:
            if exc.code == 404:
                raise ChannelNotFound(f"Channel object not found: {key}") from exc
            raise ChannelError(f"Channel read unavailable: HTTP {exc.code}") from exc
        except (OSError, URLError) as exc:
            raise ChannelError("Channel read unavailable") from exc
        if len(body) > MAX_METADATA:
            raise ChannelError("Channel metadata exceeds size limit")
        if sha256 is not None and hashlib.sha256(body).hexdigest() != sha256:
            raise ChannelError("Channel metadata SHA256 mismatch")
        return body

    def read_record(self, name: str) -> dict:
        return validate_record(decode_json(self.read_bytes(channel_key(name))), name=name, repository=self.repository)

    def resolve(self, name: str) -> ChannelResolution:
        requested = record = self.read_record(name)

        if record["state"] == "retired":
            if record["destination"] == name:
                raise ChannelError("Channel retirement cycle")
            record = self.read_record(record["destination"])
            if record["repository"].casefold() != requested["repository"].casefold():
                raise ChannelError("Retirement repository authority mismatch")
            if record["state"] != "active" or record["policy"] != "stable-release":
                raise ChannelError("Retirement requires a directly active stable-release destination")
        head = record["head"]
        if requested["state"] == "retired":
            # The first receiver stays pinned even when stable advances offline.
            head = requested["destinationHead"]
            if head is None or record["head"] is None or head["sequence"] > record["head"]["sequence"]:
                raise ChannelError("Invalid retirement destination head")
        manifest = None if head is None else validate_manifest(
            decode_json(self.read_bytes(head["manifestKey"], head["sha256"])), {**record, "head": head}, self.base_url)
        if requested["state"] == "retired":
            assert manifest is not None
            if manifest.get("receiverProtocol") != requested["receiverProtocol"]:
                raise ChannelError("Stable build has no supported retirement receiver")
            actual = tuple(map(int, manifest["request"]["sourceVersion"].split(".")))
            floor = tuple(map(int, requested["minimumVersion"].split(".")))
            if actual < floor:
                raise ChannelError("Destination does not meet retirement minimum version")
        return ChannelResolution(requested, record, manifest)


def validate_manifest(value: object, record: dict, base_url: str) -> dict:
    manifest = _schema(value)
    if "receiverProtocol" in manifest:
        _integer(manifest["receiverProtocol"], "receiver protocol", maximum=2**53 - 1)
    request = validate_request(manifest.get("request"), repository=record["repository"],
                               base_url=base_url, policy=record["policy"])
    if request["identity"] != record["identity"]:
        raise ChannelError("Manifest channel identity mismatch")
    if request["channel"] != record["name"]:
        raise ChannelError("Manifest channel mismatch")
    head = record.get("head")
    if head is not None and any(request[field] != head[field] for field in ("buildId", "sequence")):
        raise ChannelError("Manifest does not match channel head")
    packages = manifest.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ChannelError("Build manifest has no packages")
    prefixes = [build_prefix(request["buildId"])]
    if record["policy"] in {"stable-release", "canary-release"}:
        prefixes.append(f"releases/tag/{request['releaseTag']}/")
    seen = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ChannelError("Invalid native package")
        target = tuple(package.get(field) for field in ("platform", "arch", "variant"))
        if target[0] not in ("darwin", "win32") or target[1] not in ("arm64", "x64") or target[2] != "bundled":
            raise ChannelError("Unsupported channel package target")
        if target in seen:
            raise ChannelError("Duplicate package target")
        seen.add(target)
        artifact = package.get("artifact")
        if not isinstance(artifact, dict):
            raise ChannelError("Invalid package artifact")
        key = artifact_key(artifact.get("key"))
        if not any(key.startswith(prefix) for prefix in prefixes):
            raise ChannelError("Artifact outside admitted build namespace")
        require_sha256(artifact.get("sha256"))
        _integer(artifact.get("size"), "artifact size", maximum=2**53 - 1)
        identity_field, version_field, signing_field = (
            ("appId", "version", "teamId") if target[0] == "darwin"
            else ("msixAppIdWithOrg", "windowsVersion", "publisher"))
        if package.get("identity") != request["identity"][identity_field] or package.get("version") != request[version_field]:
            raise ChannelError("Native package identity or version mismatch")
        signing = package.get(signing_field)
        if not isinstance(signing, str) or not signing or any(ord(c) < 32 for c in signing):
            raise ChannelError("Missing native signing identity")
        if target[0] == "darwin":
            _match(r"[A-Z0-9]{10}", signing, "Apple team ID")
        feed = package.get("feed")
        if not isinstance(feed, dict) or feed.get("channel") != "stable":
            raise ChannelError("Channel native metadata requires a neutral stable feed")
        feed_key = artifact_key(feed.get("key"))
        if not any(feed_key.startswith(prefix) for prefix in prefixes):
            raise ChannelError("Native metadata outside admitted build namespace")
    return manifest
