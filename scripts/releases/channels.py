"""Authenticated channel administration; mutable records use ETag CAS, never CDN state."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import http.client
import secrets
import uuid

from hermes_cli.release_channels import (
    ChannelError, ChannelNotFound, ChannelReader, artifact_key, build_prefix,
    canonical_json, channel_key, decode_json, package_versions, public_base as validate_public_base,
    validate_name, validate_record, validate_repository, validate_request, validate_manifest, require_sha256,
)
from scripts.releases import r2


class ChannelConflict(ChannelError):
    pass


class PublicVisibilityError(ChannelError):
    """The authenticated write committed; do not roll it back on a stale CDN read."""


class R2ChannelStore:
    def __init__(self, creds: dict, base: str, bucket: str, *, scope=None):
        from scripts.releases.r2_scope import R2Scope
        self.creds, self.base, self.bucket = creds, base.rstrip("/"), bucket
        self.scope = R2Scope.configured() if scope is None else scope

    def get(self, key: str) -> tuple[bytes, str] | None:
        url = self.scope.object_url(self.base, self.bucket, artifact_key(key))
        try:
            response = r2.signed_request("GET", url, creds=self.creds, now=r2.amz_timestamp())
        except r2.R2RequestError as exc:
            if exc.status == 404:
                return None
            raise
        etag = response.header("etag")
        if not etag:
            raise ChannelError("Authenticated channel read missing ETag")
        body = response._body
        if len(body) > 4 * 1024 * 1024:
            raise ChannelError("Channel metadata exceeds size limit")
        return body, etag

    def put(self, key: str, body: bytes, etag: str | None = None) -> None:
        url = self.scope.object_url(self.base, self.bucket, artifact_key(key))
        condition = {"If-Match": etag} if etag is not None else {"If-None-Match": "*"}
        condition["Cache-Control"] = r2.cache_control_for(key) or "no-store"
        try:
            r2.signed_request("PUT", url, body=body, body_hash=hashlib.sha256(body).hexdigest(),
                              content_length=len(body), content_type="application/json",
                              creds=self.creds, now=r2.amz_timestamp(), extra_headers=condition, tries=1)
        except (r2.R2RequestError, OSError, http.client.HTTPException) as exc:
            # Lost success responses and conditional retries are settled by exact bytes.
            current = self.get(key)
            if current is not None and current[0] == body:
                return
            if isinstance(exc, r2.R2RequestError) and exc.status == 412:
                raise ChannelConflict(f"Channel write conflict: {key}") from exc
            raise ChannelError(f"Channel write outcome uncertain: {key}; inspect before retry") from exc
        current = self.get(key)
        if current is None or current[0] != body:
            raise ChannelConflict(f"Channel changed before authenticated readback: {key}")

    def keys(self, prefix: str) -> list[str]:
        keys, seen = [], set()
        prefix = self.scope.listing_prefix(prefix)
        token = None
        while True:
            params = {"list-type": "2", "max-keys": "1000", "prefix": prefix}
            if token is not None:
                params["continuation-token"] = token
            url = f"{self.scope.bucket_url(self.base, self.bucket)}?{r2.canonical_query(params)}"
            response = r2.signed_request("GET", url, creds=self.creds, now=r2.amz_timestamp())
            page = r2.parse_list_xml(response.text())
            keys.extend(self.scope.logical_key(key) for key in page["keys"])
            if not page["truncated"]:
                return keys
            token = page["nextToken"]
            if not token or token in seen:
                raise ChannelError("Incomplete or repeated channel pagination token")
            seen.add(token)


def preview_identity(name: str, token: str) -> dict:
    validate_name(name)
    pascal = f"HermesChannel{token}"
    return {"token": token, "displayName": f"Hermes {name}",
            "appId": f"ai.hermes.channel.h{token}", "appNamePascal": pascal,
            "artifactNamePascal": pascal, "cliName": f"hermes-{name}",
            "windowsExecutableName": pascal, "msixAppIdWithOrg": f"NousResearch.{pascal}"}


class ChannelPublisher:
    def __init__(self, store: R2ChannelStore, repository: str, public_base: str,
                 authorize, verify_build=None):
        self.store = store
        self.repository = validate_repository(repository)
        self.public_base = validate_public_base(public_base)
        self.reader = ChannelReader(self.public_base, self.repository)
        self.authorize = authorize
        self.verify_build = verify_build


    def _read(self, name: str) -> tuple[dict, str] | None:
        value = self.store.get(channel_key(name))
        if value is None:
            return None
        body, etag = value
        return validate_record(decode_json(body), name=name, repository=self.repository), etag

    def _write(self, key: str, value: dict, etag: str | None = None) -> None:
        body = canonical_json(value)
        self.store.put(key, body, etag)
        try:
            visible = self.reader.read_bytes(key)
        except ChannelError as exc:
            raise PublicVisibilityError(f"Committed {key}; public visibility verification failed") from exc
        if visible != body:
            current = self.store.get(key)
            if current is not None and current[0] != body:
                # A real subsequent CAS is not a stale CDN response. Allocation may
                # retry (leaving a gap); retirement must re-qualify the changed head.
                raise ChannelConflict(f"Committed {key}; superseded before public readback")
            raise PublicVisibilityError(f"Committed {key}; public visibility verification failed")

    def _preview(self, action: str, record: dict) -> None:
        if record["policy"] != "preview":
            raise ChannelError("Protected release policy requires its existing release gate")
        if record["state"] != "active":
            raise ChannelError("Retired channel is permanently closed to publication")
        self.authorize(action, record)

    def create(self, name: str) -> dict:
        validate_name(name)
        existing = self._read(name)
        if existing:
            self._preview("create", existing[0])
            return existing[0]
        record = {"schema": 1, "name": name, "repository": self.repository,
                  "policy": "preview", "state": "active", "revision": 1,
                  "nextSequence": 1, "head": None}
        self.authorize("create", record)
        for _ in range(16):
            token = secrets.token_hex(8)
            try:
                self.store.put(f"releases/channel-identities/{token}.json",
                               canonical_json({"schema": 1, "repository": self.repository, "channel": name}))
                record["identity"] = preview_identity(name, token)
                break
            except ChannelConflict:
                continue
        else:
            raise ChannelConflict("Could not reserve a channel identity")
        validate_record(record)
        try:
            self._write(channel_key(name), record)
        except ChannelConflict:
            winner = self._read(name)
            if winner is None:
                raise
            self._preview("create", winner[0])
            return winner[0]
        return record

    def allocate(self, name: str, commit: str, source_version: str,
                 bundle_env: dict | None = None, controller_commit: str | None = None) -> dict:
        from scripts.releases.bundle_env import validate
        bundle_env = validate({} if bundle_env is None else bundle_env)
        for _ in range(16):
            current = self._read(name)
            if current is None:
                raise ChannelNotFound(f"Channel not found: {name}")
            record, etag = current
            self._preview("allocate", record)
            sequence = record["nextSequence"]
            version, windows_version = package_versions(sequence)
            request = {"schema": 1, "buildId": uuid.uuid4().hex, "channel": name,
                       "sequence": sequence, "repository": self.repository, "commit": commit,
                       "sourceVersion": source_version, "version": version, "windowsVersion": windows_version,
                       "identity": deepcopy(record["identity"]), "bundleEnv": bundle_env,
                       "publicBase": self.public_base}
            if controller_commit is not None:
                request["controllerCommit"] = controller_commit
            validate_request(request, repository=self.repository, base_url=self.public_base)
            # Distinguish competing allocations with otherwise identical CAS bodies.
            updated = {**record, "revision": record["revision"] + 1, "nextSequence": sequence + 1,
                       "lastAllocation": {"buildId": request["buildId"], "sequence": sequence}}
            try:
                self._write(channel_key(name), updated, etag)
            except ChannelConflict:
                continue
            self._write(build_prefix(request["buildId"]) + "request.json", request)
            return request
        raise ChannelConflict("Channel allocation remained contended")

    def _protected(self, action: str, record: dict, request: dict, policy: str, release_gate) -> None:
        if policy not in {"stable-release", "canary-release"} or record["policy"] != policy:
            raise ChannelError("Protected release policy mismatch")
        if record["state"] != "active":
            raise ChannelError("Retired channel is permanently closed to publication")
        from scripts.releases.semver import is_valid_version
        version = request["version"]
        if not is_valid_version(version) or ("-canary." in version) != (policy == "canary-release"):
            raise ChannelError("Release version does not match protected policy")
        self.authorize(action, record)
        if release_gate(request) is not True:
            raise ChannelError("Protected publication requires the successful release gate")

    def allocate_protected(self, name: str, commit: str, source_version: str, *,
                           release_tag: str, version: str, windows_version: str,
                           identity: dict, policy: str, release_gate) -> dict:
        """Reserve accepted legacy bytes, never authorize a custom build as stable."""
        facts = {"schema": 1, "channel": name, "repository": self.repository, "commit": commit,
                 "sourceVersion": source_version, "version": version, "windowsVersion": windows_version,
                 "releaseTag": release_tag, "identity": deepcopy(identity), "bundleEnv": {},
                 "publicBase": self.public_base}
        build_id = hashlib.sha256(canonical_json(facts)).hexdigest()[:32]
        key = build_prefix(build_id) + "request.json"
        for _ in range(16):
            current = self._read(name)
            record, etag = current if current else ({
                "schema": 1, "name": name, "repository": self.repository, "policy": policy,
                "state": "active", "revision": 1, "nextSequence": 1, "head": None,
                "identity": deepcopy(identity)}, None)
            validate_record(record, repository=self.repository)
            request = {**facts, "buildId": build_id, "sequence": record["nextSequence"]}
            validate_request(request, repository=self.repository, base_url=self.public_base, policy=policy)
            self._protected("allocate-protected", record, request, policy, release_gate)
            if record["identity"] != identity:
                raise ChannelError("Protected release identity mismatch")
            existing = self.store.get(key)
            if existing is not None:
                pinned = self.request(build_id)
                if {k: v for k, v in pinned.items() if k != "sequence"} != {k: v for k, v in request.items() if k != "sequence"}:
                    raise ChannelError("Protected immutable request differs from accepted release")
                return pinned
            # Retain the request in the CAS reservation so a lost request PUT can
            # resume with its original sequence, including after process restart.
            last = record.get("lastAllocation", {})
            if last.get("buildId") == build_id and last.get("request"):
                request = last["request"]
            else:
                if last.get("request"):
                    # Flush a prior interrupted reservation before replacing its
                    # recovery slot. Otherwise retries would allocate new inputs.
                    prior_key = build_prefix(last["buildId"]) + "request.json"
                    self._write(prior_key, last["request"])
                updated = {**record, "revision": record["revision"] + 1,
                           "nextSequence": request["sequence"] + 1,
                           "lastAllocation": {"buildId": build_id, "sequence": request["sequence"],
                                              "request": request}}
                try:
                    self._write(channel_key(name), updated, etag)
                except ChannelConflict:
                    continue
            try:
                self._write(key, request)
            except ChannelConflict:
                # Identical concurrent releases adopt the first immutable request;
                # their other reservation is only an unused sequence gap.
                continue
            return request
        raise ChannelConflict("Protected release allocation remained contended")

    def request(self, build_id: str, sha256: str | None = None) -> dict:
        if sha256 is not None:
            require_sha256(sha256)
        key = build_prefix(build_id) + "request.json"
        found = self.store.get(key)
        if found is None:
            raise ChannelNotFound(f"Request not found: {build_id}")
        body = found[0]
        if sha256 is not None and hashlib.sha256(body).hexdigest() != sha256:
            raise ChannelError("Request SHA256 mismatch")
        decoded = decode_json(body)
        current = self._read(validate_name(decoded.get("channel")))
        if current is None:
            raise ChannelNotFound("Request channel not found")
        request = validate_request(decoded, repository=self.repository, base_url=self.public_base,
                                   policy=current[0]["policy"])
        if request["buildId"] != build_id:
            raise ChannelError("Request build ID mismatch")
        return request

    def list(self) -> list[dict]:
        result = []
        for key in self.store.keys("releases/channels/"):
            name = key.removeprefix("releases/channels/").removesuffix(".json")
            if channel_key(name) != key:
                raise ChannelError("Invalid object in channel namespace")
            current = self._read(name)
            if current is None:
                raise ChannelError("Listed channel disappeared")
            result.append(current[0])
        return result

    def bootstrap(self, record: dict, manifest: dict | None = None, *, publish: bool = False) -> dict:
        record = deepcopy(validate_record(record, repository=self.repository))
        if record["policy"] not in {"stable-release", "canary-release", "source-branch"} or record["state"] != "active":
            raise ChannelError("Bootstrap requires an active protected release policy")
        self.authorize("bootstrap", record)
        if record["policy"] == "source-branch":
            if manifest is not None:
                raise ChannelError("Source branch bootstrap has no artifact manifest")
        else:
            if record["head"] is None or manifest is None:
                raise ChannelError("Bootstrap requires actual accepted release metadata")
            validate_manifest(manifest, record, self.public_base)
            if hashlib.sha256(canonical_json(manifest)).hexdigest() != record["head"]["sha256"]:
                raise ChannelError("Bootstrap manifest SHA256 mismatch")
            if self.verify_build is None or self.verify_build(manifest["request"], manifest) is not True:
                raise ChannelError("Bootstrap requires release qualification")
        if publish:
            if manifest is not None:
                self._write(build_prefix(manifest["request"]["buildId"]) + "request.json", manifest["request"])
                self._write(record["head"]["manifestKey"], manifest)
            self._write(channel_key(record["name"]), record)
        return record

    def promote(self, build_id: str) -> dict:
        return self._promote(build_id)

    def promote_protected(self, build_id: str, *, policy: str, release_gate) -> dict:
        return self._promote(build_id, policy=policy, release_gate=release_gate)

    def _promote(self, build_id: str, *, policy: str | None = None, release_gate=None) -> dict:
        request = self.request(build_id)
        key = build_prefix(build_id) + "build.json"
        found = self.store.get(key)
        if found is None:
            raise ChannelNotFound("Complete build manifest not found")
        raw = found[0]
        manifest = decode_json(raw)
        if manifest.get("request") != request:
            raise ChannelError("Build manifest differs from immutable admitted request")
        head = {"buildId": build_id, "sequence": request["sequence"], "manifestKey": key,
                "sha256": hashlib.sha256(raw).hexdigest()}
        for _ in range(16):
            current = self._read(request["channel"])
            if current is None:
                raise ChannelNotFound("Publication channel not found")
            record, etag = current
            if policy is None:
                self._preview("promote", record)
            else:
                self._protected("promote-protected", record, request, policy, release_gate)
            if record["identity"] != request["identity"] or request["sequence"] >= record["nextSequence"]:
                raise ChannelError("Request channel identity or allocation mismatch")
            candidate = {**record, "head": head, "revision": record["revision"] + 1}
            validate_manifest(manifest, candidate, self.public_base)
            if record["head"] is not None and record["head"]["sequence"] >= request["sequence"]:
                if record["head"] == head:
                    self.reader.read_bytes(key, head["sha256"])
                    return record
                raise ChannelError("A newer build is published; stale completion refused")
            if policy is not None and record["head"] is not None:
                from scripts.releases.semver import compare
                from scripts.releases.stable import windows_version
                previous_head = record["head"]
                previous = self.store.get(previous_head["manifestKey"])
                if previous is None or hashlib.sha256(previous[0]).hexdigest() != previous_head["sha256"]:
                    raise ChannelError("Previous protected manifest is unavailable or changed")
                previous_request = validate_manifest(decode_json(previous[0]), record, self.public_base)["request"]
                if (compare(request["version"], previous_request["version"]) <= 0
                        or windows_version(request["windowsVersion"]) <= windows_version(previous_request["windowsVersion"])):
                    raise ChannelError("Protected native versions must increase; stale completion refused")
            if self.verify_build is None or self.verify_build(request, manifest) is not True:
                raise ChannelError("Publication requires complete native build qualification")
            self.reader.read_bytes(key, head["sha256"])
            try:
                self._write(channel_key(record["name"]), candidate, etag)
            except ChannelConflict:
                continue
            return candidate
        raise ChannelConflict("Channel promotion remained contended")

    def retire(self, name: str, destination: str, minimum_version: str,
               *, publish: bool = True) -> dict:
        validate_name(destination)
        if name == destination:
            raise ChannelError("Channel retirement cycle")
        current = self._read(name)
        if current is None:
            raise ChannelNotFound(f"Channel not found: {name}")
        record, etag = current
        self._preview("retire", record)
        target_read = self._read(destination)
        if target_read is None:
            raise ChannelNotFound("Retirement destination not found")
        target, target_etag = target_read
        # Do not discard incomparable intermediate identity floors.
        if target["state"] != "active" or target["policy"] != "stable-release" or target["head"] is None:
            raise ChannelError("Retirement requires a directly active stable-release destination")
        target_raw = self.reader.read_bytes(target["head"]["manifestKey"], target["head"]["sha256"])
        manifest = validate_manifest(decode_json(target_raw), target, self.public_base)
        # Two-tier retirement: the kind is derived, never asserted by the caller.
        # A suffixed (canary/commit) identity can never match stable's, so
        # "in-place" cannot be mis-assigned; the kind is a client routing hint.
        identity = record.get("identity")
        kind = "in-place" if identity is not None and identity == target.get("identity") else "discontinued"
        retired = {**record, "state": "retired", "revision": record["revision"] + 1,
                   "destination": destination, "minimumVersion": minimum_version,
                   "destinationHead": target["head"], "receiverProtocol": 1,
                   "receiver": {"kind": kind},
                   "lastHead": record["head"]}
        validate_record(retired)
        if tuple(map(int, manifest["request"]["sourceVersion"].split("."))) < tuple(map(int, minimum_version.split("."))):
            raise ChannelError("Destination does not meet minimum version")
        if manifest.get("receiverProtocol") != 1:
            raise ChannelError("Stable release does not declare retirement receiver support")
        if self._read(destination) != (target, target_etag):
            raise ChannelConflict("Retirement destination changed; review the new target before retrying")
        if publish:
            self._write(channel_key(name), retired, etag)
        return retired
