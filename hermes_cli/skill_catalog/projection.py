"""Disposable live coordination, never installation or recovery authority.

Only OS leases prove liveness. All filesystem admission, including the empty
profile branch, holds the mutation gate and writer-owned exclusion chain.
"""
import copy
import hashlib
import json
import os
import stat
import tempfile
import time
import uuid
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

import psutil

from hermes_cli.skill_catalog.identity import source_id
from hermes_cli.skill_catalog.locks import FileLease, writer_hold
from hermes_cli.skill_catalog import validation as v
from hermes_cli.skill_catalog.validation import CatalogError


def _unavailable():
    return CatalogError("ownership_state_unavailable", "Catalog coordination is unavailable.")


def _conflict():
    return CatalogError("ownership_conflict", "Catalog ownership changed or overlaps another operation.")


def _empty() -> dict:
    return dict(schema_version=1, revision=0, registrations={}, targets={}, reservations={}, path_revisions={})


def _overlap(left, right):
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _target(home, relative):
    try:
        v._path(relative)
        current = home / "skills"
        for part in (None, *relative.split("/")):
            if part is not None:
                v._require(not part.startswith("."))
                current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                continue
            v._require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400)
        return current
    except (ValueError, OSError, UnicodeError):
        raise _conflict() from None


def _process():
    try:
        addresses = sorted(address.address for rows in psutil.net_if_addrs().values() for address in rows)
        machine = hashlib.sha256(json.dumps([psutil.boot_time(), addresses]).encode()).hexdigest()
        return dict(machine_id=machine, pid=os.getpid(), birth_id=str(psutil.Process().create_time()))
    except (OSError, psutil.Error):
        raise _unavailable() from None


def _registration(value):
    process = lambda x: v._object(x, dict(machine_id=lambda y: v._text(y, 256, 1),
        pid=lambda y: v._integer(y, 1, 2147483647), birth_id=lambda y: v._text(y, 256, 1)), strict=True)
    row = v._object(value, dict(source_id=lambda x: v._pattern(x, v._SOURCE), plugin_key=v._plugin_key,
        generation=v._token, implementation_revision=v._revision, process=process,
        state=v._enum("reconciling|ready|fenced")), strict=True)
    v._require(row["source_id"] == source_id(row["plugin_key"], row["source_id"].rsplit(".", 1)[1]))
    return row


def _claim(value):
    return v._object(value, dict(registration_id=v._token, identifier=v._reference,
        installation_revision=v._revision, path_revision=v._integer,
        managed_state=v._enum("installed|pending")), strict=True)


def _reservation(value):
    return v._object(value, dict(registration_id=v._token, generation=v._token, review_id=v._token,
        identifier=v._reference, target_relative_path=v._path, operation_id=v._nullable(v._token),
        expected_path_revision=v._integer, expected_installation_revision=v._nullable(v._revision), action=v._action,
        local_result=v._nullable(lambda x: v._operation(x, None)),
        state=v._enum("reserved|bound|unknown")), strict=True)


def _map(value, key_check, row_check):
    v._require(type(value) is dict)
    return {key_check(key): row_check(row) for key, row in value.items()}


def _stored_path(value):
    v._path(value)
    v._require(not any(part.startswith(".") for part in value.split("/")))
    return value


def _validate(doc):
    try:
        v._plain(doc, set())
        result = v._object(doc, dict(schema_version=lambda x: v._integer(x, 1, 1), revision=v._integer,
            registrations=lambda x: _map(x, v._token, _registration),
            targets=lambda x: _map(x, _stored_path, _claim),
            reservations=lambda x: _map(x, v._token, _reservation),
            path_revisions=lambda x: _map(x, _stored_path, v._integer)), strict=True)
        _relationships(result)
        return result
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        raise _unavailable() from None


def _relationships(doc):
    sources = [row["source_id"] for row in doc["registrations"].values()]
    v._require(len(set(sources)) == len(sources))
    _disjoint(doc["targets"])
    for path, row in doc["targets"].items():
        owner = doc["registrations"][row["registration_id"]]
        v._require(row["identifier"].split(":")[0] == owner["source_id"])
        v._require(doc["path_revisions"][path] == row["path_revision"])
    reviews = set()
    reserved_paths = []
    for row in doc["reservations"].values():
        owner = doc["registrations"][row["registration_id"]]
        v._require(row["generation"] == owner["generation"])
        v._require(row["identifier"].split(":")[0] == owner["source_id"])
        v._require(doc["path_revisions"].get(row["target_relative_path"], 0) == row["expected_path_revision"])
        v._require(row["state"] != "bound" or row["operation_id"] is not None)
        key = (row["registration_id"], row["review_id"])
        v._require(key not in reviews)
        reviews.add(key)
        path = row["target_relative_path"]
        reserved_paths.append(path)
        for current, claim in doc["targets"].items():
            if _overlap(path, current):
                expected_revision = (row["local_result"]["installation_revision"] if row["local_result"]
                                     else row["expected_installation_revision"])
                v._require(current == path and claim["registration_id"] == row["registration_id"]
                           and claim["identifier"] == row["identifier"]
                           and claim["installation_revision"] == expected_revision)
    _disjoint(reserved_paths)
    if not doc["registrations"]:
        v._require(not any(doc[key] for key in ("targets", "reservations", "path_revisions")))


def _disjoint(paths):
    seen = set()
    for path in sorted(paths):
        parts = path.split("/")
        v._require(not any("/".join(parts[:depth]) in seen for depth in range(1, len(parts) + 1)))
        seen.add(path)


def _drop(doc, registration_id):
    doc["registrations"].pop(registration_id, None)
    for key in ("targets", "reservations"):
        doc[key] = {path: row for path, row in doc[key].items() if row["registration_id"] != registration_id}
    if not doc["registrations"]:
        doc["path_revisions"].clear()


def _admission(reservation_id, row):
    return {key: row[key] for key in ("registration_id", "generation", "review_id",
        "expected_path_revision", "expected_installation_revision")} | {"reservation_id": reservation_id}


class _Handle:
    def __init__(self, projection, entry, leases):
        self.projection, self.entry, self.leases = projection, entry, leases

    def close(self):
        if self.leases is not None:
            self.entry.stopped = True
            self.projection.stop(self.entry)


class Projection:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.directory = self.home / "cache/skill-catalog/live-v1"
        self._handles = {}

    def _absent(self):
        try:
            with FileLease(self.directory / "presence.lock", deadline=time.monotonic()):
                return True
        except CatalogError as exc:
            if exc.code != "busy":
                raise
            return False

    def _load(self):
        if self._absent():
            return _empty()
        try:
            path = self.directory / "claims.json"
            if path.is_symlink():
                raise _unavailable()
            doc = _validate(json.loads(path.read_bytes(), object_pairs_hook=_unique_object))
        except (OSError, ValueError, UnicodeError):
            raise _unavailable() from None
        return doc

    def _prune(self, doc):
        live = 0
        directory = self.directory / "registrations"
        try:
            names = {path.name for path in directory.iterdir()}
        except FileNotFoundError:
            names = set()
        except OSError:
            raise _unavailable() from None
        for registration_id in names | set(doc["registrations"]):
            try:
                v._token(registration_id)
                with FileLease(self.directory / "registrations" / registration_id, deadline=time.monotonic()):
                    _drop(doc, registration_id)
            except CatalogError as exc:
                if exc.code != "busy":
                    raise
                if registration_id not in doc["registrations"]:
                    raise _unavailable() from None
                live += 1
            except (ValueError, UnicodeError):
                raise _unavailable() from None
        if not live and not self._absent():
            raise _unavailable()

    @contextmanager
    def transaction(self, *, deadline=None):
        with FileLease(self.directory / "mutation.lock", deadline=deadline):
            doc = self._load()
            before = copy.deepcopy(doc)
            if doc["registrations"] or not self._absent():
                self._prune(doc)
            yield doc
            _validate(doc)
            if doc != before:
                doc["revision"] = v._integer(doc["revision"] + 1)
                self.save(doc)

    def save(self, doc):
        """Caller owns the mutation gate; validation precedes any disk change."""
        data = json.dumps(_validate(doc), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        path = self.directory / "claims.json"
        temporary = backup = None
        replaced = False
        try:
            try:
                prior = path.read_bytes()
            except FileNotFoundError:
                prior = None
            temporary = _stage(self.directory, data)
            if prior is not None:
                backup = _stage(self.directory, prior)
            _sync_directory(self.directory)
            os.replace(temporary, path)
            temporary = None
            replaced = True
            _sync_directory(self.directory)
        except OSError:
            if replaced:
                if backup is None:
                    path.unlink()
                else:
                    os.replace(backup, path)
                    backup = None
            raise _unavailable() from None
        finally:
            for leftover in (temporary, backup):
                if leftover is not None:
                    with suppress(OSError):
                        os.unlink(leftover)

    def read(self):
        with self.transaction() as doc:
            return copy.deepcopy(doc)

    def start(self, entry):
        handle = None
        try:
            with self.transaction() as doc:
                if entry.stopped or not entry.authorized():
                    raise CatalogError("permission_denied", "Catalog registration is not authorized.")
                if any(row["source_id"] == entry.source_id for row in doc["registrations"].values()):
                    raise _conflict()
                leases = ExitStack()
                handle = _Handle(self, entry, leases)
                leases.enter_context(FileLease(self.directory / "presence.lock", shared=True))
                leases.enter_context(FileLease(self.directory / "registrations" / entry.registration_id))
                doc["registrations"][entry.registration_id] = dict(source_id=entry.source_id,
                    plugin_key=entry.plugin_key, generation=entry.generation,
                    implementation_revision=entry.descriptor.implementation_revision,
                    process=_process(), state="reconciling")
            self._handles[entry.registration_id] = handle
            return handle
        except BaseException:
            if handle is not None:
                handle.leases.close()
            raise

    def _owner(self, entry, doc):
        row = doc["registrations"].get(entry.registration_id)
        expected = (entry.source_id, entry.plugin_key, entry.generation, entry.descriptor.implementation_revision)
        actual = None if row is None else tuple(row[key] for key in (
            "source_id", "plugin_key", "generation", "implementation_revision"))
        if entry.stopped or row is None or actual != expected or row["state"] == "fenced":
            raise _conflict()
        if not entry.authorized():
            raise CatalogError("permission_denied", "Catalog registration is not authorized.")
        return row

    def publish(self, entry, records):
        if type(records) is not list or len(records) > 10000:
            raise CatalogError("invalid_response", "Catalog installed pagination is incomplete.")
        clean = []
        for start in range(0, len(records), 100):
            page = v.validate_result("installed", {"ok": True, "data": {
                "items": records[start:start + 100], "next_cursor": None}}, entry)
            clean.extend(page["items"])
        with self.transaction() as doc:
            owner = self._owner(entry, doc)
            targets = self._claims(entry, doc, clean)
            doc["targets"] = {path: row for path, row in doc["targets"].items()
                              if row["registration_id"] != entry.registration_id} | targets
            owner["state"] = "ready"

    def _claims(self, entry, doc, records):
        targets = {}
        for row in records:
            path = row["target_relative_path"]
            target = _target(self.home, path)
            if row["managed_state"] not in ("installed", "pending") or row["installation_revision"] is None:
                continue
            if not target.exists():
                continue
            self._overlaps(doc, path, own_target=entry.registration_id, own_reservations=True)
            if any(_overlap(path, other) for other in targets):
                raise _conflict()
            old = doc["targets"].get(path)
            same = old and all(old[key] == row[key] for key in ("identifier", "installation_revision", "managed_state"))
            revision = doc["path_revisions"].get(path, 0) + (0 if same else 1)
            doc["path_revisions"][path] = revision
            targets[path] = dict(registration_id=entry.registration_id, identifier=row["identifier"],
                installation_revision=row["installation_revision"], path_revision=revision,
                managed_state=row["managed_state"])
        return targets

    def stop(self, entry, *, deadline=None):
        deadline = time.monotonic() + 3 if deadline is None else deadline
        with FileLease(self.directory / "mutation.lock", deadline=deadline):
            doc = self._load()
            self._prune(doc)
            row = doc["registrations"].get(entry.registration_id)
            if row is not None:
                row["state"] = "fenced"
            with writer_hold(self.home, deadline=deadline, root_exclusive=True):
                _drop(doc, entry.registration_id)
                doc["revision"] += 1
                self.save(doc)
                handle = self._handles.pop(entry.registration_id, None)
                if handle is not None and handle.leases is not None:
                    handle.leases.close()
                    handle.leases = None

    def _overlaps(self, doc, path, *, own_target=None, own_reservations=False):
        for current, row in doc["targets"].items():
            if _overlap(path, current) and not (current == path and row["registration_id"] == own_target):
                raise _conflict()
        if any(_overlap(path, row["target_relative_path"])
               and not (own_reservations and row["target_relative_path"] == path
                        and row["registration_id"] == own_target)
               for row in doc["reservations"].values()):
            raise _conflict()

    def _prestate(self, entry, doc, review):
        path = review["target_relative_path"]
        target = _target(self.home, path)
        current = doc["targets"].get(path)
        if review["action"] == "install":
            if target.exists() or current is not None:
                raise CatalogError("review_changed", "Catalog target is no longer absent.")
            return
        if not target.exists() or current is None:
            raise CatalogError("review_changed", "Catalog installation changed.")
        expected = (entry.registration_id, review["identifier"], review["installation_revision"])
        if tuple(current[key] for key in ("registration_id", "identifier", "installation_revision")) != expected:
            raise CatalogError("review_changed", "Catalog installation changed.")

    def reserve(self, entry, review, *, operation_id=None):
        review = v.validate_result("review", {"ok": True, "data": review}, entry)
        if not review["confirmable"]:
            raise CatalogError("review_required", "A confirmable review is required.")
        if operation_id is not None:
            _valid_token(operation_id)
        with self.transaction() as doc:
            self._owner(entry, doc)
            if any(row["state"] != "ready" for row in doc["registrations"].values()):
                raise CatalogError("busy", "Catalog installed ownership is reconciling.")
            existing = self._existing(entry, doc, review, operation_id)
            if existing is not None:
                return existing
            if review['expires_at_ms'] <= int(time.time() * 1000):
                raise CatalogError('review_expired', 'Review expired. Review again.')
            self._overlaps(doc, review["target_relative_path"], own_target=entry.registration_id)
            self._prestate(entry, doc, review)
            reservation_id = uuid.uuid4().hex
            row = dict(registration_id=entry.registration_id, generation=entry.generation,
                review_id=review["review_id"], identifier=review["identifier"],
                target_relative_path=review["target_relative_path"], operation_id=operation_id,
                expected_path_revision=doc["path_revisions"].get(review["target_relative_path"], 0),
                expected_installation_revision=review["installation_revision"], action=review["action"],
                local_result=None, state="reserved" if operation_id is None else "bound")
            doc["reservations"][reservation_id] = row
            return _admission(reservation_id, row)

    def _existing(self, entry, doc, review, operation_id):
        for reservation_id, row in doc["reservations"].items():
            if (row["registration_id"], row["review_id"]) != (entry.registration_id, review["review_id"]):
                continue
            pairs = {"generation": entry.generation, "identifier": review["identifier"],
                "target_relative_path": review["target_relative_path"],
                "expected_installation_revision": review["installation_revision"]}
            if any(row[key] != value for key, value in pairs.items()):
                raise _conflict()
            if operation_id is not None and row["operation_id"] != operation_id:
                raise _conflict()
            return _admission(reservation_id, row)
        return None

    def binding(self, entry, doc, admission, operation_id):
        self._owner(entry, doc)
        try:
            admission = v._admission(admission)
            v._token(operation_id)
            row = doc["reservations"][admission["reservation_id"]]
            v._require(admission == _admission(admission["reservation_id"], row))
            v._require(row["registration_id"] == entry.registration_id and row["generation"] == entry.generation)
            v._require(row["operation_id"] == operation_id and row["state"] in ("bound", "unknown"))
            v._require(doc["path_revisions"].get(row["target_relative_path"], 0) == row["expected_path_revision"])
            return row
        except (ValueError, KeyError, TypeError):
            raise _conflict() from None

    def bind(self, entry, payload):
        try:
            payload = v._object(payload, dict(registration_id=v._token, review_id=v._token,
                reservation_id=v._token, operation_id=v._token, expected_path_revision=v._integer,
                expected_installation_revision=v._nullable(v._revision)), strict=True)
        except (ValueError, TypeError, UnicodeError):
            raise CatalogError("invalid_request", "Invalid catalog operation binding.") from None
        admission = {key: value for key, value in payload.items() if key != "operation_id"}
        admission["generation"] = entry.generation
        with self.transaction() as doc:
            self._owner(entry, doc)
            row = doc["reservations"].get(payload["reservation_id"])
            if row is None or _admission(payload["reservation_id"], row) != admission:
                raise _conflict()
            if row["operation_id"] not in (None, payload["operation_id"]):
                raise _conflict()
            if row["operation_id"] is None:
                row.update(operation_id=payload["operation_id"], state="bound")
            self.binding(entry, doc, admission, payload["operation_id"])
            return dict(admission, operation_id=payload["operation_id"])

    @contextmanager
    def generic(self, targets, *, deadline=None):
        targets = tuple(targets)
        deadline = time.monotonic() + 3 if deadline is None else deadline
        with self.transaction(deadline=deadline) as doc:
            if any(row["state"] != "ready" for row in doc["registrations"].values()):
                raise CatalogError("busy", "Catalog installed ownership is reconciling.")
            for path in targets:
                _target(self.home, path)
                self._overlaps(doc, path)
            with writer_hold(self.home, targets, deadline=deadline):
                for path in targets:
                    _target(self.home, path)
                yield


def _valid_token(value):
    try:
        return v._token(value)
    except (ValueError, UnicodeError):
        raise CatalogError("invalid_request", "Invalid catalog operation identifier.") from None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate projection key")
        result[key] = value
    return result


def _sync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage(directory, data):
    descriptor, path = tempfile.mkstemp(prefix=".claims-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except BaseException:
        with suppress(OSError):
            os.unlink(path)
        raise
