"""Profile-scoped model settings: typed schema, CAS store, resolver, wire map.

Generic core infrastructure for the Local Models settings surface (see
``docs/next-upgrade/MODEL-SETTINGS.md`` in the unsloth-studio-upgrade docs).
This module owns NOTHING but settings data: no app-config mutation, no routes,
no dispatcher, no ``presets.ini`` writes, no memory-policy interaction. The
existing llama.cpp preset/sampling deference ladder stays authoritative.

Owner scope contract
--------------------
``ModelSettingsStore(root)`` takes a SERVER-OWNED root — the active profile's
``get_hermes_home()`` path (or an equivalent isolated home). Callers must never
build this path from frontend input; profile isolation comes from passing the
right root, one store instance per profile.

Public API (exact signatures)
-----------------------------
``STORE_SCHEMA_VERSION: int`` — the persistence schema this build reads/writes.

Errors (all subclass ``ModelSettingsError``):
  ``InvalidSettingsKeyError``, ``UnknownSettingError``,
  ``InvalidSettingValueError``, ``UnsupportedSchemaVersionError``,
  ``SettingsCorruptionError``, ``ScopeMismatchError``,
  ``RevisionConflictError(expected_revision, current_revision)``.

``AUTO`` — typed sentinel for an engine-advertised automatic choice. It is
persisted/serialized as ``{"auto": true}`` and is distinct from ``None``,
``False``, ``""`` and ``0``. Explicit zero is always preserved.

``SettingsKey(model_fingerprint: str, engine: str, workflow: str)`` — frozen.
``model_fingerprint`` is exactly 64 lowercase hex chars; ``engine`` and
``workflow`` are simple stable ids (``^[a-z0-9][a-z0-9._-]{0,63}$``).
Methods: ``to_dict() -> dict``, ``SettingsKey.from_dict(mapping)`` classmethod.

``SettingSpec(key: str, kind: str, minimum=None, maximum=None,
             choices=None, supports_auto=False, apply="request")`` — frozen.
``kind`` is ``"int" | "float" | "bool" | "enum"``; ``apply`` is
``"request"`` (per-request sampling; no reload) or ``"load"`` (load-time;
never sent as a request field). Construction validates (``ValueError``).

``EngineSchema(engine: str, workflow: str, schema_version: int,
               settings: tuple[SettingSpec, ...])`` — frozen. Supplied by the
TRUSTED adapter, never by a renderer. Methods: ``keys()``,
``spec(key) -> SettingSpec | None``, ``validate_value(key, value) -> value``,
``validate_overrides(mapping) -> dict``.

``SettingsSnapshot`` — attributes ``key``, ``schema_version``, ``revision``
(0 when absent), ``overrides`` (immutable mapping, empty when absent);
``to_dict() -> dict`` for a JSON response.

``ModelSettingsStore(root: Path)`` — SQLite-backed, cross-process safe
(``BEGIN IMMEDIATE`` + busy timeout + compare-and-swap). Reads open the
database read-only and never create files. Methods:
  ``read(key) -> SettingsSnapshot``
  ``write(key, overrides: Mapping[str, Any], expected_revision: int)
      -> SettingsSnapshot``  (complete replacement; raises
      ``RevisionConflictError`` on a stale revision)
  ``reset(key, expected_revision: int, keys: Sequence[str] | None = None)
      -> SettingsSnapshot``  (all keys -> empty-override snapshot at a
      strictly newer revision — reset deletes the overrides, never the
      revision history; selected keys keep the rest, revision bumps;
      reset-all on an absent scope is a revision-0 no-op creating no files)

``ModelSettingsService(store: ModelSettingsStore)`` — the seam routes call:
  ``read(key) -> SettingsSnapshot``
  ``save(key, schema: EngineSchema, overrides: Mapping[str, Any],
         expected_revision: int) -> SettingsSnapshot`` — validates the whole
      payload against the adapter schema first (unknown keys and invalid
      values are rejected, not dropped), then CAS-replaces.
  ``reset(key, expected_revision: int, keys=None) -> SettingsSnapshot``
  ``resolve(key, schema: EngineSchema, *, request=None, profile=None,
            publisher=None, catalog=None, engine_reported=None)
            -> ResolutionResult``
  All layer arguments are ``Mapping[str, Any] | None``.

``ResolutionResult`` — attributes ``values`` (resolved values, only for keys
with a source; ``AUTO`` stays the sentinel) and ``sources`` (per schema key:
``"request" | "model" | "profile" | "publisher" | "catalog" | "engine" |
"inherit"``); ``get(key)``; ``to_dict()`` (JSON-safe).

Precedence: request > saved model override ("model") > profile default >
publisher/file metadata > curated catalog > reported engine default. An
unset key means inherit; a missing engine default is omitted, never invented.
The request, saved-model and profile layers are authoritative (caller-supplied
user settings): unknown keys and invalid values raise, failing the request.
Only the untrusted metadata layers (publisher, catalog, engine) tolerate junk
values, falling through to the next layer.

``OPENAI_STANDARD_FIELDS: frozenset[str]`` and
``split_request_fields(schema, resolution) -> tuple[dict, dict]`` —
returns ``(standard_fields, extra_body)``: ``apply="request"`` settings whose
key is an OpenAI chat field go to the standard dict, engine extensions go to
``extra_body`` (callers must merge these per key into any existing
``extra_body``, never blind-update). ``apply="load"`` settings are excluded;
inherited keys are omitted; an ``AUTO`` extension serializes as ``"auto"``
while an ``AUTO`` standard field is omitted for the client/server to decide.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Sequence

STORE_SCHEMA_VERSION = 1

_DB_RELATIVE = ("local_runtime", "model_settings.sqlite3")
_BUSY_TIMEOUT_MS = 10_000

_FINGERPRINT_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_SIMPLE_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,63}\Z")

_SETTING_KINDS = ("int", "float", "bool", "enum")
_APPLY_MODES = ("request", "load")

#: OpenAI chat-completions request fields; anything else the adapter
#: advertises is an engine extension that belongs in ``extra_body``.
OPENAI_STANDARD_FIELDS = frozenset({
    "frequency_penalty", "logit_bias", "logprobs", "max_tokens", "n",
    "presence_penalty", "response_format", "seed", "stop", "temperature",
    "top_logprobs", "top_p",
})


# ---------------------------------------------------------------------------
# Errors


class ModelSettingsError(Exception):
    """Base class for every model-settings domain failure."""


class InvalidSettingsKeyError(ModelSettingsError):
    """Owner-scope identity (fingerprint/engine/workflow) is malformed."""


class UnknownSettingError(ModelSettingsError):
    """A setting name is not in the trusted adapter schema."""


class InvalidSettingValueError(ModelSettingsError):
    """A value is the wrong type, non-finite, out of range, or AUTO-unsupported."""


class UnsupportedSchemaVersionError(ModelSettingsError):
    """Persisted or supplied schema version is newer than this build reads."""


class SettingsCorruptionError(ModelSettingsError):
    """The stored payload cannot be trusted: bad JSON, wrong value shapes, an
    unreadable database file, or a table layout this build does not know."""


class ScopeMismatchError(ModelSettingsError):
    """The supplied EngineSchema does not match the SettingsKey scope."""


class RevisionConflictError(ModelSettingsError):
    """A compare-and-swap write lost: someone else saved a newer revision."""

    def __init__(self, expected_revision: int, current_revision: int) -> None:
        super().__init__(
            f"settings revision conflict: expected {expected_revision}, "
            f"store holds {current_revision}"
        )
        self.expected_revision = expected_revision
        self.current_revision = current_revision


# ---------------------------------------------------------------------------
# The typed AUTO sentinel


class _AutoSentinel:
    """Singleton for the engine-advertised automatic choice.

    Deliberately distinct from ``None``/``False``/``""``/``0`` so a typed
    "auto" never collapses into unset or a disabled sentinel.
    """

    _instance: _AutoSentinel | None = None

    def __new__(cls) -> _AutoSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "AUTO"

    def __reduce__(self):  # keep the singleton across pickle/copy
        return (_AutoSentinel, ())


AUTO = _AutoSentinel()


def _encode_value(value: Any) -> Any:
    """JSON-encode one setting value, AUTO tagged as {"auto": true}."""
    if value is AUTO:
        return {"auto": True}
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidSettingValueError(f"non-finite number: {value!r}")
        return value
    raise InvalidSettingValueError(f"unsupported setting value type: {value!r}")


# ---------------------------------------------------------------------------
# Owner-scope identity


@dataclass(frozen=True)
class SettingsKey:
    """Owner scope: gateway+profile live in the store root; this is the rest."""

    model_fingerprint: str
    engine: str
    workflow: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_fingerprint, str) or not _FINGERPRINT_RE.match(
            self.model_fingerprint
        ):
            raise InvalidSettingsKeyError(
                "model_fingerprint must be 64 lowercase hex characters"
            )
        for name in ("engine", "workflow"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SIMPLE_ID_RE.match(value):
                raise InvalidSettingsKeyError(
                    f"{name} must be a simple stable id "
                    f"([a-z0-9][a-z0-9._-], max 64), got {value!r}"
                )

    def to_dict(self) -> dict:
        return {
            "model_fingerprint": self.model_fingerprint,
            "engine": self.engine,
            "workflow": self.workflow,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SettingsKey:
        try:
            return cls(
                model_fingerprint=payload["model_fingerprint"],
                engine=payload["engine"],
                workflow=payload["workflow"],
            )
        except (KeyError, TypeError) as exc:
            raise InvalidSettingsKeyError(f"bad settings key payload: {exc}") from exc


# ---------------------------------------------------------------------------
# Trusted adapter schema


@dataclass(frozen=True)
class SettingSpec:
    """One allowlisted setting: type, bounds/enum, auto support, apply point."""

    key: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None
    supports_auto: bool = False
    apply: str = "request"

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _SIMPLE_ID_RE.match(self.key):
            raise ValueError(f"setting key must be a simple stable id: {self.key!r}")
        if self.kind not in _SETTING_KINDS:
            raise ValueError(f"unknown setting kind {self.kind!r} for {self.key!r}")
        if self.apply not in _APPLY_MODES:
            raise ValueError(f"unknown apply mode {self.apply!r} for {self.key!r}")
        if self.kind == "enum":
            if (not isinstance(self.choices, tuple) or not self.choices
                    or not all(isinstance(c, str) for c in self.choices)):
                raise ValueError(f"enum setting {self.key!r} needs non-empty choices")
            if self.minimum is not None or self.maximum is not None:
                raise ValueError(f"enum setting {self.key!r} takes no numeric range")
        elif self.choices is not None:
            raise ValueError(f"kind {self.kind!r} setting {self.key!r} takes no choices")
        if (self.minimum is not None and self.maximum is not None
                and self.minimum > self.maximum):
            raise ValueError(f"inverted range for setting {self.key!r}")
        if not isinstance(self.supports_auto, bool):
            raise ValueError(f"supports_auto must be a bool for {self.key!r}")


@dataclass(frozen=True)
class EngineSchema:
    """Allowlisted settings one engine build advertises for one workflow.

    Supplied by the trusted adapter that pins the engine, never by a renderer.
    """

    engine: str
    workflow: str
    schema_version: int
    settings: tuple[SettingSpec, ...]
    _by_key: Mapping[str, SettingSpec] = field(init=False, repr=False,
                                               compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.engine, str) or not _SIMPLE_ID_RE.match(self.engine):
            raise ValueError(f"bad engine id: {self.engine!r}")
        if not isinstance(self.workflow, str) or not _SIMPLE_ID_RE.match(self.workflow):
            raise ValueError(f"bad workflow id: {self.workflow!r}")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) \
                or self.schema_version < 1:
            raise ValueError("schema_version must be a positive int")
        if not isinstance(self.settings, tuple):
            raise ValueError("settings must be a tuple of SettingSpec")
        by_key: dict[str, SettingSpec] = {}
        for spec in self.settings:
            if not isinstance(spec, SettingSpec):
                raise ValueError(f"not a SettingSpec: {spec!r}")
            if spec.key in by_key:
                raise ValueError(f"duplicate setting key {spec.key!r}")
            by_key[spec.key] = spec
        object.__setattr__(self, "_by_key", MappingProxyType(by_key))

    def keys(self) -> tuple[str, ...]:
        return tuple(self._by_key)

    def spec(self, key: str) -> SettingSpec | None:
        return self._by_key.get(key)

    def validate_value(self, key: str, value: Any) -> Any:
        """Return the value if the schema accepts it; raise otherwise."""
        spec = self._by_key.get(key)
        if spec is None:
            raise UnknownSettingError(f"setting {key!r} is not in the {self.engine} "
                                      f"{self.workflow} schema")
        if value is AUTO:
            if not spec.supports_auto:
                raise InvalidSettingValueError(f"setting {key!r} does not support auto")
            return value
        self._check_range_and_kind(spec, key, value)
        return value

    @staticmethod
    def _check_range_and_kind(spec: SettingSpec, key: str, value: Any) -> None:
        if spec.kind == "bool":
            if not isinstance(value, bool):
                raise InvalidSettingValueError(f"setting {key!r} needs a bool")
            return
        if spec.kind == "enum":
            choices = spec.choices or ()
            if not isinstance(value, str) or value not in choices:
                raise InvalidSettingValueError(
                    f"setting {key!r} needs one of {list(choices)}"
                )
            return
        # int / float kinds: bool is never a number, NaN/Inf never acceptable.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidSettingValueError(
                f"setting {key!r} needs a number, got {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidSettingValueError(f"setting {key!r} needs a finite number")
        if spec.kind == "int" and not isinstance(value, int):
            raise InvalidSettingValueError(f"setting {key!r} needs an integer")
        if spec.minimum is not None and value < spec.minimum:
            raise InvalidSettingValueError(f"setting {key!r} below minimum {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise InvalidSettingValueError(f"setting {key!r} above maximum {spec.maximum}")

    def validate_overrides(self, overrides: Mapping[str, Any]) -> dict:
        """Validate a complete override payload; returns a plain dict copy."""
        validated: dict[str, Any] = {}
        for key, value in overrides.items():
            validated[key] = self.validate_value(key, value)
        return validated


# ---------------------------------------------------------------------------
# Snapshots


@dataclass(frozen=True)
class SettingsSnapshot:
    key: SettingsKey
    schema_version: int
    revision: int
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    def to_dict(self) -> dict:
        return {
            "key": self.key.to_dict(),
            "schema_version": self.schema_version,
            "revision": self.revision,
            "overrides": {k: _encode_value(v) for k, v in self.overrides.items()},
        }


def _absent_snapshot(key: SettingsKey) -> SettingsSnapshot:
    return SettingsSnapshot(key=key, schema_version=STORE_SCHEMA_VERSION,
                            revision=0, overrides={})


# ---------------------------------------------------------------------------
# Persistence


def _decode_overrides(raw: str) -> dict:
    try:
        payload = json.loads(
            raw, parse_constant=lambda name: (_ for _ in ()).throw(
                SettingsCorruptionError(f"non-finite JSON constant {name!r}")
            )
        )
    except (json.JSONDecodeError, SettingsCorruptionError) as exc:
        raise SettingsCorruptionError(f"undecodable overrides: {exc}") from exc
    if not isinstance(payload, dict):
        raise SettingsCorruptionError("overrides payload is not a JSON object")
    decoded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (bool, str, int)):
            decoded[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise SettingsCorruptionError(f"non-finite stored value for {key!r}")
            decoded[key] = value
        elif (isinstance(value, dict) and set(value) == {"auto"}
              and value["auto"] is True):
            decoded[key] = AUTO
        else:
            raise SettingsCorruptionError(f"stored value of unsupported shape for {key!r}")
    return decoded


class ModelSettingsStore:
    """SQLite settings store for one profile-scoped Hermes home.

    ``root`` must be the server-owned profile home (``get_hermes_home()``
    scope). The database lives at ``<root>/local_runtime/model_settings.sqlite3``.
    All writes run under ``BEGIN IMMEDIATE`` with a busy timeout, so separate
    processes serialise correctly; every mutation is compare-and-swap on the
    per-scope ``revision``. Reads never create the database file.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def path(self) -> Path:
        return self._root.joinpath(*_DB_RELATIVE)

    # -- connection ---------------------------------------------------------

    def _connect(self, create_parent: bool,
                 read_only: bool = False) -> sqlite3.Connection:
        if create_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            # Open an existing database strictly read-only: a read can never
            # recreate (or "repair") the file, not even under a path race.
            uri = f"{self.path.absolute().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True,
                                   timeout=_BUSY_TIMEOUT_MS / 1000.0)
        else:
            conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000.0)
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        except sqlite3.Error:
            conn.close()
            raise
        return conn

    # -- reads (never create anything) --------------------------------------

    def read(self, key: SettingsKey) -> SettingsSnapshot:
        if not self.path.exists():
            return _absent_snapshot(key)
        try:
            conn = self._connect(create_parent=False, read_only=True)
        except sqlite3.DatabaseError as exc:
            _fail_domain_for_sqlite(exc)  # e.g. the file vanished or is unreadable
        try:
            try:
                row = conn.execute(
                    "SELECT schema_version, revision, overrides_json FROM model_settings "
                    "WHERE fingerprint = ? AND engine = ? AND workflow = ?",
                    (key.model_fingerprint, key.engine, key.workflow),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                if "no such table: model_settings" in str(exc):
                    # A database file with no table yet (created, never
                    # committed) is absent settings — but only if it holds no
                    # tables this build does not recognise.
                    _require_known_table_layout(conn)
                    return _absent_snapshot(key)
                _fail_domain_for_sqlite(exc)  # e.g. no such column: foreign layout
            except sqlite3.DatabaseError as exc:
                _fail_domain_for_sqlite(exc)  # e.g. file is not a database
        finally:
            conn.close()
        if row is None:
            return _absent_snapshot(key)
        stored_version, revision, overrides_json = row
        _require_supported_stored_version(stored_version)
        return SettingsSnapshot(
            key=key, schema_version=stored_version, revision=revision,
            overrides=_decode_overrides(overrides_json),
        )

    # -- compare-and-swap writes ---------------------------------------------

    def write(self, key: SettingsKey, overrides: Mapping[str, Any],
              expected_revision: int) -> SettingsSnapshot:
        _require_expected_revision(expected_revision)
        payload = {
            str(k): _encode_value(v) for k, v in overrides.items()
        }
        if expected_revision == 0 and not self.path.exists():
            return self._first_write(key, overrides)
        return self._cas(key, expected_revision,
                         lambda current: json.dumps(payload, allow_nan=False,
                                                    sort_keys=True))

    def reset(self, key: SettingsKey, expected_revision: int,
              keys: Sequence[str] | None = None) -> SettingsSnapshot:
        """Reset deletes the overrides, never the revision history.

        Reset-all on a present scope leaves an empty-override snapshot at a
        strictly newer revision (a rewound revision would let a stale reader
        that saw revision 0 before the create write after the reset — ABA).
        Reset-all on an absent scope is a no-op at revision 0 that creates no
        files; a partial reset only removes the selected keys.
        """
        _require_expected_revision(expected_revision)
        if expected_revision == 0 and not self.path.exists():
            # Resetting an absent store to inherit is a no-op (no files).
            return _absent_snapshot(key)
        selected = None if keys is None else [str(k) for k in keys]

        def mutate(current_json: str | None) -> str | None:
            if selected is None and current_json is None:
                return None  # reset-all on an absent scope: no-op, revision 0
            current = ({} if selected is None
                       else _decode_overrides(current_json or "{}"))
            for name in selected or ():
                current.pop(name, None)
            return json.dumps(current, allow_nan=False, sort_keys=True)

        return self._cas(key, expected_revision, mutate)

    # -- internals -----------------------------------------------------------

    def _first_write(self, key: SettingsKey,
                     overrides: Mapping[str, Any]) -> SettingsSnapshot:
        payload = {str(k): _encode_value(v) for k, v in overrides.items()}
        blob = json.dumps(payload, allow_nan=False, sort_keys=True)
        conn = self._connect(create_parent=True)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                _require_known_table_layout(conn)
                conn.execute(_CREATE_SQL)
                exists = conn.execute(
                    "SELECT revision FROM model_settings "
                    "WHERE fingerprint = ? AND engine = ? AND workflow = ?",
                    (key.model_fingerprint, key.engine, key.workflow),
                ).fetchone()
                if exists is not None:
                    conn.execute("ROLLBACK")
                    raise RevisionConflictError(0, exists[0])
                conn.execute(
                    "INSERT INTO model_settings (fingerprint, engine, workflow, "
                    "schema_version, revision, overrides_json, updated_at) "
                    "VALUES (?, ?, ?, ?, 1, ?, ?)",
                    (key.model_fingerprint, key.engine, key.workflow,
                     STORE_SCHEMA_VERSION, blob, _utc_now()),
                )
                conn.execute("COMMIT")
            except sqlite3.DatabaseError as exc:
                _fail_domain_for_sqlite(exc)
        except BaseException:
            _safe_rollback(conn)
            raise
        finally:
            conn.close()
        return SettingsSnapshot(key=key, schema_version=STORE_SCHEMA_VERSION,
                                revision=1, overrides=dict(overrides))

    def _cas(self, key: SettingsKey, expected_revision: int,
             mutate) -> SettingsSnapshot:
        """Run one BEGIN IMMEDIATE transaction implementing the revision CAS.

        ``mutate(current_json) -> new_json | None`` returns the new payload,
        or ``None`` to leave an absent scope untouched (reset-all on a scope
        that never existed: a revision-0 no-op, no delete, no insert).
        """
        conn = self._connect(create_parent=True)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                _require_known_table_layout(conn)
                conn.execute(_CREATE_SQL)
                row = conn.execute(
                    "SELECT schema_version, revision, overrides_json FROM model_settings "
                    "WHERE fingerprint = ? AND engine = ? AND workflow = ?",
                    (key.model_fingerprint, key.engine, key.workflow),
                ).fetchone()
                current_revision = 0 if row is None else row[1]
                if row is not None:
                    _require_supported_stored_version(row[0])
                if current_revision != expected_revision:
                    conn.execute("ROLLBACK")
                    raise RevisionConflictError(expected_revision, current_revision)
                if row is None:  # expected 0, absent: first write of this scope
                    new_json = mutate(None)
                    if new_json is None:  # reset-all on an absent scope: no-op
                        conn.execute("ROLLBACK")
                        return _absent_snapshot(key)
                    conn.execute(
                        "INSERT INTO model_settings (fingerprint, engine, workflow, "
                        "schema_version, revision, overrides_json, updated_at) "
                        "VALUES (?, ?, ?, ?, 1, ?, ?)",
                        (key.model_fingerprint, key.engine, key.workflow,
                         STORE_SCHEMA_VERSION, new_json, _utc_now()),
                    )
                    conn.execute("COMMIT")
                    conn.close()
                    return SettingsSnapshot(key=key,
                                            schema_version=STORE_SCHEMA_VERSION,
                                            revision=1,
                                            overrides=_decode_overrides(new_json))
                # Present row: every mutation bumps the revision; the row is
                # never deleted, so the revision history stays monotonic and
                # stale compare-and-swaps keep losing.
                new_json = mutate(row[2])
                if new_json is None:  # unreachable: present-row mutations yield a payload
                    raise ModelSettingsError(
                        "internal error: CAS mutation yielded no payload for an "
                        "existing row")
                conn.execute(
                    "UPDATE model_settings SET overrides_json = ?, revision = ?, "
                    "updated_at = ? WHERE fingerprint = ? AND engine = ? AND workflow = ?",
                    (new_json, current_revision + 1, _utc_now(),
                     key.model_fingerprint, key.engine, key.workflow),
                )
                conn.execute("COMMIT")
            except sqlite3.DatabaseError as exc:
                _fail_domain_for_sqlite(exc)
        except BaseException:
            _safe_rollback(conn)
            raise
        finally:
            conn.close()
        return SettingsSnapshot(key=key, schema_version=STORE_SCHEMA_VERSION,
                                revision=current_revision + 1,
                                overrides=_decode_overrides(new_json))


_CREATE_SQL = (
    "CREATE TABLE IF NOT EXISTS model_settings ("
    " fingerprint TEXT NOT NULL,"
    " engine TEXT NOT NULL,"
    " workflow TEXT NOT NULL,"
    " schema_version INTEGER NOT NULL,"
    " revision INTEGER NOT NULL,"
    " overrides_json TEXT NOT NULL,"
    " updated_at TEXT NOT NULL,"
    " PRIMARY KEY (fingerprint, engine, workflow))"
)


def _require_supported_stored_version(stored: int) -> None:
    if stored != STORE_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"stored settings schema version {stored} is not readable by this "
            f"build (supports {STORE_SCHEMA_VERSION})"
        )


def _require_known_table_layout(conn: sqlite3.Connection) -> None:
    """Reject a database that holds tables this build does not know.

    An absent ``model_settings`` table normally means "created, never
    committed" (= absent settings). But a database carrying other tables —
    e.g. one migrated by a future build — must never read as absence or be
    silently repopulated beside the foreign layout.
    """
    names = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    unknown = sorted(names - {"model_settings"})
    if unknown:
        raise SettingsCorruptionError(
            "model settings database has an unrecognized table layout: "
            f"{', '.join(unknown)} (written by a newer or different build?)"
        )


def _fail_domain_for_sqlite(exc: sqlite3.Error) -> NoReturn:
    """Translate an sqlite failure on an existing database into a domain error.

    Lock contention is transient scheduling, not broken data, and stays the
    raw sqlite error for callers to retry. Everything else — a file that is
    not a database, a malformed image, an unreadable/missing file, a table
    layout this build cannot read — is corruption: fail loud, never let it
    masquerade as absent settings.
    """
    message = str(exc).lower()
    if "database is locked" in message or "database is busy" in message:
        raise exc
    raise SettingsCorruptionError(
        f"model settings database is unreadable: {exc}") from exc


def _require_expected_revision(expected_revision: int) -> None:
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise ModelSettingsError("expected_revision must be an int")
    if expected_revision < 0:
        raise ModelSettingsError("expected_revision cannot be negative")


def _safe_rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Resolver


_SOURCE_REQUEST = "request"
_SOURCE_MODEL = "model"
_SOURCE_PROFILE = "profile"
_SOURCE_PUBLISHER = "publisher"
_SOURCE_CATALOG = "catalog"
_SOURCE_ENGINE = "engine"
_SOURCE_INHERIT = "inherit"

_METADATA_SOURCES = (_SOURCE_PROFILE, _SOURCE_PUBLISHER, _SOURCE_CATALOG,
                     _SOURCE_ENGINE)


@dataclass(frozen=True)
class ResolutionResult:
    """Effective values plus per-key provenance for one owner scope."""

    values: Mapping[str, Any]
    sources: Mapping[str, str]

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def to_dict(self) -> dict:
        return {
            "values": {k: _encode_value(v) for k, v in self.values.items()},
            "sources": dict(self.sources),
        }


class ModelSettingsService:
    """Domain service over one profile's store: save / reset / resolve."""

    def __init__(self, store: ModelSettingsStore) -> None:
        self._store = store

    @property
    def store(self) -> ModelSettingsStore:
        return self._store

    def read(self, key: SettingsKey) -> SettingsSnapshot:
        return self._store.read(key)

    def save(self, key: SettingsKey, schema: EngineSchema,
             overrides: Mapping[str, Any],
             expected_revision: int) -> SettingsSnapshot:
        _require_schema_scope(schema, key)
        _require_supported_supplied_version(schema)
        validated = schema.validate_overrides(overrides)
        return self._store.write(key, validated, expected_revision)

    def reset(self, key: SettingsKey, expected_revision: int,
              keys: Sequence[str] | None = None) -> SettingsSnapshot:
        return self._store.reset(key, expected_revision, keys)

    def resolve(self, key: SettingsKey, schema: EngineSchema, *,
                request: Mapping[str, Any] | None = None,
                profile: Mapping[str, Any] | None = None,
                publisher: Mapping[str, Any] | None = None,
                catalog: Mapping[str, Any] | None = None,
                engine_reported: Mapping[str, Any] | None = None) -> ResolutionResult:
        _require_schema_scope(schema, key)
        _require_supported_supplied_version(schema)
        saved = self._store.read(key).overrides
        request_layer = _strict_layer(schema, request, _SOURCE_REQUEST)
        saved_layer = _strict_layer(schema, saved, _SOURCE_MODEL)
        # Profile defaults are caller-supplied authoritative user settings,
        # not untrusted metadata: they fail closed like request/saved layers.
        profile_layer = _strict_layer(schema, profile, _SOURCE_PROFILE)
        layers = [
            (request_layer, _SOURCE_REQUEST),
            (saved_layer, _SOURCE_MODEL),
            (profile_layer, _SOURCE_PROFILE),
            (_lenient_layer(schema, publisher), _SOURCE_PUBLISHER),
            (_lenient_layer(schema, catalog), _SOURCE_CATALOG),
            (_lenient_layer(schema, engine_reported), _SOURCE_ENGINE),
        ]
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for spec_key in schema.keys():
            for layer, source in layers:
                if spec_key in layer:
                    values[spec_key] = layer[spec_key]
                    sources[spec_key] = source
                    break
            else:
                sources[spec_key] = _SOURCE_INHERIT
        return ResolutionResult(values=values, sources=sources)


def _require_schema_scope(schema: EngineSchema, key: SettingsKey) -> None:
    if schema.engine != key.engine or schema.workflow != key.workflow:
        raise ScopeMismatchError(
            f"schema for {schema.engine}/{schema.workflow} does not match "
            f"settings scope {key.engine}/{key.workflow}"
        )


def _require_supported_supplied_version(schema: EngineSchema) -> None:
    if schema.schema_version != STORE_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"adapter schema version {schema.schema_version} is not supported "
            f"by this build (supports {STORE_SCHEMA_VERSION})"
        )


def _strict_layer(schema: EngineSchema, layer: Mapping[str, Any] | None,
                  source: str) -> dict:
    """Authoritative layers (request, saved model, profile defaults): these
    carry caller-supplied user settings, so unknown keys and bad values raise."""
    result: dict[str, Any] = {}
    for name, value in (layer or {}).items():
        try:
            result[name] = schema.validate_value(name, value)
        except UnknownSettingError as exc:
            raise UnknownSettingError(f"{source} layer: {exc}") from exc
        except InvalidSettingValueError as exc:
            raise InvalidSettingValueError(f"{source} layer: {exc}") from exc
    return result


def _lenient_layer(schema: EngineSchema, layer: Mapping[str, Any] | None) -> dict:
    """Untrusted metadata layers (publisher/file metadata, curated catalog,
    reported engine defaults): keys outside the schema or invalid values are
    skipped, so stale file/catalog metadata cannot poison resolution.

    Profile defaults are deliberately NOT lenient — they are authoritative
    user settings and go through ``_strict_layer`` instead.
    """
    result: dict[str, Any] = {}
    for name, value in (layer or {}).items():
        try:
            result[name] = schema.validate_value(name, value)
        except (UnknownSettingError, InvalidSettingValueError):
            continue
    return result


# ---------------------------------------------------------------------------
# Wire mapping


def split_request_fields(schema: EngineSchema,
                         resolution: ResolutionResult) -> tuple[dict, dict]:
    """Split resolved settings into OpenAI standard fields and extra_body.

    ``apply="load"`` settings never reach a request body; inherited keys are
    omitted entirely. The caller merges ``extra_body`` PER KEY into any
    existing request extensions (never a blind ``dict.update``) so unrelated
    extensions survive.
    """
    standard: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    for spec in schema.settings:
        if spec.apply != "request" or spec.key not in resolution.values:
            continue
        value = resolution.values[spec.key]
        if spec.key in OPENAI_STANDARD_FIELDS:
            if value is AUTO:  # no standard wire token for auto: omit
                continue
            standard[spec.key] = value
        else:
            extra_body[spec.key] = "auto" if value is AUTO else value
    return standard, extra_body


__all__ = [
    "AUTO",
    "OPENAI_STANDARD_FIELDS",
    "STORE_SCHEMA_VERSION",
    "EngineSchema",
    "InvalidSettingValueError",
    "InvalidSettingsKeyError",
    "ModelSettingsError",
    "ModelSettingsService",
    "ModelSettingsStore",
    "ResolutionResult",
    "RevisionConflictError",
    "ScopeMismatchError",
    "SettingSpec",
    "SettingsCorruptionError",
    "SettingsKey",
    "SettingsSnapshot",
    "UnknownSettingError",
    "UnsupportedSchemaVersionError",
    "split_request_fields",
]
