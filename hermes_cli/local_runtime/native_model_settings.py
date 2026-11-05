"""Native llama.cpp settings adapter: artifact identity, trusted schema, wire map.

Sits on top of the generic domain core (``model_settings``) and owns exactly
three things:

* **Identity** — the settings fingerprint is derived from the EXACT bytes of
  the staged GGUF (every split part) plus the REQUIRED catalog companions
  (mmproj projector / MTP head) when present on disk. Never the optional
  draft (rotating one must not orphan saved settings), never catalog display
  names (editorial text changes must not orphan them either). Files are
  hashed by streaming with a per-file stat guard: a file that changes
  mid-hash raises instead of yielding a mixed-generation digest.
* **Trusted schema** — ``NATIVE_CHAT_SCHEMA`` is the pinned llama.cpp chat
  sampling set. It contains no load knobs (n_ctx and friends stay Auto
  policy, decided elsewhere), so everything in it is ``apply="request"``.
* **Wire mapping** — decode tagged-AUTO payloads from the OpenAI shape, and
  resolve a request into ``(standard_fields, extra_body)`` with full
  precedence (request > saved model > publisher GGUF metadata > catalog).

Layered defaults for display follow the same ladder: real GGUF metadata
first, curated catalog second; nothing is ever invented for a key no layer
answers (it reads as ``inherit``).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from hermes_cli.local_runtime import bootstrap, catalog
from hermes_cli.local_runtime.gguf import model_id_from_stem, read_gguf_header, split_parts
from hermes_cli.local_runtime.model_settings import (
    AUTO,
    EngineSchema,
    InvalidSettingValueError,
    ModelSettingsError,
    ModelSettingsService,
    ModelSettingsStore,
    SettingSpec,
    SettingsKey,
    STORE_SCHEMA_VERSION,
    UnknownSettingError,
    split_request_fields,
)

logger = logging.getLogger(__name__)

ENGINE = "llamacpp"
WORKFLOW = "chat"

__all__ = [
    "NATIVE_CHAT_SCHEMA",
    "STORE_SCHEMA_VERSION",
    "ArtifactIdentityError",
    "InvalidModelIdError",
    "MissingBindingError",
    "ModelChangedError",
    "NativeModelSettingsError",
    "UnknownModelError",
    "artifact_fingerprint",
    "catalog_sampling_defaults",
    "clear_fingerprint_cache",
    "decode_wire_overrides",
    "gguf_sampling_metadata",
    "read_model_settings",
    "reset_model_settings",
    "resolve_native_request_settings",
    "resolve_staged_model",
    "save_model_settings",
    "settings_key_for_model",
]


# ---------------------------------------------------------------------------
# Errors (adapter-specific; all inside the model-settings domain family)


class NativeModelSettingsError(ModelSettingsError):
    """Base for adapter failures (identity, binding, staged-model lookup)."""


class InvalidModelIdError(NativeModelSettingsError):
    """The supplied model id is malformed (empty, hidden, or path-shaped)."""


class UnknownModelError(NativeModelSettingsError):
    """No servable staged GGUF carries this model id."""


class ArtifactIdentityError(NativeModelSettingsError):
    """The artifact changed while it was being hashed: refuse the torn digest."""


class MissingBindingError(NativeModelSettingsError):
    """A write arrived without the fingerprint the caller read it against."""


class ModelChangedError(NativeModelSettingsError):
    """The artifact on disk no longer matches the caller's fingerprint.

    ``current_fingerprint`` is the live identity so the caller can re-read
    against it; the write is refused BEFORE any compare-and-swap touches the
    store, so a stale binding can never adopt settings onto a new artifact.
    """

    def __init__(self, model_id: str, current_fingerprint: str) -> None:
        super().__init__(
            f"model {model_id!r} changed on disk since it was read "
            f"(fingerprint {current_fingerprint[:12]}… is current)"
        )
        self.current_fingerprint = current_fingerprint


# ---------------------------------------------------------------------------
# Trusted engine schema: the pinned llama.cpp chat sampling set


NATIVE_CHAT_SCHEMA = EngineSchema(
    engine=ENGINE,
    workflow=WORKFLOW,
    schema_version=STORE_SCHEMA_VERSION,
    settings=(
        # Ranges are pinned to the OpenAI-compatible surface llama.cpp
        # exposes — not invented. top_k/seed are llama.cpp extensions with
        # no OpenAI upper bound; only a sanity floor is pinned.
        SettingSpec("temperature", "float", minimum=0.0, maximum=2.0,
                    supports_auto=True),
        SettingSpec("top_p", "float", minimum=0.0, maximum=1.0,
                    supports_auto=True),
        SettingSpec("top_k", "int", minimum=0, supports_auto=True),
        SettingSpec("min_p", "float", minimum=0.0, maximum=1.0,
                    supports_auto=True),
        SettingSpec("repeat_penalty", "float", minimum=0.0, maximum=2.0,
                    supports_auto=True),
        SettingSpec("presence_penalty", "float", minimum=-2.0, maximum=2.0,
                    supports_auto=True),
        SettingSpec("frequency_penalty", "float", minimum=-2.0, maximum=2.0,
                    supports_auto=True),
        SettingSpec("seed", "int", minimum=0, supports_auto=True),
    ),
)


# ---------------------------------------------------------------------------
# Staged-model lookup


def _validate_model_id(model_id: Any) -> str:
    if (not isinstance(model_id, str) or not model_id
            or model_id != model_id.strip()):
        raise InvalidModelIdError(f"model id must be a non-empty name: {model_id!r}")
    if "\x00" in model_id or "/" in model_id or "\\" in model_id:
        raise InvalidModelIdError(f"model id must not be a path: {model_id!r}")
    if model_id.startswith(".") or model_id in {".", ".."}:
        raise InvalidModelIdError(f"model id must not be hidden or relative: {model_id!r}")
    return model_id


def resolve_staged_model(model_id: str) -> Path:
    """The primary GGUF path a staged model loads from (split: first part)."""
    model_id = _validate_model_id(model_id)
    mdir = bootstrap.models_dir()
    if mdir.is_dir():
        for path in bootstrap.staged_in(mdir):
            if model_id_from_stem(path.stem) == model_id:
                return path
    raise UnknownModelError(f"no staged model named {model_id!r}")


# ---------------------------------------------------------------------------
# Artifact identity: exact bytes, streamed, stat-guarded

_CHUNK_BYTES = 1 << 20
_FINGERPRINT_RE = re.compile(r"\A[0-9a-f]{64}\Z")

# model_id -> (per-file stat tokens that produced the digest, digest)
_fingerprint_cache: dict[str, tuple[tuple, str]] = {}


def clear_fingerprint_cache() -> None:
    """Drop every memoised fingerprint (test seam; the cache is stat-guarded)."""
    _fingerprint_cache.clear()


def _iter_file_chunks(path: Path):
    """Stream one file's bytes; never materialises a multi-GB model."""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_BYTES)
            if not chunk:
                return
            yield chunk


def _stat_token(path: Path) -> tuple:
    st = path.stat()
    return (path.name, st.st_size, st.st_mtime_ns, st.st_ino, st.st_mode)


def _artifact_files(model_id: str) -> list[Path]:
    """Every file whose bytes define this artifact, in stable order.

    The model (all split parts) plus REQUIRED catalog companions — mmproj
    and MTP head — once they are on disk. The optional draft is deliberately
    absent: its arrival or rotation must not orphan the model's settings.
    Catalog names/ids are never bytes, so they cannot leak in here.
    """
    primary = resolve_staged_model(model_id)
    files = list(split_parts(primary) or [primary])
    entry = catalog.entry_for_model(model_id)
    if entry is not None:
        adir = bootstrap.assets_dir()
        for asset in (entry.mmproj, entry.mtp_head):
            if asset is None:
                continue
            companion = adir / asset.local_name
            if companion.is_file():
                files.append(companion)
    files.sort(key=lambda p: p.name)
    return files


def _hash_files(files: list[Path]) -> tuple[str, tuple]:
    """SHA-256 over (name, bytes) of each file, stat-checked before/after.

    A file whose stat moves across its own streaming means a concurrent
    writer: the digest would mix generations, so refuse it. Refusals raise
    and are never cached — the next clean hash reflects the CURRENT bytes.
    """
    digest = hashlib.sha256()
    tokens: list[tuple] = []
    for path in files:
        before = _stat_token(path)
        digest.update(before[0].encode("utf-8") + b"\0")
        for chunk in _iter_file_chunks(path):
            digest.update(chunk)
        if _stat_token(path) != before:
            raise ArtifactIdentityError(
                f"{path.name} changed while its fingerprint was being computed; "
                "refusing a mixed-generation digest")
        tokens.append(before)
    return digest.hexdigest(), tuple(tokens)


def artifact_fingerprint(model_id: str) -> str:
    """64-hex identity of the exact installed artifact (cache + stat guard).

    The cache stores the per-file stat tokens it hashed; any model part
    appearing, vanishing or being rewritten changes the token tuple and
    forces a re-hash, so a hit is always the current generation.
    """
    files = _artifact_files(model_id)
    tokens = tuple(_stat_token(p) for p in files)
    hit = _fingerprint_cache.get(model_id)
    if hit is not None and hit[0] == tokens:
        return hit[1]
    fingerprint, hashed_tokens = _hash_files(files)
    _fingerprint_cache[model_id] = (hashed_tokens, fingerprint)
    return fingerprint


def settings_key_for_model(model_id: str) -> SettingsKey:
    """Owner scope: the live artifact fingerprint under llamacpp/chat."""
    return SettingsKey(model_fingerprint=artifact_fingerprint(model_id),
                       engine=ENGINE, workflow=WORKFLOW)


def _require_fingerprint_binding(model_id: str,
                                 expected_fingerprint: Any) -> SettingsKey:
    """Every write must carry the fingerprint the caller read the model at.

    Checked BEFORE any store access: a stale binding fails as a model-change
    conflict, never as a revision conflict or (worse) a silent adoption.
    """
    current = artifact_fingerprint(model_id)
    if (not isinstance(expected_fingerprint, str)
            or not _FINGERPRINT_RE.match(expected_fingerprint)):
        raise MissingBindingError(
            f"write for {model_id!r} must carry the exact model_fingerprint "
            "returned by the settings read")
    if expected_fingerprint != current:
        raise ModelChangedError(model_id, current)
    return SettingsKey(model_fingerprint=current, engine=ENGINE,
                       workflow=WORKFLOW)


# ---------------------------------------------------------------------------
# Layered defaults: GGUF publisher metadata first, curated catalog second


def _read_gguf_metadata(model_id: str) -> dict:
    """Raw GGUF metadata of the staged model; {} when it cannot be read."""
    try:
        return dict(read_gguf_header(resolve_staged_model(model_id)).metadata)
    except Exception:  # the settings pane must render for a missing or damaged file of any kind
        logger.debug("GGUF metadata of %s is unreadable", model_id, exc_info=True)
        return {}


# ``general.sampling.*`` suffix -> setting key (both spellings publishers use).
_GGUF_SAMPLING_KEYS = {
    "temp": "temperature", "temperature": "temperature",
    "top_p": "top_p", "top-p": "top_p",
    "top_k": "top_k", "top-k": "top_k",
    "min_p": "min_p", "min-p": "min_p",
    "repeat_penalty": "repeat_penalty", "repeat-penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty", "presence-penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty", "frequency-penalty": "frequency_penalty",
    "seed": "seed",
}

# Curated catalog ``sampling`` dict uses preset-INI long-form keys.
_CATALOG_SAMPLING_KEYS = {
    "temp": "temperature",
    "top-p": "top_p", "top_p": "top_p",
    "top-k": "top_k", "top_k": "top_k",
    "min-p": "min_p", "min_p": "min_p",
    "repeat-penalty": "repeat_penalty", "repeat_penalty": "repeat_penalty",
    "presence-penalty": "presence_penalty", "presence_penalty": "presence_penalty",
    "frequency-penalty": "frequency_penalty", "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}


def _coerce_to_kind(setting: str, value: Any) -> Any | None:
    """Coerce one untrusted metadata value to the schema's kind; None skips."""
    spec = NATIVE_CHAT_SCHEMA.spec(setting)
    if spec is None or isinstance(value, bool):
        return None
    try:
        if spec.kind == "int":
            number = float(value)
            if not number.is_integer():
                return None
            return int(number)
        if spec.kind == "float":
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def gguf_sampling_metadata(model_id: str) -> dict:
    """Publisher-baked sampling defaults from the GGUF itself (may be empty)."""
    out: dict[str, Any] = {}
    prefix = "general.sampling."
    for key, value in _read_gguf_metadata(model_id).items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        setting = _GGUF_SAMPLING_KEYS.get(key[len(prefix):])
        if setting is None:
            continue
        coerced = _coerce_to_kind(setting, value)
        if coerced is not None:
            out[setting] = coerced
    return out


def catalog_sampling_defaults(model_id: str) -> dict:
    """Curated catalog sampling hints for this staged model (may be empty)."""
    entry = catalog.entry_for_model(model_id)
    if entry is None:
        return {}
    out: dict[str, Any] = {}
    for key, value in (entry.sampling or {}).items():
        setting = _CATALOG_SAMPLING_KEYS.get(str(key).strip().lower())
        if setting is None:
            continue
        coerced = _coerce_to_kind(setting, value)
        if coerced is not None:
            out[setting] = coerced
    return out


# ---------------------------------------------------------------------------
# Wire decoding: tagged AUTO only, scalars only, schema keys only


def decode_wire_overrides(payload: Mapping[str, Any]) -> dict:
    """Decode a wire-shape override payload into domain values.

    Scalars pass through schema validation; the only accepted composite is
    the tagged auto ``{"auto": true}`` -> ``AUTO``. Nested objects, lists
    and untagged dicts are rejected outright — nothing is silently dropped.
    """
    decoded: dict[str, Any] = {}
    for name, value in dict(payload).items():
        if value is AUTO:
            decoded[name] = NATIVE_CHAT_SCHEMA.validate_value(name, AUTO)
        elif isinstance(value, dict):
            if set(value) == {"auto"} and value["auto"] is True:
                decoded[name] = NATIVE_CHAT_SCHEMA.validate_value(name, AUTO)
            else:
                raise InvalidSettingValueError(
                    f"setting {name!r}: the only accepted object form is "
                    '{{"auto": true}}')
        else:
            decoded[name] = NATIVE_CHAT_SCHEMA.validate_value(name, value)
    return decoded


# ---------------------------------------------------------------------------
# Read / save / reset round-trip through the domain service


def _service(root: Path | str) -> ModelSettingsService:
    return ModelSettingsService(ModelSettingsStore(Path(root)))


def _field_descriptors() -> list[dict]:
    """The trusted schema rendered for a UI: one fully typed entry per setting.
    Renderer sees bounds and modes, never re-derives them."""
    return [
        {"key": spec.key, "kind": spec.kind, "minimum": spec.minimum,
         "maximum": spec.maximum,
         "choices": list(spec.choices) if spec.choices else None,
         "supports_auto": spec.supports_auto, "apply": spec.apply}
        for spec in NATIVE_CHAT_SCHEMA.settings
    ]


def _runtime_facts() -> dict:
    """Facts about the engine that will serve this model — read from the
    installed runtime, never guessed. Degrades to nulls: the settings pane
    must render even before a runtime is installed."""
    facts = {"engine": ENGINE, "backend": None, "installed_tag": None,
             "pinned_tag": None}
    try:
        from hermes_cli.local_runtime import binaries
        engine = binaries.installed_engine()
        if engine is not None:
            facts["backend"] = engine.backend
            facts["installed_tag"] = engine.tag
            facts["pinned_tag"] = binaries.pinned_tag(engine.backend)
    except Exception:  # facts degrade to nulls; the settings pane must render before any runtime is installed
        logger.debug("runtime facts unavailable", exc_info=True)
    return facts


def _loading_facts(model_id: str) -> dict:
    """Load policy stays Auto (no load knob is user-settable here); the one
    fact shown is the window the preset planner actually granted this model,
    read back from the INI that spawned it. None until a launch has decided."""
    context_length = None
    try:
        from hermes_cli.local_runtime import presets
        decision = presets.read_preset_decisions().get(model_id)
        if decision is not None and decision.window > 0:
            context_length = int(decision.window)
    except Exception:  # a missing or unreadable preset record just means "no window decided yet"
        logger.debug("preset decision of %s unreadable", model_id, exc_info=True)
        context_length = None
    return {"policy": "auto", "context_length": context_length}


def _settings_response(root: Path | str, model_id: str, key: SettingsKey,
                       snapshot) -> dict:
    """Snapshot + effective resolution (publisher > catalog layered below
    the saved model). Keys no layer answers stay ``inherit``, never invented.
    Carries the full parent-UI contract: identity, typed fields, engine
    facts and the read-only loading view, so a save response equals a GET.
    """
    resolution = _service(root).resolve(
        key, NATIVE_CHAT_SCHEMA,
        publisher=gguf_sampling_metadata(model_id),
        catalog=catalog_sampling_defaults(model_id))
    response = snapshot.to_dict()
    response["model_id"] = model_id
    response["effective"] = resolution.to_dict()
    response["fields"] = _field_descriptors()
    response["runtime"] = _runtime_facts()
    response["loading"] = _loading_facts(model_id)
    return response


def read_model_settings(model_id: str, root: Path | str) -> dict:
    """Current settings view for the staged artifact under a profile root."""
    key = settings_key_for_model(model_id)
    snapshot = _service(root).read(key)
    return _settings_response(root, model_id, key, snapshot)


def save_model_settings(model_id: str, root: Path | str,
                        overrides: Mapping[str, Any], expected_revision: int,
                        expected_fingerprint: Any) -> dict:
    """CAS-replace overrides, gated on the caller's artifact fingerprint."""
    key = _require_fingerprint_binding(model_id, expected_fingerprint)
    decoded = decode_wire_overrides(overrides)
    snapshot = _service(root).save(key, NATIVE_CHAT_SCHEMA, decoded,
                                   expected_revision)
    return _settings_response(root, model_id, key, snapshot)


def reset_model_settings(model_id: str, root: Path | str,
                         expected_revision: int, expected_fingerprint: Any,
                         keys: list[str] | None = None) -> dict:
    """Drop overrides (all, or the selected schema keys) under the same gate."""
    key = _require_fingerprint_binding(model_id, expected_fingerprint)
    if keys is not None:
        for name in keys:
            if NATIVE_CHAT_SCHEMA.spec(name) is None:
                raise UnknownSettingError(
                    f"setting {name!r} is not in the {ENGINE} {WORKFLOW} schema")
    snapshot = _service(root).reset(key, expected_revision, keys)
    return _settings_response(root, model_id, key, snapshot)


# ---------------------------------------------------------------------------
# Dispatch helper: request precedence + OpenAI wire split


def resolve_native_request_settings(
        model_id: str,
        request: Mapping[str, Any] | None = None,
        *, inherited: bool = True) -> tuple[dict, dict]:
    """Resolve effective settings for a live request on the active profile.

    Returns ``(standard_fields, extra_body)`` for the OpenAI wire: request
    values win over saved overrides, which win over publisher GGUF metadata
    and curated catalog hints. The caller merges ``extra_body`` PER KEY into
    any extensions it already holds.

    ``inherited=False`` drops the publisher/catalog layers so only values the
    USER chose (explicit request + saved model settings) are returned. Live
    dispatch uses that: llama-server already applies publisher/catalog values
    as its own launch defaults, and re-sending them per request would override
    a hand-edited ``presets.ini``. The settings API keeps ``inherited=True`` to
    display the full effective ladder.
    """
    from hermes_constants import get_hermes_home

    key = settings_key_for_model(model_id)
    decoded = decode_wire_overrides(request) if request else {}
    resolution = _service(get_hermes_home()).resolve(
        key, NATIVE_CHAT_SCHEMA,
        request=decoded,
        publisher=gguf_sampling_metadata(model_id) if inherited else None,
        catalog=catalog_sampling_defaults(model_id) if inherited else None)
    return split_request_fields(NATIVE_CHAT_SCHEMA, resolution)


