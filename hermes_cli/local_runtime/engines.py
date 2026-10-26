"""Registry of local inference engines (Settings → Local models → Local engine).

By default Hermes serves local models with the PM-owned llama.cpp build (the implicit
``managed`` entry, resolved from ``local_runtime.backend``). Users can also register
their own engine binaries — a locally built llama.cpp, a nightly that outruns the PM
pin — select one active engine, and give it its own launch arguments and environment.
Everything lives under ``local_runtime`` in config, so an absent registry keeps the
pre-registry behavior bit-for-bit:

    local_runtime:
      engine: "managed"            # active engine id; "managed" = the PM build
      engines:                     # user-registered binaries (add/edit/delete via API)
        - id: my-vulkan-build
          name: "My Vulkan build"
          path: D:/llama/build/bin/llama-server.exe
          backend: vulkan          # hardware hint for budgets/policy: auto|cuda|metal|vulkan|hip|cpu
          args: ["--flash-attn"]   # appended verbatim to the launch command (list, no shell)
          env: {GGML_CUDA_ENABLE_UNIFIED: "1"}

The ``managed`` entry is never stored: it is the engine pm pins and the Local models
pane installs, so it has no path to edit and cannot be deleted.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

MANAGED_ID = "managed"
# Backends a custom binary may declare; mirrors binaries.BACKEND_PACKAGES + auto.
BACKENDS = ("auto", "cuda", "metal", "vulkan", "hip", "cpu")

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_ARGS = 64
_MAX_ARG_LEN = 4096
_MAX_ENV = 64
_MAX_NAME_LEN = 80


class EngineError(ValueError):
    """A registry operation or engine launch was rejected; message is user-facing."""


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug[:64]


def _unique_id(name: str, taken: set[str]) -> str:
    base = _slug(name) or "engine"
    if base == MANAGED_ID:
        base = "engine"
    candidate, n = base, 2
    while candidate in taken:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def parse_record(raw: object) -> dict | None:
    """One stored engine entry, sanitized; None when the entry is unusable (config is
    hand-editable: a broken row must not take the whole pane down)."""
    if not isinstance(raw, dict):
        return None
    engine_id = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()[:_MAX_NAME_LEN]
    path = str(raw.get("path") or "").strip()
    if not engine_id or not _ID_RE.match(engine_id) or engine_id == MANAGED_ID or not path:
        return None
    backend = str(raw.get("backend") or "auto").strip().lower()
    if backend not in BACKENDS:
        backend = "auto"
    args = [str(a) for a in (raw.get("args") or []) if str(a).strip()][:_MAX_ARGS]
    env: dict[str, str] = {}
    if isinstance(raw.get("env"), dict):
        for key, value in list(raw["env"].items())[:_MAX_ENV]:
            key = str(key).strip()
            if _ENV_KEY_RE.match(key):
                env[key] = str(value)
    return {"id": engine_id, "name": name or engine_id, "kind": "external",
            "path": path, "backend": backend, "args": args, "env": env}


def user_engines(section: dict) -> list[dict]:
    """Sanitized user-registered engines, in stored order."""
    out, seen = [], {MANAGED_ID}
    for raw in (section or {}).get("engines") or []:
        record = parse_record(raw)
        if record is not None and record["id"] not in seen:
            seen.add(record["id"])
            out.append(record)
    return out


def managed_record(section: dict) -> dict:
    return {"id": MANAGED_ID, "name": "Hermes-managed llama.cpp", "kind": "managed",
            "path": "", "backend": str((section or {}).get("backend") or "auto"),
            "args": [], "env": {}}


def get(section: dict, engine_id: str) -> dict | None:
    if engine_id == MANAGED_ID:
        return managed_record(section)
    return next((e for e in user_engines(section) if e["id"] == engine_id), None)


def active_id(section: dict) -> str:
    """The selected engine, falling back to managed when unset or deleted."""
    engine_id = str((section or {}).get("engine") or MANAGED_ID)
    if engine_id == MANAGED_ID or get(section, engine_id) is not None:
        return engine_id
    return MANAGED_ID


def active_engine(section: dict) -> dict:
    return get(section, active_id(section)) or managed_record(section)


def _installed_for(record: dict, section: dict) -> bool:
    if record["kind"] == "managed":
        from hermes_cli.local_runtime.binaries import BinaryResolutionError, installed_engine

        try:
            return installed_engine(section.get("backend", "auto")) is not None
        except (BinaryResolutionError, OSError):
            return False
    return _executable(Path(record["path"]))


def list_for_view(section: dict) -> list[dict]:
    """Every selectable engine with its state, for the pane. ``installed`` is cheap:
    managed asks pm's store, a custom engine just stats its path (never runs it)."""
    active = active_id(section)
    out = []
    for record in [managed_record(section), *user_engines(section)]:
        out.append({**record, "builtin": record["kind"] == "managed",
                    "active": record["id"] == active,
                    "installed": _installed_for(record, section)})
    return out


def _executable(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _fields_to_record(fields: dict, *, base: dict | None = None) -> dict:
    """Merge request fields onto a stored record with validation. Unknown keys are
    ignored; anything malformed raises EngineError with a user-facing message."""
    merged = {**(base or {}), **{k: v for k, v in fields.items()
                                 if k in ("name", "path", "backend", "args", "env")}}
    name = str(merged.get("name") or "").strip()
    if not name:
        raise EngineError("The engine needs a name.")
    path = str(merged.get("path") or "").strip()
    if not path:
        raise EngineError("The engine needs a binary path (e.g. .../llama-server).")
    backend = str(merged.get("backend") or "auto").strip().lower()
    if backend not in BACKENDS:
        raise EngineError(f"Unknown backend {backend!r}; expected one of: {', '.join(BACKENDS)}.")
    raw_args = merged.get("args") or []
    if not isinstance(raw_args, list):
        raise EngineError("Launch arguments must be a list.")
    args = [str(a).strip() for a in raw_args if str(a).strip()]
    if len(args) > _MAX_ARGS:
        raise EngineError(f"At most {_MAX_ARGS} launch arguments are allowed.")
    if any(len(a) > _MAX_ARG_LEN for a in args):
        raise EngineError("A launch argument is too long.")
    raw_env = merged.get("env") or {}
    if not isinstance(raw_env, dict):
        raise EngineError("Environment variables must be a mapping.")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        key = str(key).strip()
        if not _ENV_KEY_RE.match(key):
            raise EngineError(f"Invalid environment variable name: {key!r}")
        env[key] = str(value)
    if len(env) > _MAX_ENV:
        raise EngineError(f"At most {_MAX_ENV} environment variables are allowed.")
    return {"name": name[:_MAX_NAME_LEN], "path": str(Path(path).expanduser()),
            "backend": backend, "args": args, "env": env}


def build_new(section: dict, fields: dict) -> dict:
    """A brand-new engine record (validated), with a generated unique id."""
    record = _fields_to_record(fields)
    taken = {e["id"] for e in user_engines(section)}
    explicit = _slug(str(fields.get("id") or ""))
    if explicit == MANAGED_ID:
        raise EngineError(f"{MANAGED_ID!r} is reserved for the built-in engine.")
    if explicit and explicit in taken:
        raise EngineError(f"An engine with id {explicit!r} already exists.")
    return {"id": explicit or _unique_id(record["name"], taken), **record}


def build_update(section: dict, engine_id: str, fields: dict) -> dict:
    """The updated record for a stored engine (validated). The id never changes."""
    existing = get(section, engine_id)
    if existing is None:
        raise EngineError(f"No engine with id {engine_id!r}.")
    if existing["kind"] == "managed":
        raise EngineError("The built-in managed engine cannot be edited; register your "
                          "own build instead.")
    return {**existing, **_fields_to_record(fields, base=existing)}


def probe(binary_path: str) -> str:
    """Run ``<binary> --version`` and return its first meaningful line (llama.cpp prints
    ``version: N (HASH)`` plus build info to stderr). Raises EngineError when the binary
    cannot run — the message is what the pane shows next to the Test button."""
    path = Path(str(binary_path or "").strip()).expanduser()
    if not _executable(path):
        raise EngineError(f"No file at {path}.")
    try:
        proc = subprocess.run([str(path), "--version"], capture_output=True, timeout=15,
                              cwd=str(path.parent), text=True, errors="replace", check=False)
    except subprocess.TimeoutExpired:
        raise EngineError("The engine did not answer --version within 15 s.") from None
    except OSError as exc:
        raise EngineError(f"Could not run this binary: {exc}") from None
    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        tail = next((line.strip() for line in output.splitlines() if line.strip()), "")
        raise EngineError(f"Exit code {proc.returncode}. {tail}".strip())
    for line in output.splitlines():
        if line.strip():
            return line.strip()[:200]
    return "ok"


def resolve(section: dict) -> tuple[object, list[str], dict[str, str]]:
    """The (Engine, extra_args, extra_env) the supervisor must launch for the active engine.

    Managed keeps today's behavior exactly (PM store, backend auto-detect). A custom
    engine resolves to its own binary; the backend it declares is a budget hint, and its
    version is unknown to PM, so version-specific residency policy (tag) is skipped."""
    from hermes_cli.local_runtime.binaries import BinaryResolutionError, Engine, installed_engine

    engine = active_engine(section)
    if engine["kind"] == "managed":
        try:
            pm_engine = installed_engine(section.get("backend", "auto"))
        except (BinaryResolutionError, OSError) as exc:
            raise EngineError(str(exc)) from None
        if pm_engine is None:
            raise EngineError("No managed llama.cpp is installed yet — install it from the "
                              "Local models pane or pick another engine.")
        return pm_engine, list(engine["args"]), dict(engine["env"])
    path = Path(engine["path"]).expanduser()
    if not _executable(path):
        raise EngineError(f"Engine {engine['name']!r}: no file at {path}.")
    return Engine(backend=engine["backend"] if engine["backend"] != "auto" else "cpu",
                  tag="", binary=path), list(engine["args"]), dict(engine["env"])
