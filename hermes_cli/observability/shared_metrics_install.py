"""Install-snapshot fields for version lag, release channel and coarse hardware.

Everything here is read offline: the installed version's own commit date, the install stamp and
per-install channel record, the update check's cached result (never a network call), and cached
``hermes_platform.host`` facts. Only closed enums and buckets leave; the git remote, branch names,
paths and endpoint URLs stay local.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import shared_metrics_contract as contract
from .shared_metrics_contract import _bucket, _norm

logger = logging.getLogger(__name__)

_DAY_S = 86_400
_VERSION_AGE_THRESHOLDS = ((7 * _DAY_S, "lt_7d"), (30 * _DAY_S, "7d_to_30d"), (90 * _DAY_S, "30d_to_90d"))
_GIB = 1024 ** 3
# The OS reports installed RAM minus firmware/iGPU reservations (a 16 GB laptop shows ~15.5 GiB),
# so the total is scaled up before bucketing to land on the nominal size users buy.
_RAM_NOMINAL_FACTOR = 1.1
_RAM_THRESHOLDS = (
    (8, "lt_8g"), (16, "8g_to_16g"), (32, "16g_to_32g"), (64, "32g_to_64g"), (128, "64g_to_128g"),
)
# A cached behind count this old no longer says anything about how current the install is.
_BEHIND_MAX_AGE_S = 7 * _DAY_S
_CHANNELS = {"stable": "stable", "main": "main", "canary": "main"}
_MAIN_BRANCHES = frozenset({"main", "master"})
# Provider ids that always name a server on the user's own machine or network.
_LOCAL_PROVIDER_ALIASES = frozenset({"ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp", "lmstudio"})
_INHERITS_MAIN = frozenset({"", "auto", "main"})


def _sub(config: Any, key: str) -> Any:
    return config.get(key) if isinstance(config, dict) else None


def _safe(reader, default: str) -> str:
    try:
        return reader()
    except Exception:
        logger.debug("Shared-metrics install field %s unavailable", getattr(reader, "__name__", reader), exc_info=True)
        return default


def _version_info():
    from hermes_cli.version_info import get_version_info

    return get_version_info()


def _project_root() -> Path:
    from hermes_cli.config import get_project_root

    return get_project_root()


def release_channel(config: dict[str, Any]) -> str:
    """The channel this install follows: a package's baked channel, a source install's channel
    record, else the checkout's branch (``main`` for main/master, ``dev`` for any other)."""
    from hermes_cli.update_channel import _package_channel, _read_stamp, resolve_update_channel

    root = _project_root()
    resolved = resolve_update_channel(config, root)
    if resolved != "main" or _package_channel(_read_stamp(root)):
        return _CHANNELS.get(resolved, "unknown")
    branch = _version_info().branch
    if not branch:
        return "unknown"
    return "main" if branch in _MAIN_BRANCHES else "dev"


def version_age_bucket(now: float | None = None) -> str:
    """Age of the INSTALLED version, from its own commit date (stamp or checkout)."""
    committed = _version_info().commit_date
    if not isinstance(committed, int) or committed <= 0:
        return "unknown"
    age = max(0.0, (now or time.time()) - committed)
    return _bucket(age, _VERSION_AGE_THRESHOLDS, "gte_90d")


def behind_bucket(home: Path | None = None, now: float | None = None) -> str:
    """Commits/releases behind, from the update check's cached result for this exact revision."""
    from hermes_cli.update_channel import install_id
    from hermes_constants import get_hermes_home

    root = _project_root()
    cache_file = (home or get_hermes_home()) / "source-checks" / f"{install_id(root)}.json"
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "unknown"
    if not isinstance(cached, dict):
        return "unknown"
    status, identity, ts = cached.get("status"), cached.get("identity"), cached.get("ts")
    behind = status.get("behind") if isinstance(status, dict) else None
    head = identity.get("head") if isinstance(identity, dict) else None
    commit = _version_info().commit
    if (
        isinstance(behind, bool) or not isinstance(behind, int) or behind < 0
        or not isinstance(ts, (int, float)) or not 0 <= (now or time.time()) - ts < _BEHIND_MAX_AGE_S
        or (head and commit and head != commit)
    ):
        return "unknown"
    return contract.count_bucket(behind)


def ram_bucket() -> str:
    from hermes_platform.host.facts import ram_total_bytes

    total = ram_total_bytes()
    if not total:
        return "unknown"
    return _bucket(total * _RAM_NOMINAL_FACTOR / _GIB, _RAM_THRESHOLDS, "gte_128g")


def gpu_class() -> str:
    from hermes_platform.host.facts import gpu_class as host_gpu_class

    value = host_gpu_class()
    return value if value in contract.GPU_CLASSES else "unknown"


def _is_local_endpoint(base_url: Any) -> bool:
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    from agent.model_metadata import is_local_endpoint

    return is_local_endpoint(base_url)


def _custom_base_url(provider: str) -> Any:
    """A ``providers:`` / ``custom_providers:`` entry's endpoint, by ``custom:<name>`` or bare name
    (built-in ids shadow entries, as at runtime)."""
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider

    entry = _get_named_custom_provider(provider)
    return entry.get("base_url") if isinstance(entry, dict) else None


def _slot_is_local(slot: Any) -> bool:
    provider = _norm(_sub(slot, "provider"))
    if provider in _LOCAL_PROVIDER_ALIASES:
        return True
    if _is_local_endpoint(_sub(slot, "base_url")):
        return True
    from .shared_metrics_catalog import CUSTOM, _safe as catalog_safe, user_named_model_providers

    if provider in catalog_safe(user_named_model_providers) - {CUSTOM}:
        return True
    return provider not in _INHERITS_MAIN and _is_local_endpoint(_custom_base_url(provider))


def _aux_slot_routes_itself(slot: dict[str, Any]) -> bool:
    """An auxiliary task off the main model: a named provider, or (mirroring
    ``agent.auxiliary_client._resolve_task_provider_model``) a bare ``base_url`` + ``api_key``,
    which the runtime sends to that endpoint as ``custom`` even under ``provider: auto``."""
    if _norm(slot.get("provider")) not in _INHERITS_MAIN:
        return True
    return all(isinstance(slot.get(k), str) and slot[k].strip() for k in ("base_url", "api_key"))


def local_model_provider_used(config: dict[str, Any]) -> str:
    """``yes`` when the main model or any auxiliary task runs on a local/self-hosted server."""
    model = _sub(config, "model")
    auxiliary = _sub(config, "auxiliary")
    slots = [model] if isinstance(model, dict) else []
    if isinstance(auxiliary, dict):
        slots += [s for s in auxiliary.values() if isinstance(s, dict) and _aux_slot_routes_itself(s)]
    return "yes" if any(_slot_is_local(slot) for slot in slots) else "no"


def install_v4_snapshot_fields(config: dict[str, Any]) -> dict[str, str]:
    """The version-lag, channel and hardware dimensions of the daily snapshot."""
    fields = {
        "behind_bucket": _safe(behind_bucket, "unknown"),
        "gpu_class": _safe(gpu_class, "unknown"),
        "local_model_provider_used": _safe(lambda: local_model_provider_used(config), "no"),
        "ram_bucket": _safe(ram_bucket, "unknown"),
        "release_channel": _safe(lambda: release_channel(config), "unknown"),
        "version_age_bucket": _safe(version_age_bucket, "unknown"),
    }
    allowed = contract._INSTALL_V4_SNAPSHOT_DIMENSIONS
    return {k: v if v in allowed[k] else ("no" if k == "local_model_provider_used" else "unknown") for k, v in fields.items()}
