"""Local model runtime discovery helpers.

This module is intentionally read-only: it discovers local models exposed by
common desktop/local runtimes, but it never starts a server or loads a model.
Side-effectful bootstrap belongs behind an explicit opt-in config in a later
feature layer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

Backend = Literal["lmstudio", "ollama", "openai-compatible"]


@dataclass(frozen=True)
class LocalModel:
    """A model discovered from a local runtime."""

    backend: str
    id: str
    source: str
    kind: str = "llm"
    status: str = "available"
    name: str = ""
    params: str = ""
    architecture: str = ""
    size: str = ""
    device: str = ""
    digest: str = ""
    modified: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LocalRuntimeReport:
    """Discovery result for one backend."""

    backend: str
    available: bool
    models: list[LocalModel]
    source: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["models"] = [m.to_dict() for m in self.models]
        return data


_LMSTUDIO_VARIANT_SUFFIX = re.compile(r"\s+\(\d+\s+variants?\)\s*$", re.I)


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        text=True,
        capture_output=True,
        timeout=timeout,
        shell=False,
    )


def _split_table_row(line: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]


def _clean_lmstudio_model_id(raw: str) -> str:
    return _LMSTUDIO_VARIANT_SUFFIX.sub("", raw).strip()


def parse_lmstudio_ls_output(output: str, *, include_embeddings: bool = False) -> list[LocalModel]:
    """Parse `lms ls` table output.

    LM Studio prints section headers (LLM / EMBEDDING) followed by a whitespace
    table. We split on 2+ spaces so model IDs containing slashes or hyphens are
    preserved.
    """

    models: list[LocalModel] = []
    section = ""
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper in {"LLM", "EMBEDDING"}:
            section = upper.lower()
            continue
        if line.startswith("You have ") or set(line) <= {"─", "-", " "}:
            continue
        if upper.startswith("LLM "):
            # Header row: LLM PARAMS ARCH SIZE DEVICE
            section = "llm"
            continue
        if upper.startswith("EMBEDDING "):
            section = "embedding"
            continue

        if section not in {"llm", "embedding"}:
            continue
        if section == "embedding" and not include_embeddings:
            continue

        cols = _split_table_row(line)
        if not cols:
            continue
        model_id = _clean_lmstudio_model_id(cols[0])
        if not model_id:
            continue
        models.append(
            LocalModel(
                backend="lmstudio",
                id=model_id,
                source="lms ls",
                kind=section,
                name=model_id,
                params=cols[1] if len(cols) > 1 else "",
                architecture=cols[2] if len(cols) > 2 else "",
                size=cols[3] if len(cols) > 3 else "",
                device=cols[4] if len(cols) > 4 else "",
            )
        )
    return models


def discover_lmstudio_models(*, include_embeddings: bool = False, timeout: float = 10.0) -> LocalRuntimeReport:
    """Discover models installed in LM Studio via `lms ls`.

    This does not require the LM Studio OpenAI-compatible server to be running.
    """

    lms = shutil.which("lms") or shutil.which("lms.exe")
    if not lms:
        return LocalRuntimeReport(
            backend="lmstudio",
            available=False,
            models=[],
            error="lms CLI not found on PATH",
        )
    try:
        proc = _run([lms, "ls"], timeout)
    except Exception as exc:  # noqa: BLE001 - CLI discovery is best-effort
        return LocalRuntimeReport(
            backend="lmstudio",
            available=False,
            models=[],
            source=lms,
            error=str(exc),
        )
    if proc.returncode != 0:
        return LocalRuntimeReport(
            backend="lmstudio",
            available=True,
            models=[],
            source=lms,
            error=(proc.stderr or proc.stdout or "lms ls failed").strip(),
        )
    return LocalRuntimeReport(
        backend="lmstudio",
        available=True,
        models=parse_lmstudio_ls_output(proc.stdout, include_embeddings=include_embeddings),
        source=lms,
    )


def parse_ollama_list_output(output: str) -> list[LocalModel]:
    """Parse `ollama list` table output."""

    models: list[LocalModel] = []
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if not line or line.upper().startswith("NAME"):
            continue
        cols = _split_table_row(line)
        if not cols:
            continue
        model_id = cols[0]
        models.append(
            LocalModel(
                backend="ollama",
                id=model_id,
                source="ollama list",
                name=model_id,
                digest=cols[1] if len(cols) > 1 else "",
                size=cols[2] if len(cols) > 2 else "",
                modified=cols[3] if len(cols) > 3 else "",
                detail="  ".join(cols[4:]) if len(cols) > 4 else "",
            )
        )
    return models


def _ollama_manifest_name(manifest: Path, manifests_root: Path) -> Optional[str]:
    try:
        rel = manifest.relative_to(manifests_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 3:
        return None
    tag = parts[-1]
    namespace_parts = list(parts[1:-1])  # drop registry host
    if namespace_parts and namespace_parts[0] == "library":
        namespace_parts = namespace_parts[1:]
    if not namespace_parts:
        return None
    return f"{'/'.join(namespace_parts)}:{tag}"


def discover_ollama_manifest_models(manifests_root: Optional[Path] = None) -> list[LocalModel]:
    """Discover Ollama models by reading manifest paths without the daemon."""

    root = manifests_root or Path(os.getenv("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models"))) / "manifests"
    if not root.exists():
        return []
    models: list[LocalModel] = []
    for manifest in sorted(p for p in root.rglob("*") if p.is_file()):
        model_id = _ollama_manifest_name(manifest, root)
        if not model_id:
            continue
        models.append(
            LocalModel(
                backend="ollama",
                id=model_id,
                source="ollama manifest",
                name=model_id,
                detail=str(manifest),
            )
        )
    return models


def _dedupe_models(models: Iterable[LocalModel]) -> list[LocalModel]:
    seen: set[tuple[str, str]] = set()
    out: list[LocalModel] = []
    for model in models:
        key = (model.backend, model.id)
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    return out


def discover_ollama_models(*, timeout: float = 10.0) -> LocalRuntimeReport:
    """Discover local Ollama models via `ollama list`, with manifest fallback."""

    ollama = shutil.which("ollama") or shutil.which("ollama.exe")
    manifest_models = discover_ollama_manifest_models()
    if not ollama:
        return LocalRuntimeReport(
            backend="ollama",
            available=bool(manifest_models),
            models=manifest_models,
            source="manifest" if manifest_models else "",
            error="ollama CLI not found on PATH" if not manifest_models else "",
        )
    try:
        proc = _run([ollama, "list"], timeout)
    except Exception as exc:  # noqa: BLE001
        return LocalRuntimeReport(
            backend="ollama",
            available=bool(manifest_models),
            models=manifest_models,
            source=ollama if not manifest_models else "manifest",
            error=str(exc),
        )
    if proc.returncode == 0:
        live = parse_ollama_list_output(proc.stdout)
        return LocalRuntimeReport(
            backend="ollama",
            available=True,
            models=_dedupe_models([*live, *manifest_models]),
            source=ollama,
        )
    return LocalRuntimeReport(
        backend="ollama",
        available=bool(manifest_models),
        models=manifest_models,
        source=ollama if not manifest_models else "manifest",
        error=(proc.stderr or proc.stdout or "ollama list failed").strip(),
    )


def _models_endpoint(base_url: str) -> str:
    root = (base_url or "").strip().rstrip("/")
    if not root:
        return ""
    return root + "/models"


def discover_openai_compatible_models(
    *,
    base_url: str,
    api_key: str = "",
    timeout: float = 5.0,
) -> LocalRuntimeReport:
    """Discover models from a local OpenAI-compatible `/v1/models` endpoint."""

    url = _models_endpoint(base_url)
    if not url:
        return LocalRuntimeReport(
            backend="openai-compatible",
            available=False,
            models=[],
            error="base URL is required",
        )
    headers = {"User-Agent": "Hermes-Agent/local-runtime-discovery"}
    token = api_key.strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return LocalRuntimeReport(
            backend="openai-compatible",
            available=False,
            models=[],
            source=url,
            error=f"HTTP {exc.code}",
        )
    except Exception as exc:  # noqa: BLE001
        return LocalRuntimeReport(
            backend="openai-compatible",
            available=False,
            models=[],
            source=url,
            error=str(exc),
        )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return LocalRuntimeReport(
            backend="openai-compatible",
            available=False,
            models=[],
            source=url,
            error="malformed /models response",
        )
    models = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        models.append(
            LocalModel(
                backend="openai-compatible",
                id=model_id,
                source=url,
                name=model_id,
                detail=str(item.get("owned_by") or ""),
            )
        )
    return LocalRuntimeReport(
        backend="openai-compatible",
        available=True,
        models=_dedupe_models(models),
        source=url,
    )


def discover_local_models(
    *,
    backend: str = "all",
    base_url: str = "",
    api_key: str = "",
    include_embeddings: bool = False,
    timeout: float = 10.0,
) -> list[LocalRuntimeReport]:
    """Discover local models across selected backends."""

    normalized = (backend or "all").strip().lower()
    reports: list[LocalRuntimeReport] = []
    if normalized in {"all", "lmstudio"}:
        reports.append(discover_lmstudio_models(include_embeddings=include_embeddings, timeout=timeout))
    if normalized in {"all", "ollama"}:
        reports.append(discover_ollama_models(timeout=timeout))
    if normalized in {"openai", "openai-compatible"} or (normalized == "all" and base_url):
        reports.append(discover_openai_compatible_models(base_url=base_url, api_key=api_key, timeout=timeout))
    return reports
