"""Extras: the runtime features of the python venv.

Feature names ARE pyproject extra names. This table maps each extra to the
import that proves it (the anchor), so availability is one find_spec — no
package-manager machinery on the hot path. sync_venv([extra]) makes an extra
true; pm owns HOW (uv sync inside the venv package).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable


# extra name -> module that proves it is installed
ANCHORS: dict[str, str | tuple[str, ...]] = {
    "anthropic": "anthropic",
    "bedrock": "boto3",
    "vertex": "google.auth",
    "azure-identity": "azure.identity",
    "exa": "exa_py",
    "firecrawl": "firecrawl",
    "parallel-web": "parallel",
    "ddgs": "ddgs",
    "otlp": "opentelemetry.sdk",
    "langfuse": "langfuse",
    "mistral": "mistralai",
    "edge-tts": "edge_tts",
    "neutts": "neutts",
    "kittentts": ("kittentts", "soundfile"),
    "piper": "piper",
    "tts-premium": "elevenlabs",
    "voice": "faster_whisper",
    "stt-whisper": "faster_whisper",
    "audio-io": ("sounddevice", "numpy"),
    "silk": "pilk",
    "wake": "pyopen_wakeword",
    "wake-openwakeword": "pyopen_wakeword",
    "wake-sherpa": "sherpa_onnx",
    "wake-porcupine": "pvporcupine",
    "fal": "fal_client",
    "honcho": "honcho",
    "supermemory": "supermemory",
    "mem0": "mem0",
    "messaging": "telegram",
    "telegram": "telegram",
    "discord": "discord",
    "slack": "slack_bolt",
    "matrix": ("mautrix", "asyncpg", "aiosqlite", "markdown", "aiohttp_socks"),
    "dingtalk": "dingtalk_stream",
    "feishu": "lark_oapi",
    "wecom": "defusedxml",
    "teams": "microsoft_teams.apps",
    "modal": "modal",
    "daytona": "daytona",
    "vercel": "vercel",
    "google": ("googleapiclient", "google.auth", "google_auth_oauthlib.flow", "google_auth_httplib2"),
    "google-chat": "google.cloud.pubsub_v1",
    "google-meet": ("playwright.sync_api", "websockets"),
    "youtube": "youtube_transcript_api",
    "acp": "acp",
    "web": "fastapi",
    "doc-extract": "anydoc",
    "computer-use": "mcp",
    "trace-upload": "huggingface_hub",
    # Pillow resize recovery for vision tools (the `vision` extra is a no-op
    # back-compat alias — Pillow is core — but ensure_import("vision") must
    # still resolve an anchor so availability checks work).
    "vision": "PIL",
}


def _anchors(extra: str) -> tuple[str, ...]:
    got = ANCHORS.get(extra, extra.replace("-", "_"))
    return got if isinstance(got, tuple) else (got,)


def _importable(anchor: str) -> bool:
    """An anchor already present in sys.modules counts even without a
    findable spec — tests fake SDKs by inserting modules there, and the
    caller's import right after this check resolves the same way."""
    import sys

    if anchor in sys.modules:
        return True
    try:
        return importlib.util.find_spec(anchor) is not None
    except (ImportError, ValueError):
        return False


def available(extra: str) -> bool:
    """Fast, side-effect-free: are all of this extra's anchors importable?
    A platform-gated extra whose gate excludes this machine reads as
    unavailable — the anchors are absent by design, not by accident."""
    if not extra_supported(extra):
        return False
    return all(_importable(a) for a in _anchors(extra))


def _platform_gates() -> dict[str, str]:
    """The [tool.hermes.extras-platforms] table from pyproject.toml:
    extra -> PEP 508 marker string. Cached per process."""
    global _PLATFORM_GATES
    if _PLATFORM_GATES is not None:
        return _PLATFORM_GATES
    import tomllib

    from pm.paths import repo_root

    gates: dict[str, str] = {}
    try:
        with (repo_root() / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        table = data.get("tool", {}).get("hermes", {}).get("extras-platforms", {})
        if isinstance(table, dict):
            gates = {str(k): str(v) for k, v in table.items()}
    except (OSError, ValueError):
        pass
    _PLATFORM_GATES = gates
    return gates


_PLATFORM_GATES: dict[str, str] | None = None


def extra_supported(extra: str, *, environment: dict[str, str] | None = None,
                    importable: Callable[[str], bool] | None = None) -> bool:
    """Is this extra installable on THIS platform? True when the extra
    carries no gate, or its marker matches the running platform. An
    extra that IS present on this machine (anchors importable) is always
    supported — an installed override beats the table (dev machines,
    hand-synced venvs)."""
    if all((importable or _importable)(a) for a in _anchors(extra)):
        return True
    marker = _platform_gates().get(extra)
    if marker is None:
        return True
    import os
    import platform
    import sys

    if environment is None:
        environment = {
            "sys_platform": sys.platform,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "os_name": os.name,
        }
    try:
        from packaging.markers import Marker
    except ImportError:
        # The historical update takeover runs PM in the OLD venv's
        # interpreter, which need not ship `packaging`. PM's runtime does.
        return _evaluate_in_runtime(marker, environment)
    try:
        return bool(Marker(marker).evaluate(environment=environment))
    except Exception:
        # A malformed marker must never brick availability — treat as
        # ungated and let the resolver be the authority.
        return True


def _evaluate_in_runtime(marker: str, environment: dict[str, str]) -> bool:
    import json
    import subprocess

    from pm.runtime import runtime_command, runtime_environment

    try:
        command = runtime_command(Path(__file__).with_name("_marker_eval.py"),
                                  [marker, json.dumps(environment)])
        result = subprocess.run(command, env=runtime_environment(), capture_output=True,
                                text=True, timeout=60)
    except Exception:
        return True  # same fallback as a malformed marker: the resolver decides
    if result.returncode != 0:
        return True
    return result.stdout.strip() == "1"


def install_hint(extra: str) -> str:
    """The one command users are told to run for a missing extra."""
    return f"hermes pm install --extra {extra}"


def ensure_import(extra: str) -> None:
    """Make an extra available: no-op when the anchor imports, otherwise
    sync the venv with the extra enabled. Raises InstallError on failure
    — including when a platform gate excludes this machine."""
    if available(extra):
        return
    if not extra_supported(extra):
        from pm.package import InstallError

        marker = _platform_gates().get(extra, "")
        raise InstallError(
            "venv",
            f"extra {extra!r} is not supported on this platform "
            f"(gate: {marker!r}); the adapter degrades without it",
        )
    import sys
    from pm.package import InstallError

    # prompt_toolkit already owns stdin during a CLI turn; never read from it.
    app_running = False
    if "prompt_toolkit.application.current" in sys.modules:
        from prompt_toolkit.application.current import get_app_or_none

        app_running = bool(getattr(get_app_or_none(), "is_running", False))
    if not app_running and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            answer = input(f"\nThis needs Hermes' optional {extra!r} feature, which isn't installed yet.\n"
                           "Install it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in {"y", "yes"}:
            raise InstallError("venv", f"installation of extra {extra!r} declined")
    from pm.client import sync_venv

    sync_venv([extra])
    # The sync published a new generation. Swap this process onto it when nothing
    # already imported would change underneath it (adopt_selected); otherwise only
    # a restart can load it.
    from pm.environments import selected_venv, site_packages
    from pm.environments_adopt import adopt_selected, restart_needed
    from pm.paths import repo_root, runtime_facts_path
    import sys
    from pathlib import Path

    if runtime_facts_path().is_file():
        root = repo_root()
        adopt_selected(root)
        selected = site_packages(selected_venv(root)).resolve()
        if selected not in {Path(entry).resolve() for entry in sys.path}:
            reason = restart_needed(root) or "this process does not run from the install's dependency environment"
            raise InstallError("venv", f"{extra} installed; restart Hermes to activate it ({reason})")


def ensure_and_bind(extra, importer, target_globals) -> bool:
    """ensure_import + rebind module-level names after a mid-process install.
    importer returns {name: value}; bound into target_globals on success."""
    try:
        ensure_import(extra)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("extra %r unavailable: %s", extra, exc)
        return False
    try:
        bindings = importer()
    except ImportError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "import after installing %r failed: %s", extra, exc
        )
        return False
    target_globals.update(bindings)
    return True


def missing(extra: str) -> tuple[str, ...]:
    return tuple(a for a in _anchors(extra) if not _importable(a))


def _installed_in(site_packages: Path, anchor: str) -> bool:
    """Is ``anchor`` installed under a site-packages we must not import from?"""
    *parents, leaf = anchor.split(".")
    parent = site_packages.joinpath(*parents)
    if (parent / leaf).is_dir():
        return True
    # A top-level module can be a plain file or a compiled extension.
    return any(entry.name.rsplit(".", 1)[-1] in {"py", "so", "pyd"}
               for entry in parent.glob(f"{leaf}.*"))


def legacy_selection(project_root: Path) -> list[str]:
    """The extras PM's first generation selects when it replaces a main-era venv.

    Main-era installers selected ``[all]`` and then lazily installed opt-in
    extras (FAL, messaging SDKs, ...) into the checkout's own venv, with no
    ledger. Selecting only ``[all]`` drops those, and the first launch after
    the migration asks to reinstall a feature that already worked. The old
    venv's site-packages is read, never imported: the migrating process may
    not run from it.
    """
    root = Path(project_root)
    trees = [tree for venv in (root / "venv", root / ".venv")
             for tree in (*venv.glob("lib/python*/site-packages"), venv / "Lib" / "site-packages")
             if tree.is_dir()]
    carried = sorted(
        # An umbrella extra shares its anchor with one member; carrying it
        # would install every sibling the user never chose.
        extra for extra in ANCHORS if extra not in {"messaging", "voice", "wake"}
        # PM refuses a gated extra outside its platform even if a hand-synced venv carried it.
        # Installed first: judging a gate may cost a PM-runtime subprocess.
        if any(all(_installed_in(tree, anchor) for anchor in _anchors(extra)) for tree in trees)
        and extra_supported(extra, importable=lambda _anchor: False)
    )
    return ["all", *carried]
