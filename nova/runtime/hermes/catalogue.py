"""The channel catalogue, read from the runtime rather than kept by hand.

NOVA used to ship a hand-written tuple of six providers. The runtime bundles twenty-two
platform plugins, each with a ``plugin.yaml`` that already states its label, its
description, and every credential variable it needs — including which of them are secrets.
Maintaining a second, smaller list beside that was guaranteed to drift, and had: a Control
Centre that offered Telegram and nothing else is what prompted this module.

**Discovery, not invention.** Every entry here comes from a manifest on disk. Nothing is
added from memory, and a provider NOVA has notes about but the runtime does not bundle does
not appear — the runtime is the source of truth for what can actually connect.

**Why the manifest is read twice.** ``hermes_cli.plugins_manifest.parse_manifest_file`` is
the runtime's own parser and decides the one thing that must not be guessed: whether a
plugin is ``kind: platform``. It also normalises name, description and ``requires_env``.
What it drops is ``label`` and ``optional_env``, which a connect-a-channel form needs. So
the manifest is parsed by the runtime for the authoritative fields and re-read as YAML for
the two presentational ones. The alternative was widening ``PluginManifest`` — a change to
Hermes core for a field only NOVA reads.

This module is in ``nova/runtime/hermes/`` because it imports the runtime, and that is the
only package allowed to (``tests/platform/test_boundaries.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

#: Manifest filenames, in the order the runtime looks for them.
MANIFEST_NAMES = ("plugin.yaml", "plugin.yml")


def platforms_dir() -> Optional[Path]:
    """Where the runtime keeps its bundled platform plugins.

    Derived from the runtime package's own location, the same way
    ``gateway.config.Platform._scan_bundled_plugin_platforms`` does it, so an installed
    Hermes and a source checkout both resolve correctly.
    """
    try:
        import gateway
    except Exception:  # pragma: no cover — runtime not importable
        return None
    candidate = Path(gateway.__file__).resolve().parent.parent / "plugins" / "platforms"
    return candidate if candidate.is_dir() else None


def _env_entry(raw: Any, *, required: bool) -> Optional[dict[str, Any]]:
    """One credential variable, normalised.

    A manifest may write a variable as a bare string or as a mapping. Both are accepted
    because both appear in the bundled manifests.

    ``secret`` decides whether the Control Centre masks the value. It defaults to **True**
    for a bare string: a manifest that did not say is not a manifest saying "safe to show",
    and the cost of masking a non-secret is cosmetic where the reverse is a leak.
    """
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            return None
        return {
            "name": name, "description": "", "prompt": "", "url": "",
            "secret": True, "required": required,
        }
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "description": str(raw.get("description") or "").strip(),
        "prompt": str(raw.get("prompt") or "").strip(),
        "url": str(raw.get("url") or "").strip(),
        "secret": bool(raw.get("password", True)),
        "required": required,
    }


def _env_list(raws: Any, *, required: bool) -> list[dict[str, Any]]:
    if not isinstance(raws, (list, tuple)):
        return []
    out = []
    for raw in raws:
        entry = _env_entry(raw, required=required)
        if entry is not None:
            out.append(entry)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _manifest_path(plugin_dir: Path) -> Optional[Path]:
    for name in MANIFEST_NAMES:
        candidate = plugin_dir / name
        if candidate.is_file():
            return candidate
    return None


def _is_platform(manifest_file: Path, plugin_dir: Path) -> bool:
    """Whether the runtime considers this plugin a gateway platform.

    Asked of the runtime's parser rather than by reading ``kind:`` directly: the parser
    applies defaulting and path-based inference this module should not reimplement.
    """
    try:
        from hermes_cli.plugins_manifest import parse_manifest_file
    except Exception:  # pragma: no cover — runtime not importable
        return False
    try:
        parsed = parse_manifest_file(manifest_file, plugin_dir, "bundled", "platforms")
    except Exception:
        return False
    return parsed is not None and getattr(parsed, "kind", "") == "platform"


#: Capability -> the adapter methods that implement it natively, and what the runtime's base
#: adapter does when none is overridden (``gateway/platforms/base.py::BasePlatformAdapter``).
#: Read from the adapter's source rather than imported: the platform client libraries are
#: not installed everywhere NOVA runs, and what the runtime ships is the source.
NATIVE_METHODS: dict[str, tuple[tuple[str, ...], str]] = {
    "images": (
        ("send_image", "send_image_file", "send_multiple_images"),
        "sends the image URL as text",
    ),
    "documents": (("send_document",), "sends a notice that the file could not be delivered"),
    "voice": (("send_voice",), "sends a notice instead of audio"),
    "video": (("send_video",), "sends a notice instead of video"),
    "message_edits": (("edit_message",), "every update is a new message; no streamed edits"),
    "message_delete": (("delete_message",), "stale previews are left in place"),
    "typing_indicator": (("send_typing",), "no typing indicator"),
    "draft_streaming": (("supports_draft_streaming", "send_draft"), "no native draft stream"),
    "approval_buttons": (
        ("_send_exec_approval_prompt",),
        "approvals arrive as a plain-text /approve prompt",
    ),
}

#: Capabilities the base adapter declares as class attributes, false by default.
NATIVE_FLAGS: dict[str, str] = {"code_blocks": "supports_code_blocks"}

BASE_ADAPTER = "BasePlatformAdapter"


def adapter_capabilities(plugin_dir: Path) -> dict[str, dict[str, Any]]:
    """What a platform adapter implements natively, read from its source. Empty: unreadable.

    ``supported`` is True only where the adapter class itself overrides the method (or sets
    the flag); otherwise False with the base adapter's fallback named, which is what a
    customer on that channel actually gets.
    """
    import re

    # Every top-level class in the plugin package, so capabilities a mixin contributes
    # (Discord and WeCom keep their media methods in one) count for the adapter that
    # inherits them. A line scan, not an ``ast`` parse: parsing ~70k lines of adapters
    # took seconds, and the runtime's source is formatted, so class headers, methods and
    # flags sit at fixed indents.
    header = re.compile(r"^class\s+(\w+)\s*(?:\((.*?)\))?\s*:")
    method = re.compile(r"^    (?:async\s+)?def\s+(\w+)\s*\(")
    flag = re.compile(r"^    (\w+)\s*(?::[^=]*)?=\s*True\b")
    classes: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for source in sorted(plugin_dir.rglob("*.py")):
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError):
            continue
        current: Optional[str] = None
        pending = ""
        for line in lines:
            if pending:
                pending += " " + line.strip()
                if not pending.rstrip().endswith(":"):
                    continue
                line, pending = pending, ""
            elif line.startswith("class ") and not line.rstrip().endswith(":"):
                pending = line.strip()
                continue
            match = header.match(line)
            if match:
                current = match.group(1)
                bases = {
                    part.strip().split(".")[-1]
                    for part in (match.group(2) or "").split(",")
                    if part.strip() and "=" not in part
                }
                classes[current] = (bases, set(), set())
                continue
            if line and not line[0].isspace() and not line.startswith(("#", "@", ")")):
                current = None
                continue
            if current is None:
                continue
            found_method = method.match(line)
            if found_method:
                classes[current][1].add(found_method.group(1))
                continue
            found_flag = flag.match(line)
            if found_flag:
                classes[current][2].add(found_flag.group(1))

    def collect(name: str, seen: set[str]) -> tuple[set[str], set[str]]:
        if name in seen or name not in classes:
            return set(), set()
        seen.add(name)
        bases, own_methods, own_flags = classes[name]
        methods, flags = set(own_methods), set(own_flags)
        for base in bases - {BASE_ADAPTER}:
            more_methods, more_flags = collect(base, seen)
            methods |= more_methods
            flags |= more_flags
        return methods, flags

    methods: set[str] = set()
    flags: set[str] = set()
    adapters = [name for name, (bases, _, _) in classes.items() if BASE_ADAPTER in bases]
    if not adapters:
        return {}
    for name in adapters:
        adapter_methods, adapter_flags = collect(name, set())
        methods |= adapter_methods
        flags |= adapter_flags

    implementation = f"plugins/platforms/{plugin_dir.name}/"
    out: dict[str, dict[str, Any]] = {}
    for capability, (names, fallback) in NATIVE_METHODS.items():
        native = sorted(set(names) & methods)
        out[capability] = (
            {"supported": True, "note": f"native: {', '.join(native)} in {implementation}"}
            if native
            else {"supported": False, "note": f"not native; the runtime {fallback}"}
        )
    for capability, flag in NATIVE_FLAGS.items():
        out[capability] = (
            {"supported": True, "note": f"{flag} = True in {implementation}"}
            if flag in flags
            else {"supported": False, "note": f"{flag} is not set; rendered as plain text"}
        )
    return out


def discover() -> tuple[dict[str, Any], ...]:
    """Every bundled platform plugin, as plain dictionaries.

    Sorted by label so the Control Centre's list is stable between calls. An unreadable or
    non-platform plugin is skipped rather than surfaced as a broken row: the catalogue
    answers "what can this deployment connect?", and something that will not load cannot.
    """
    root = platforms_dir()
    if root is None:
        return ()

    found: list[dict[str, Any]] = []
    for plugin_dir in sorted(root.iterdir()):
        if not plugin_dir.is_dir() or not (plugin_dir / "__init__.py").is_file():
            continue
        manifest_file = _manifest_path(plugin_dir)
        if manifest_file is None or not _is_platform(manifest_file, plugin_dir):
            continue

        raw = _read_yaml(manifest_file)
        provider_id = plugin_dir.name.lower()
        required = _env_list(raw.get("requires_env"), required=True)
        optional = _env_list(raw.get("optional_env"), required=False)
        found.append(
            {
                "id": provider_id,
                "label": str(raw.get("label") or "").strip() or provider_id.replace("_", " ").title(),
                "description": " ".join(str(raw.get("description") or "").split()),
                "version": str(raw.get("version") or "").strip(),
                "author": str(raw.get("author") or "").strip(),
                "required_env": required,
                "optional_env": optional,
                # Presentational only. A provider needing no declared credential is
                # reported as such rather than as "ready": whether it can actually reach
                # anything is a runtime question this catalogue does not answer.
                "credential_count": len(required),
                "implementation": f"plugins/platforms/{provider_id}/",
                "source": "plugin-manifest",
            }
        )
    found.sort(key=lambda entry: entry["label"].lower())
    return tuple(found)


def discover_with_capabilities() -> tuple[dict[str, Any], ...]:
    """:func:`discover`, with each platform's native capabilities read from its adapter.

    Separate from :func:`discover` so validation paths that only need ids never read
    adapter source; the channel catalogue that shows capabilities caches the result.
    """
    root = platforms_dir()
    rows = []
    for row in discover():
        caps = adapter_capabilities(root / row["id"]) if root is not None else {}
        rows.append({**row, "capabilities": caps})
    return tuple(rows)


def discover_ids() -> tuple[str, ...]:
    """Just the ids, for callers validating a declaration against what exists."""
    return tuple(entry["id"] for entry in discover())
