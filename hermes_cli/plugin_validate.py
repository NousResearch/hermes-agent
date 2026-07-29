"""``hermes plugins validate`` — admission checks for a plugin directory.

This is the command the plugin-catalog admission CI (and the
``.github/actions/plugin-validate`` composite action) runs against a
candidate plugin. It performs static manifest checks plus a
subprocess-isolated capability probe: the plugin is imported and its
``register(ctx)`` called against a minimal recording stub context in a
scratch child process (with a throwaway ``HERMES_HOME``), so a crashing or
malicious plugin cannot take down the CLI, and the *actually registered*
tools/hooks/middleware are compared against the manifest's declared
``provides_*`` lists.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CONFIG_TYPES = {"str", "bool", "int"}
_PROBE_TIMEOUT = 30
_PROBE_SENTINEL = "HERMES_VALIDATE_JSON:"


@dataclass
class ValidationReport:
    """Result of validating one plugin directory."""

    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[str]:
        return [detail or name for name, ok, detail in self.checks if not ok]

    @property
    def ok(self) -> bool:
        return all(ok for _name, ok, _detail in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": name, "ok": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
            "warnings": list(self.warnings),
        }


# ─── Static checks ───────────────────────────────────────────────────────────


def _requires_hermes_spec_valid(spec: str) -> bool:
    """Strictly validate a ``requires_hermes`` spec.

    Unlike :func:`hermes_cli.plugins._version_satisfies` (permissive at load
    time), validation REJECTS clauses whose version segment doesn't parse —
    a typo'd spec should fail admission, not silently gate nothing.
    """
    from hermes_cli.plugins import _VERSION_COMPARATOR_RE, _version_tuple

    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        m = _VERSION_COMPARATOR_RE.match(clause)
        target = m.group(2) if m else clause
        if _version_tuple(target) is None:
            return False
    return True


def _check_manifest_fields(report: ValidationReport, manifest: dict) -> None:
    missing = [
        f for f in ("name", "version", "description") if not manifest.get(f)
    ]
    if missing:
        report.add(
            "manifest fields",
            False,
            f"plugin.yaml missing required field(s): {', '.join(missing)}",
        )
    else:
        report.add("manifest fields", True, "name, version, description present")


def _check_requires_hermes(report: ValidationReport, manifest: dict) -> None:
    spec = str(manifest.get("requires_hermes") or "").strip()
    if not spec:
        report.add("requires_hermes", True, "not declared")
        return
    if _requires_hermes_spec_valid(spec):
        report.add("requires_hermes", True, f"spec {spec!r} parses")
    else:
        report.add(
            "requires_hermes",
            False,
            f"requires_hermes spec {spec!r} does not parse "
            "(expected e.g. \">=0.19\" or \">=0.19, <1.0\")",
        )


def _check_config_spec(report: ValidationReport, manifest: dict) -> None:
    raw = manifest.get("config")
    if raw in (None, [], {}):
        report.add("config spec", True, "not declared")
        return
    problems: List[str] = []
    if not isinstance(raw, list):
        problems.append("config: must be a list of mappings")
    else:
        for i, item in enumerate(raw):
            if not isinstance(item, dict) or not item.get("key"):
                problems.append(f"config[{i}]: must be a mapping with a 'key'")
                continue
            typ = item.get("type")
            if typ is not None and str(typ) not in _CONFIG_TYPES:
                problems.append(
                    f"config[{i}] ({item['key']}): type must be one of "
                    f"{'/'.join(sorted(_CONFIG_TYPES))}"
                )
            secret = item.get("secret")
            if secret is not None and not isinstance(secret, bool):
                problems.append(
                    f"config[{i}] ({item['key']}): secret must be a boolean"
                )
    if problems:
        report.add("config spec", False, "; ".join(problems))
    else:
        report.add("config spec", True, "shape valid")


def _check_requires_env(report: ValidationReport, manifest: dict) -> None:
    raw = manifest.get("requires_env") or []
    problems: List[str] = []
    if not isinstance(raw, list):
        problems.append("requires_env: must be a list")
        raw = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = str(entry.get("name") or "")
        else:
            problems.append(f"requires_env[{i}]: must be a string or mapping")
            continue
        if not _UPPER_SNAKE_RE.match(name):
            problems.append(
                f"requires_env[{i}]: {name!r} is not UPPER_SNAKE_CASE"
            )
    if problems:
        report.add("requires_env", False, "; ".join(problems))
    else:
        report.add("requires_env", True, "all entries UPPER_SNAKE")


# ─── Capability probe (subprocess-isolated) ──────────────────────────────────

# Self-contained harness run in a scratch child process. Imports the plugin
# module using the same file-location mechanics PluginManager uses, calls
# register() against a recording stub ctx, and prints a sentinel-prefixed
# JSON line of what was actually registered. Deliberately imports NOTHING
# from hermes so a hostile plugin only sees a bare interpreter.
_PROBE_SCRIPT = r"""
import importlib.util
import json
import sys

plugin_dir = sys.argv[1]
sentinel = sys.argv[2]

recorded = {"tools": [], "hooks": [], "middleware": [], "commands": []}


class RecordingContext:
    plugin_config = {}
    profile_name = "default"

    def register_tool(self, name, *args, **kwargs):
        recorded["tools"].append(str(name))

    def register_hook(self, hook_name, callback):
        recorded["hooks"].append(str(hook_name))

    def register_middleware(self, kind, callback):
        recorded["middleware"].append(str(kind))

    def register_command(self, name, *args, **kwargs):
        recorded["commands"].append(str(name))

    def register_cli_command(self, name, *args, **kwargs):
        recorded["commands"].append(str(name))

    def __getattr__(self, _name):
        # Any other registration surface (platforms, providers, skills,
        # context engines, ...) is accepted as a no-op — the probe only
        # audits the declared-capability categories.
        def _noop(*args, **kwargs):
            return None

        return _noop


def emit(payload):
    print(sentinel + json.dumps(payload))


try:
    spec = importlib.util.spec_from_file_location(
        "hermes_validate_probe_plugin",
        plugin_dir + "/__init__.py",
        submodule_search_locations=[plugin_dir],
    )
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [plugin_dir]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
except Exception as exc:
    emit({"error": "import failed: %s" % exc})
    sys.exit(0)

register = getattr(module, "register", None)
if register is None:
    emit({"error": "no register() function"})
    sys.exit(0)

try:
    register(RecordingContext())
except Exception as exc:
    emit({"error": "register() raised: %s" % exc})
    sys.exit(0)

emit(recorded)
"""


def _run_capability_probe(plugin_dir: Path) -> Tuple[Optional[dict], str]:
    """Run the recording probe in a scratch subprocess.

    Returns ``(recorded, error)`` — exactly one is meaningful: *recorded*
    is the ``{tools, hooks, middleware, commands}`` dict on success, and
    *error* is a human-readable failure description otherwise.
    """
    with tempfile.TemporaryDirectory(prefix="hermes-validate-") as scratch:
        env = dict(os.environ)
        env["HERMES_HOME"] = scratch
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _PROBE_SCRIPT,
                    str(plugin_dir),
                    _PROBE_SENTINEL,
                ],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return None, f"capability probe timed out after {_PROBE_TIMEOUT}s"

    payload: Optional[dict] = None
    for line in (result.stdout or "").splitlines():
        if line.startswith(_PROBE_SENTINEL):
            try:
                payload = json.loads(line[len(_PROBE_SENTINEL):])
            except json.JSONDecodeError:
                payload = None

    if payload is None:
        err = (result.stderr or "").strip()
        return None, (
            "capability probe produced no result "
            f"(exit {result.returncode})" + (f": {err}" if err else "")
        )
    if "error" in payload:
        return None, str(payload["error"])
    return payload, ""


def _declared_list(manifest: dict, key: str) -> List[str]:
    raw = manifest.get(key) or []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str)]


def _check_capabilities(
    report: ValidationReport, manifest: dict, plugin_dir: Path
) -> Optional[dict]:
    """Probe actual registrations and diff against declared capabilities.

    Returns the recorded dict (for the built-in collision check) or None
    when the probe failed / was skipped.
    """
    if not (plugin_dir / "__init__.py").is_file():
        report.warn(
            "no __init__.py — capability probe skipped (manifest-only plugin)"
        )
        report.add("capability probe", True, "skipped (no __init__.py)")
        return None

    recorded, error = _run_capability_probe(plugin_dir)
    if recorded is None:
        report.add("capability probe", False, error)
        return None
    report.add("capability probe", True, "register() ran in isolation")

    for kind, manifest_key in (
        ("tools", "provides_tools"),
        ("hooks", "provides_hooks"),
        ("middleware", "provides_middleware"),
    ):
        declared = set(_declared_list(manifest, manifest_key))
        actual = set(recorded.get(kind) or [])
        undeclared = sorted(actual - declared)
        unregistered = sorted(declared - actual)
        if undeclared:
            report.add(
                f"declared {kind}",
                False,
                f"undeclared {kind} registered (not in {manifest_key}): "
                f"{', '.join(undeclared)}",
            )
        else:
            report.add(f"declared {kind}", True, "matches registrations")
        if unregistered:
            report.warn(
                f"{manifest_key} declares {', '.join(unregistered)} "
                f"but register() did not register them"
            )
    return recorded


def _builtin_tool_names() -> List[str]:
    """Return the built-in tool registry names (discovery-timing safe).

    ``tools.registry`` starts empty — built-in tool modules self-register on
    import, so we must run ``discover_builtin_tools()`` first (idempotent;
    see the AGENTS.md discover_plugins timing pitfall).
    """
    try:
        from tools.registry import discover_builtin_tools, registry

        discover_builtin_tools()
        return list(registry.get_all_tool_names())
    except Exception:
        return []


def _check_builtin_collisions(
    report: ValidationReport, manifest: dict, recorded: Optional[dict]
) -> None:
    candidate_tools = set(_declared_list(manifest, "provides_tools"))
    if recorded:
        candidate_tools.update(recorded.get("tools") or [])
    if not candidate_tools:
        report.add("built-in tool collisions", True, "no tools to check")
        return
    builtin = set(_builtin_tool_names())
    collisions = sorted(candidate_tools & builtin)
    if collisions:
        report.add(
            "built-in tool collisions",
            False,
            "tool name(s) collide with built-in tools: "
            f"{', '.join(collisions)}",
        )
    else:
        report.add("built-in tool collisions", True, "no collisions")


# ─── Entry point ─────────────────────────────────────────────────────────────


def validate_plugin_dir(plugin_dir: Path) -> ValidationReport:
    """Run every admission check against *plugin_dir* and return the report."""
    report = ValidationReport()
    plugin_dir = Path(plugin_dir)

    if not plugin_dir.is_dir():
        report.add(
            "plugin directory", False, f"{plugin_dir} is not a directory"
        )
        return report

    manifest_file = plugin_dir / "plugin.yaml"
    if not manifest_file.is_file():
        manifest_file = plugin_dir / "plugin.yml"
    if not manifest_file.is_file():
        report.add("manifest", False, "no plugin.yaml in the plugin directory")
        return report

    import yaml

    try:
        manifest = yaml.safe_load(
            manifest_file.read_text(encoding="utf-8")
        )
    except Exception as exc:
        report.add("manifest", False, f"plugin.yaml failed to parse: {exc}")
        return report
    if not isinstance(manifest, dict):
        report.add("manifest", False, "plugin.yaml must be a mapping")
        return report
    report.add("manifest", True, "plugin.yaml parses")

    _check_manifest_fields(report, manifest)
    _check_requires_hermes(report, manifest)
    _check_config_spec(report, manifest)
    _check_requires_env(report, manifest)
    recorded = _check_capabilities(report, manifest, plugin_dir)
    _check_builtin_collisions(report, manifest, recorded)
    return report
