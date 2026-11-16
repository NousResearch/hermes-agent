"""Cron capability preflight — will a job actually get the toolsets it needs?

A cron job's agent receives its per-job ``enabled_toolsets`` (with enabled MCP
servers layered on), minus the toolsets cron always strips (``messaging``,
``clarify``, usually ``cronjob``), capped to the job owner's CURRENT RBAC role
(cron/rbac_ceiling). Every one of those steps is silent, so a job could be
created naming ``messaging`` — which cron removes — or relying on a write
toolset its owner's role lacks, report success at creation, and then spend
every run discovering it cannot do its job (the Ready-for-Staging auto-merger,
2026-09-16: 17 events, 0 merges, 17 undelivered blocker notices, every run
``ok``).

This module evaluates that pipeline up front so the surfaces agree:

* ``cronjob`` create/update (tools/cronjob_tools.py) REJECTS a job whose
  needed toolsets would be unavailable at run time, and echoes the effective
  toolsets so the agent can't mis-describe what the job gets.
* The scheduler's pre-dispatch preflight (cron/scheduler.py) blocks a run —
  alert-once, no LLM spend — when a needed toolset stopped resolving (owner
  demoted, job transferred to a narrower role, config edited) or a script
  imports a module the gateway's Python lacks.
* ``ownership transfer`` of a ``cron:`` item refuses (unless confirmed) when
  the new owner's role would strip what the job needs.

"Needed" means the job's ``required_toolsets`` field, ``requires_toolsets`` in
the ``automation.yaml`` of the job's ``workdir`` (an automation bundle), and
every toolset the job names explicitly in ``enabled_toolsets`` — naming a
toolset the job can never receive is always a misconfiguration. (Measured on
the VM 2026-09-23: that rule flagged exactly the two broken jobs.) Cron agents
never receive ``terminal``/``code_execution`` (fork policy): deterministic
steps live in the job's pre-run ``script`` / ``post_script``.

Everything here fails open: an internal error yields an empty problem list,
never a blocked job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Iterable, List, Optional

logger = logging.getLogger(__name__)

AUTOMATION_MANIFEST = "automation.yaml"

# What to use instead when a job asks for a toolset cron always removes.
CRON_STRIPPED_HINTS = {
    "messaging": (
        "cron never has send_message — post to Slack with the `slack_post` "
        "toolset (slack_post_thread; omit thread_ts for a new root message)"
    ),
    "clarify": (
        "a cron run cannot ask questions — decide up front in the prompt, or "
        "have the job post the question with `slack_post` and stop"
    ),
    "cronjob": (
        "cron agents may not manage cron jobs unless "
        "`cron.allow_agent_scheduling: true` is set"
    ),
    "terminal": (
        "unattended agents never get a shell — put deterministic steps in the "
        "job's pre-run `script` (collect) or `post_script` (apply the agent's "
        "plan file); a super admin attaches those"
    ),
    "code_execution": (
        "unattended agents never get a shell — put deterministic steps in the "
        "job's pre-run `script` or `post_script`; a super admin attaches those"
    ),
}


@dataclass
class CapabilityReport:
    """Outcome of evaluating one job's toolsets.

    ``effective`` is the sorted list the agent will receive, or ``None`` when
    it would receive the full default set (no per-job list and no resolvable
    cron platform config). ``problems`` holds one human-readable line per
    unmet requirement; ``missing`` the corresponding toolset names.
    ``gated_unacked`` lists approval-gated tools the agent can reach but that
    are not in ``unattended_approved_tools`` — they will be BLOCKED at run
    time (the approval gate denies them headless), which is correct but
    usually not what the author meant.
    """

    effective: Optional[List[str]] = None
    required: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    gated_unacked: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _clean_names(values: Iterable) -> List[str]:
    out: List[str] = []
    for value in values or []:
        name = str(value or "").strip().lower()
        if name and name not in out:
            out.append(name)
    return out


def automation_required_toolsets(workdir: Optional[str]) -> List[str]:
    """``requires_toolsets`` from ``<workdir>/automation.yaml``, or []."""
    if not workdir:
        return []
    try:
        manifest = Path(str(workdir)).expanduser() / AUTOMATION_MANIFEST
        if not manifest.is_file():
            return []
        import yaml

        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return []
        raw = data.get("requires_toolsets") or []
        if isinstance(raw, str):
            raw = [part for part in raw.split(",")]
        return _clean_names(raw)
    except Exception as err:
        logger.debug("automation manifest read failed for %s: %s", workdir, err)
        return []


def declared_required_toolsets(job: dict) -> List[str]:
    """The job's own ``required_toolsets`` plus its bundle's ``requires_toolsets``."""
    declared = _clean_names(job.get("required_toolsets") or [])
    for name in automation_required_toolsets(job.get("workdir")):
        if name not in declared:
            declared.append(name)
    return declared


def _mcp_server_names(cfg: dict) -> FrozenSet[str]:
    try:
        from hermes_cli.tools_config import enabled_mcp_server_names

        return frozenset(n.lower() for n in enabled_mcp_server_names(cfg or {}))
    except Exception:
        return frozenset()


def _canonical(name: str, mcp_names: FrozenSet[str]) -> str:
    """Fold the ``mcp-<server>`` spelling onto the bare server name — the form
    the cron resolver and the RBAC ceiling both use."""
    name = name.strip().lower()
    if name.startswith("mcp-") and name[4:] in mcp_names:
        return name[4:]
    return name


def _is_known_toolset(name: str, mcp_names: FrozenSet[str]) -> bool:
    if name in mcp_names:
        return True
    try:
        from toolsets import validate_toolset

        return bool(validate_toolset(name))
    except Exception:
        return True  # can't tell — don't block on it


def _toolset_tool_names(name: str) -> List[str]:
    """Tool names the agent would actually get for this toolset right now
    (after each tool's check_fn — missing creds/binaries/disconnected MCP
    server yield nothing). Monkeypatched in tests."""
    try:
        from model_tools import get_tool_definitions

        defs = get_tool_definitions(
            enabled_toolsets=[name], quiet_mode=True, skip_tool_search_assembly=True,
        )
        return sorted(
            (d.get("function") or {}).get("name") or d.get("name") or ""
            for d in defs or []
        )
    except Exception:
        return []


def _owner_grants(grant: Optional[FrozenSet[str]], name: str) -> bool:
    if grant is None or "*" in grant:
        return True
    try:
        from gateway.tool_access import FLOOR_TOOLSETS, _granted

        return _granted(grant, name) or name in FLOOR_TOOLSETS
    except Exception:
        return True


def _gated_unacked_tools(effective: List[str], acked: Iterable[str]) -> List[str]:
    try:
        from tools.approval import tool_requires_approval
    except Exception:
        return []
    acked_set = set(acked or [])
    gated: set = set()
    for toolset in effective:
        for tool_name in _toolset_tool_names(toolset):
            if tool_name and tool_name not in acked_set and tool_requires_approval(tool_name):
                gated.add(tool_name)
    return sorted(gated)


def evaluate_job_capabilities(
    job: dict,
    cfg: dict,
    owner_grant: Optional[FrozenSet[str]],
    *,
    include_enabled: bool,
    check_availability: bool,
) -> CapabilityReport:
    """Evaluate what ``job`` will receive at run time against what it needs.

    ``owner_grant`` is the job owner's RBAC toolset grant (``None`` = no
    ceiling). ``include_enabled`` adds the explicitly listed
    ``enabled_toolsets`` to the requirements (create/update time).
    ``check_availability`` additionally requires each needed toolset to
    expose at least one usable tool on this host — a create-time check; the
    runtime check leaves it off so a transient MCP disconnect can't block
    (and alert on) a run.
    """
    report = CapabilityReport()
    try:
        from cron.rbac_ceiling import apply_cron_toolset_ceiling
        from cron.scheduler import (
            _resolve_cron_disabled_toolsets,
            _resolve_cron_enabled_toolsets,
        )

        mcp_names = _mcp_server_names(cfg)
        disabled = {_canonical(n, mcp_names) for n in _resolve_cron_disabled_toolsets(cfg or {})}

        resolved = _resolve_cron_enabled_toolsets(job, cfg or {})
        capped = apply_cron_toolset_ceiling(resolved, owner_grant)
        if capped is not None:
            report.effective = sorted(
                {_canonical(t, mcp_names) for t in capped} - disabled
            )

        required = [_canonical(n, mcp_names) for n in declared_required_toolsets(job)]
        if include_enabled:
            for name in _clean_names(job.get("enabled_toolsets") or []):
                name = _canonical(name, mcp_names)
                if name != "no_mcp" and name not in required:
                    required.append(name)
        report.required = required

        for name in required:
            problem = None
            if not _is_known_toolset(name, mcp_names):
                problem = f"'{name}' is not a known toolset or enabled MCP server"
            elif name in disabled:
                hint = CRON_STRIPPED_HINTS.get(
                    name, "it is disabled for cron runs by `agent.disabled_toolsets`"
                )
                problem = f"'{name}' is removed from every cron run: {hint}"
            elif not _owner_grants(owner_grant, name):
                problem = (
                    f"'{name}' is outside the job owner's RBAC role — an admin "
                    "must add it to that role, or the job needs an owner whose "
                    "role grants it"
                )
            elif report.effective is not None and name not in report.effective:
                problem = (
                    f"'{name}' is not among the job's enabled toolsets — add it "
                    "to enabled_toolsets"
                )
            elif check_availability and not _toolset_tool_names(name):
                problem = (
                    f"'{name}' exposes no usable tools on this host (missing "
                    "credentials or binary, or its MCP server is not connected)"
                )
            if problem:
                report.problems.append(problem)
                report.missing.append(name)

        if report.effective is not None:
            report.gated_unacked = _gated_unacked_tools(
                report.effective, job.get("unattended_approved_tools") or []
            )
    except Exception as err:  # fail open — never block on an internal error
        logger.debug("cron capability evaluation failed (fail-open): %s", err)
        return CapabilityReport()
    return report


def grant_for_identity(identity, chat_id: Optional[str]) -> Optional[FrozenSet[str]]:
    """RBAC grant for a live identity (the creator at create time), or None
    when RBAC is inactive for its platform. Mirrors cron_owner_grant."""
    try:
        if identity is None or not identity.user_id or not identity.platform:
            return None
        from gateway.tool_access import policy_for_platform

        policy = policy_for_platform(str(identity.platform))
        if policy is None or not policy.enabled:
            return None
        return policy.grant_for(str(identity.user_id), chat_id or None)
    except Exception:
        return None


def _module_imports_outside_try(source: str) -> List[str]:
    """Top-level module names a Python source imports unconditionally.

    Imports inside a ``try`` body are skipped — that is how scripts declare an
    optional dependency with a fallback — as are relative imports and
    ``__future__``."""
    import ast

    tree = ast.parse(source)
    names: List[str] = []

    def visit(node, in_try: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_in_try = in_try or isinstance(child, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(child, ast.TryStar)
            )
            if not in_try and isinstance(child, ast.Import):
                names.extend(alias.name.split(".")[0] for alias in child.names)
            elif not in_try and isinstance(child, ast.ImportFrom):
                if child.level == 0 and child.module:
                    names.append(child.module.split(".")[0])
            visit(child, child_in_try)

    visit(tree, False)
    return [n for n in dict.fromkeys(names) if n != "__future__"]


def script_import_problems(script: Optional[str]) -> Optional[str]:
    """Describe modules a cron job's Python script imports that the GATEWAY's
    interpreter cannot find, or None.

    Cron scripts run under ``sys.executable`` — the gateway's venv — which is
    not necessarily the Python the author tested with (the VM's ``hermes``
    CLI and a bare ``python3`` both resolve elsewhere). Sibling modules in the
    script's own directory count as present. Fails open (None) on anything it
    cannot evaluate: a missing file, a shell script, a syntax error."""
    if not script:
        return None
    try:
        import importlib.util
        import sys

        from hermes_constants import get_hermes_home

        scripts_dir = (get_hermes_home() / "scripts").resolve()
        raw = Path(str(script)).expanduser()
        path = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()
        if path.suffix.lower() in {".sh", ".bash"} or not path.is_file():
            return None
        missing = []
        for name in _module_imports_outside_try(path.read_text(encoding="utf-8")):
            if (path.parent / f"{name}.py").exists() or (path.parent / name).is_dir():
                continue
            if name in sys.builtin_module_names:
                continue
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        if not missing:
            return None
        return (
            f"script '{path.name}' imports {', '.join(missing)}, which the "
            f"gateway's Python ({sys.executable}) cannot find — install into "
            f"that interpreter, e.g. `uv pip install --python {sys.executable} "
            f"{' '.join(missing)}` (the package name may differ from the module)"
        )
    except Exception as err:
        logger.debug("script import check failed for %s: %s", script, err)
        return None


def format_problems(report: CapabilityReport) -> str:
    return "; ".join(report.problems)
