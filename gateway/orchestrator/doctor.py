"""Read-only doctor for external agent lane availability."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .command import CommandResult, CommandRunner, SubprocessCommandRunner
from .redaction import redact_for, redact_text
from .registry import KNOWN_AGENTS, AgentKind, AgentSpec

_PERMISSION_RE = re.compile(
    r"(operation not permitted|namespace|uid_map|newuidmap|newgidmap|unshare|clone|bwrap|bubblewrap)",
    re.IGNORECASE,
)

# systemd user services start with a minimal PATH that omits ~/.local/bin, where
# pip/pipx-installed agents such as ``codex`` live. Hermes launches those agents
# through ``bash -ic`` (the interactive-shell PATH), so the doctor must discover
# them the same way; otherwise it falsely reports an installed agent as missing.
_USER_BIN_SUBDIRS = (".local/bin", "bin")


def _augmented_path() -> str | None:
    base = os.environ.get("PATH", "")
    parts = base.split(os.pathsep) if base else []
    home = os.path.expanduser("~")
    prefix: list[str] = []
    if home and home != "~":
        for sub in _USER_BIN_SUBDIRS:
            candidate = os.path.join(home, sub)
            if candidate not in parts and candidate not in prefix:
                prefix.append(candidate)
    combined = os.pathsep.join([*prefix, *parts])
    return combined or None


def _default_which(name: str) -> str | None:
    """Resolve a binary including user-local bin dirs the service PATH omits."""
    return shutil.which(name, path=_augmented_path())


def _probe_env() -> dict[str, str]:
    env = dict(os.environ)
    path = _augmented_path()
    if path:
        env["PATH"] = path
    return env


@dataclass
class SandboxHealth:

    status: str
    probe: str
    detail: str


@dataclass
class ExternalIsolationHealth:
    status: str
    mode: str
    detail: str


@dataclass
class AgentReport:
    name: str
    kind: str
    status: str
    path: str | None = None
    version: str | None = None
    sandbox: SandboxHealth | None = None
    external_isolation: ExternalIsolationHealth | None = None
    execution_mode: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DoctorReport:
    agents: list[AgentReport]
    tool: str = "agent-doctor/0.1"

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _first_line(text: str | None) -> str:
    for line in (text or "").splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _safe_detail(result: CommandResult, *, max_chars: int = 240) -> str:
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    return redact_text(combined)[:max_chars]


def check_codex_external_isolation(
    codex_path: str,
    *,
    runner: CommandRunner,
) -> ExternalIsolationHealth:
    """Probe whether Codex can run when Hermes provides isolation externally."""

    result = runner.run([codex_path, "sandbox", "-P", ":danger-full-access", "true"], timeout=10)
    detail = _safe_detail(result)
    if result.returncode == 0 and not result.timed_out:
        return ExternalIsolationHealth(
            status="available",
            mode="danger-full-access",
            detail="external worktree/copy isolation required",
        )
    if result.timed_out:
        return ExternalIsolationHealth("unknown", "danger-full-access", "probe timed out")
    return ExternalIsolationHealth(
        "unavailable",
        "danger-full-access",
        detail or "danger-full-access probe failed",
    )


def detect_binary(
    spec: AgentSpec,
    *,
    which_fn: Callable[[str], str | None],
    runner: CommandRunner,
) -> AgentReport:
    path = which_fn(spec.name)
    if not path:
        return AgentReport(
            name=spec.name,
            kind=spec.kind.value,
            status="missing",
            notes=["binary not found on PATH"],
        )

    status = "available"
    version: str | None = None
    external_isolation: ExternalIsolationHealth | None = None
    execution_mode: str | None = None
    notes: list[str] = []
    if spec.version_argv:
        # Probe via the resolved absolute path so the version check does not
        # depend on the runner's PATH (the service PATH omits ~/.local/bin).
        probe_argv = (path, *spec.version_argv[1:])
        result = runner.run(probe_argv, timeout=10)
        if result.timed_out:
            status = "unknown"
            notes.append("version probe timed out")
        elif result.returncode != 0:
            status = "unknown"
            detail = _safe_detail(result)
            notes.append(f"version probe failed: {detail}" if detail else "version probe failed")
        else:
            version = redact_for(spec, _first_line(result.stdout) or _first_line(result.stderr))
            if version == "<suppressed>":
                version = None

    sandbox = check_codex_sandbox(which_fn=which_fn, runner=runner) if spec.sandbox else None
    if sandbox is not None and sandbox.status == "degraded":
        status = "degraded"
        notes.append("sandbox degraded")
        if spec.external_isolation:
            external_isolation = check_codex_external_isolation(path, runner=runner)
            if external_isolation.status == "available":
                execution_mode = "external-isolated"
                notes.append("external isolation available")
            else:
                notes.append(f"external isolation {external_isolation.status}")
    elif sandbox is not None and sandbox.status == "unavailable":
        notes.append("sandbox probe unavailable")

    return AgentReport(
        name=spec.name,
        kind=spec.kind.value,
        status=status,
        path=path,
        version=version,
        sandbox=sandbox,
        external_isolation=external_isolation,
        execution_mode=execution_mode,
        notes=[redact_text(note) for note in notes],
    )


def detect_shell_function(spec: AgentSpec, *, runner: CommandRunner) -> AgentReport:
    # Names come from the fixed registry, not user input. Keep this probe to
    # `type -t`; never execute aliases/functions during Phase 1.
    result = runner.run(["bash", "-ic", f"type -t {spec.name}"], timeout=5)
    detected = _first_line(result.stdout) == "function" and result.returncode == 0
    if detected:
        notes = ["available via bash -ic"]
        if spec.secrets:
            notes.append("output suppressed for sensitive wrapper")
        return AgentReport(
            name=spec.name,
            kind=spec.kind.value,
            status="available",
            version=None,
            notes=notes,
        )
    notes = ["shell function not found via bash -ic"]
    if result.timed_out:
        notes = ["shell function probe timed out"]
    elif result.returncode != 0 and not spec.secrets:
        detail = _safe_detail(result)
        if detail:
            notes.append(detail)
    return AgentReport(
        name=spec.name,
        kind=spec.kind.value,
        status="missing" if not result.timed_out else "unknown",
        version=None,
        notes=[redact_text(note) for note in notes],
    )


def check_codex_sandbox(
    *,
    which_fn: Callable[[str], str | None] = _default_which,
    runner: CommandRunner | None = None,
) -> SandboxHealth:
    """Best-effort local namespace/sandbox health check without invoking Codex."""

    runner = runner or SubprocessCommandRunner(env=_probe_env())
    probes: list[tuple[str, list[str]]] = []
    if which_fn("bwrap"):
        probes.append(("bwrap user namespace", ["bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "true"]))
    if which_fn("unshare"):
        probes.append(("unshare user map", ["unshare", "-Ur", "true"]))
        probes.append(("unshare network", ["unshare", "-n", "true"]))

    if not probes:
        return SandboxHealth(
            status="unavailable",
            probe="bwrap/unshare namespace smoke",
            detail="no bwrap or unshare probe binary found",
        )

    unknown_details: list[str] = []
    for label, argv in probes:
        result = runner.run(argv, timeout=5)
        detail = _safe_detail(result)
        if result.returncode == 0 and not result.timed_out:
            continue
        if result.timed_out:
            return SandboxHealth("unknown", label, "probe timed out")
        if _PERMISSION_RE.search(detail):
            return SandboxHealth("degraded", label, detail or "namespace permission failure")
        unknown_details.append(f"{label}: {detail or 'non-zero exit'}")

    if unknown_details:
        return SandboxHealth("unknown", "bwrap/unshare namespace smoke", redact_text("; ".join(unknown_details))[:240])
    return SandboxHealth("healthy", "bwrap/unshare namespace smoke", "all probes passed")


def run_doctor(
    *,
    specs: Sequence[AgentSpec] = KNOWN_AGENTS,
    which_fn: Callable[[str], str | None] = _default_which,
    runner: CommandRunner | None = None,
) -> DoctorReport:
    runner = runner or SubprocessCommandRunner(env=_probe_env())
    reports: list[AgentReport] = []
    for spec in specs:
        if spec.kind is AgentKind.BINARY:
            reports.append(detect_binary(spec, which_fn=which_fn, runner=runner))
        elif spec.kind is AgentKind.SHELL_FUNCTION:
            reports.append(detect_shell_function(spec, runner=runner))
        else:  # pragma: no cover - defensive for future enum values
            reports.append(AgentReport(spec.name, spec.kind.value, "unknown", notes=["unsupported agent kind"]))
    return DoctorReport(agents=reports)


def main() -> None:
    print(json.dumps(run_doctor().to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
