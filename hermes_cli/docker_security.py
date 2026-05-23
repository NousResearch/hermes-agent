"""Docker/container permission review and enforcement helpers.

The functions in this module inspect configuration shape and synthetic command
text only. They never read Docker config files, contact the Docker daemon, or
resolve environment variable values.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from hermes_cli.security_policy import RiskClass


SENSITIVE_ENV_NAME_RE = re.compile(
    r"(secret|token|api[_-]?key|password|passwd|private[_-]?key|credential|auth|"
    r"session|cookie|ssh_auth_sock|docker_config|docker_host|aws_|gcp_|google_|"
    r"azure_|gh_|github_|npm_|pypi_|twine_)",
    re.IGNORECASE,
)

ENV_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

DOCKER_BINARIES = {"docker", "podman"}
DOCKER_FALSEY_BOOL_VALUES = {"0", "false", "f", "no", "n", "off"}
SENSITIVE_HOST_PATH_MARKERS = (
    "/.aws",
    "/.azure",
    "/.config/gh",
    "/.docker",
    "/.gnupg",
    "/.kube",
    "/.netrc",
    "/.npmrc",
    "/.pypirc",
    "/.ssh",
    "/library/keychains",
)
HIGH_SEVERITIES = frozenset({"high", "critical"})


class DockerSecurityPolicyError(RuntimeError):
    """Raised when Docker execution is blocked by container security policy."""


@dataclass(frozen=True)
class DockerSecurityFinding:
    code: str
    severity: str
    risk_category: str
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        data = {
            "code": self.code,
            "severity": self.severity,
            "risk_category": self.risk_category,
            "message": self.message,
        }
        if self.detail:
            data["detail"] = self.detail
        return data


def _finding(
    code: str,
    severity: str,
    risk_category: str,
    message: str,
    detail: str = "",
) -> DockerSecurityFinding:
    return DockerSecurityFinding(
        code=code,
        severity=severity,
        risk_category=risk_category,
        message=message,
        detail=detail,
    )


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _as_sequence(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _is_sensitive_env_name(name: str) -> bool:
    return bool(SENSITIVE_ENV_NAME_RE.search(name.strip()))


def _docker_bool_flag_enabled(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in DOCKER_FALSEY_BOOL_VALUES


def _redacted_env_detail(name: str, *, source: str) -> str:
    return f"{source}:{name.strip() or '<empty>'}"


def _normalize_host_path(path: str) -> str:
    text = path.strip().strip("\"'")
    if text.startswith("~"):
        return "~" + text[1:]
    if text.startswith("$HOME"):
        return "$HOME" + text[len("$HOME"):]
    if text.startswith("${HOME}"):
        return "${HOME}" + text[len("${HOME}"):]
    return text


def _redacted_path_detail(path: str) -> str:
    normalized = _normalize_host_path(path)
    if normalized in {"", "."}:
        return "relative-or-empty-host-path"
    if normalized in {"/", "/private"}:
        return "host-root"
    if normalized.startswith("/Users/"):
        parts = PurePosixPath(normalized).parts
        if len(parts) <= 3:
            return "host-user-home"
        return f"host-user-home/{parts[3]}"
    if normalized.startswith("/home/"):
        parts = PurePosixPath(normalized).parts
        if len(parts) <= 3:
            return "host-user-home"
        return f"host-user-home/{parts[3]}"
    return normalized


def _mount_source_from_volume_spec(spec: str) -> str:
    text = spec.strip()
    if not text:
        return ""
    if ":" not in text:
        return text
    return text.split(":", 1)[0]


def _mount_source_from_mount_spec(spec: str) -> str:
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        if sep and key.strip().lower() in {"source", "src"}:
            return value.strip()
    return ""


def _findings_for_mount(source: str, *, source_kind: str) -> list[DockerSecurityFinding]:
    source = source.strip()
    if not source:
        return []
    normalized = _normalize_host_path(source)
    lowered = normalized.lower()
    detail = _redacted_path_detail(normalized)
    findings: list[DockerSecurityFinding] = []
    if "/var/run/docker.sock" in lowered or lowered.endswith("docker.sock"):
        findings.append(_finding(
            "docker_socket_mount",
            "critical",
            RiskClass.CREDENTIAL_SENSITIVE,
            "Docker socket mount gives the container host-level Docker control.",
            source_kind,
        ))
    if lowered in {"/", "/private"}:
        findings.append(_finding(
            "host_root_mount",
            "critical",
            RiskClass.DESTRUCTIVE,
            "Host root mount exposes broad host filesystem state.",
            detail,
        ))
    if normalized in {"~", "$HOME", "${HOME}"} or (
        normalized.startswith(("/Users/", "/home/")) and len(PurePosixPath(normalized).parts) <= 3
    ):
        findings.append(_finding(
            "host_home_mount",
            "high",
            RiskClass.PRIVATE_DATA_ACCESS,
            "Host home mount can expose private files and credentials.",
            detail,
        ))
    if any(marker in lowered for marker in SENSITIVE_HOST_PATH_MARKERS):
        findings.append(_finding(
            "credential_path_mount",
            "high",
            RiskClass.CREDENTIAL_SENSITIVE,
            "Sensitive host credential path is mounted into the container.",
            detail,
        ))
    return findings


def _consume_next(tokens: Sequence[str], index: int) -> str | None:
    next_index = index + 1
    if next_index >= len(tokens):
        return None
    return tokens[next_index]


def analyze_docker_args(args: Iterable[object], *, source: str = "docker_args") -> list[DockerSecurityFinding]:
    """Analyze Docker/Podman run-like argument lists without executing them."""
    tokens = [str(arg) for arg in args if arg is not None]
    findings: list[DockerSecurityFinding] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--privileged" or (
            token.startswith("--privileged=")
            and _docker_bool_flag_enabled(token.split("=", 1)[1])
        ):
            findings.append(_finding(
                "privileged_container",
                "critical",
                RiskClass.CREDENTIAL_SENSITIVE,
                "Privileged containers weaken host isolation.",
                source,
            ))
        elif token in {"--network", "--net"}:
            value = _consume_next(tokens, index) or ""
            if value == "host":
                findings.append(_finding(
                    "host_network",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Host networking weakens container isolation.",
                    source,
                ))
                index += 1
        elif token in {"--pid", "--ipc", "--userns"}:
            value = _consume_next(tokens, index) or ""
            if value == "host":
                findings.append(_finding(
                    "host_namespace",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Host namespace sharing weakens container isolation.",
                    f"{source}:{token}",
                ))
                index += 1
        elif token.startswith("--network=") or token.startswith("--net="):
            if token.split("=", 1)[1] == "host":
                findings.append(_finding(
                    "host_network",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Host networking weakens container isolation.",
                    source,
                ))
        elif token.startswith(("--pid=", "--ipc=", "--userns=")):
            if token.split("=", 1)[1] == "host":
                findings.append(_finding(
                    "host_namespace",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Host namespace sharing weakens container isolation.",
                    f"{source}:{token.split('=', 1)[0]}",
                ))
        elif token in {"-v", "--volume"}:
            spec = _consume_next(tokens, index) or ""
            findings.extend(_findings_for_mount(_mount_source_from_volume_spec(spec), source_kind=source))
            index += 1
        elif token.startswith("-v") and token != "-v" and not token.startswith("--"):
            spec = token[2:]
            findings.extend(_findings_for_mount(_mount_source_from_volume_spec(spec), source_kind=source))
        elif token.startswith(("-v=", "--volume=")):
            spec = token.split("=", 1)[1]
            findings.extend(_findings_for_mount(_mount_source_from_volume_spec(spec), source_kind=source))
        elif token in {"--mount"}:
            spec = _consume_next(tokens, index) or ""
            findings.extend(_findings_for_mount(_mount_source_from_mount_spec(spec), source_kind=source))
            index += 1
        elif token.startswith("--mount="):
            spec = token.split("=", 1)[1]
            findings.extend(_findings_for_mount(_mount_source_from_mount_spec(spec), source_kind=source))
        elif token in {"-e", "--env"}:
            value = _consume_next(tokens, index) or ""
            env_name = value.split("=", 1)[0].strip()
            if _is_sensitive_env_name(env_name):
                findings.append(_finding(
                    "sensitive_env_forward",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Sensitive environment variable is forwarded into the container.",
                    _redacted_env_detail(env_name, source=source),
                ))
            index += 1
        elif token.startswith("-e") and token != "-e" and not token.startswith("--"):
            env_name = token[2:].split("=", 1)[0].strip()
            if _is_sensitive_env_name(env_name):
                findings.append(_finding(
                    "sensitive_env_forward",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Sensitive environment variable is forwarded into the container.",
                    _redacted_env_detail(env_name, source=source),
                ))
        elif token.startswith("--env="):
            env_name = token.split("=", 1)[1].split("=", 1)[0].strip()
            if _is_sensitive_env_name(env_name):
                findings.append(_finding(
                    "sensitive_env_forward",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Sensitive environment variable is forwarded into the container.",
                    _redacted_env_detail(env_name, source=source),
                ))
        elif token in {"--env-file"} or token.startswith("--env-file="):
            findings.append(_finding(
                "env_file_forward",
                "high",
                RiskClass.CREDENTIAL_SENSITIVE,
                "Docker env-file can forward many secrets at once.",
                source,
            ))
            if token == "--env-file":
                index += 1
        elif token.startswith("--cap-add"):
            value = token.split("=", 1)[1] if "=" in token else (_consume_next(tokens, index) or "")
            if value.upper() == "ALL":
                findings.append(_finding(
                    "all_capabilities",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "Adding all Linux capabilities weakens container isolation.",
                    source,
                ))
                if "=" not in token:
                    index += 1
        elif token in {"--device", "--group-add"} or token.startswith(("--device=", "--group-add=")):
            findings.append(_finding(
                "host_device_or_group",
                "medium",
                RiskClass.CREDENTIAL_SENSITIVE,
                "Host device or group access may weaken container isolation.",
                source,
            ))
            if token in {"--device", "--group-add"}:
                index += 1
        index += 1
    return _dedupe_findings(findings)


def analyze_docker_command(command: str) -> list[DockerSecurityFinding]:
    """Analyze Docker/Podman command text without executing it."""
    tokens = _split_command(command or "")
    if not tokens:
        return []

    docker_indices = [
        index
        for index, token in enumerate(tokens)
        if os.path.basename(token) in DOCKER_BINARIES
    ]
    if not docker_indices:
        return []

    findings: list[DockerSecurityFinding] = []
    # Env-prefix review is scoped to Docker/Podman invocations only; ordinary
    # non-container commands belong to the general command risk policy.
    first_docker_index = docker_indices[0] if docker_indices else len(tokens)
    for index, token in enumerate(tokens[:first_docker_index]):
        env_match = ENV_ASSIGNMENT_RE.match(token)
        if env_match and _is_sensitive_env_name(env_match.group(1)):
            findings.append(_finding(
                "sensitive_env_prefix",
                "high",
                RiskClass.CREDENTIAL_SENSITIVE,
                "Sensitive environment variable is set for the Docker command.",
                _redacted_env_detail(env_match.group(1), source="command_prefix"),
            ))

    for docker_index in docker_indices:
        findings.extend(analyze_docker_args(tokens[docker_index + 1:], source="command"))
    return _dedupe_findings(findings)


def analyze_docker_terminal_config(config: Mapping[str, Any]) -> list[DockerSecurityFinding]:
    """Analyze Hermes terminal Docker config without reading secret values."""
    terminal = config.get("terminal") if isinstance(config, Mapping) else {}
    if not isinstance(terminal, Mapping):
        terminal = {}

    findings: list[DockerSecurityFinding] = []
    backend = str(terminal.get("backend", "")).strip().lower()
    has_docker_settings = any(str(key).startswith("docker_") for key in terminal)
    if backend and backend != "docker" and not has_docker_settings:
        return []

    forward_env = _as_sequence(terminal.get("docker_forward_env"))
    for item in forward_env:
        name = str(item).strip()
        if not name:
            continue
        if _is_sensitive_env_name(name):
            findings.append(_finding(
                "sensitive_forward_env_config",
                "high",
                RiskClass.CREDENTIAL_SENSITIVE,
                "docker_forward_env includes a credential-sensitive variable name.",
                _redacted_env_detail(name, source="terminal.docker_forward_env"),
            ))

    docker_env = terminal.get("docker_env")
    if isinstance(docker_env, Mapping):
        for key in docker_env:
            name = str(key).strip()
            if _is_sensitive_env_name(name):
                findings.append(_finding(
                    "sensitive_docker_env_config",
                    "high",
                    RiskClass.CREDENTIAL_SENSITIVE,
                    "docker_env includes a credential-sensitive variable name.",
                    _redacted_env_detail(name, source="terminal.docker_env"),
                ))
    elif docker_env:
        findings.append(_finding(
            "invalid_docker_env_shape",
            "medium",
            RiskClass.CREDENTIAL_SENSITIVE,
            "docker_env is not a mapping; Docker backend will ignore or fail safe.",
            "terminal.docker_env",
        ))

    for item in _as_sequence(terminal.get("docker_volumes")):
        if isinstance(item, str):
            findings.extend(_findings_for_mount(_mount_source_from_volume_spec(item), source_kind="terminal.docker_volumes"))
        elif item is not None:
            findings.append(_finding(
                "invalid_docker_volume_shape",
                "medium",
                RiskClass.CREDENTIAL_SENSITIVE,
                "docker_volumes should contain string mount specs only.",
                "terminal.docker_volumes",
            ))

    if str(terminal.get("docker_mount_cwd_to_workspace", "")).strip().lower() in {"true", "1", "yes", "on"}:
        findings.append(_finding(
            "host_cwd_workspace_mount",
            "medium",
            RiskClass.PRIVATE_DATA_ACCESS,
            "Host working directory mount is enabled for Docker workspace.",
            "terminal.docker_mount_cwd_to_workspace",
        ))

    findings.extend(analyze_docker_args(_as_sequence(terminal.get("docker_extra_args")), source="terminal.docker_extra_args"))
    return _dedupe_findings(findings)


def analyze_docker_backend_options(
    *,
    forward_env: Iterable[object] | None = None,
    env_keys: Iterable[object] | None = None,
    volumes: Iterable[object] | None = None,
    extra_args: Iterable[object] | None = None,
    auto_mount_cwd: bool = False,
) -> list[DockerSecurityFinding]:
    """Analyze Docker backend options without including env values."""
    docker_env = {str(key): "<redacted>" for key in (env_keys or []) if str(key).strip()}
    return analyze_docker_terminal_config({
        "terminal": {
            "backend": "docker",
            "docker_forward_env": list(forward_env or []),
            "docker_env": docker_env,
            "docker_volumes": list(volumes or []),
            "docker_extra_args": list(extra_args or []),
            "docker_mount_cwd_to_workspace": bool(auto_mount_cwd),
        }
    })


def high_severity_findings(
    findings: Iterable[DockerSecurityFinding],
) -> list[DockerSecurityFinding]:
    """Return findings that should block Docker execution by default."""
    return [finding for finding in findings if finding.severity in HIGH_SEVERITIES]


def enforce_no_high_severity_findings(
    findings: Iterable[DockerSecurityFinding],
    *,
    context: str = "Docker backend",
) -> None:
    """Fail closed before Docker execution for high-severity container risks."""
    blocked = high_severity_findings(findings)
    if not blocked:
        return
    codes = ", ".join(sorted({finding.code for finding in blocked}))
    raise DockerSecurityPolicyError(
        f"{context} blocked by Hermes container security policy "
        f"({len(blocked)} high-severity finding(s): {codes}). "
        "Remove high-risk Docker config or use a future typed-confirmed "
        "per-job override."
    )


def summarize_findings(findings: Iterable[DockerSecurityFinding]) -> dict[str, Any]:
    findings = list(findings)
    severity_order = {"info": 0, "medium": 1, "high": 2, "critical": 3}
    max_severity = "info"
    for finding in findings:
        if severity_order.get(finding.severity, 0) > severity_order.get(max_severity, 0):
            max_severity = finding.severity
    return {
        "finding_count": len(findings),
        "max_severity": max_severity if findings else "none",
        "codes": sorted({finding.code for finding in findings}),
        "typed_confirmation_required": any(f.severity in {"high", "critical"} for f in findings),
    }


def finding_notes(findings: Iterable[DockerSecurityFinding]) -> list[str]:
    notes = []
    for finding in findings:
        notes.append(f"Docker review: {finding.code} ({finding.severity})")
    return sorted(set(notes))


def _dedupe_findings(findings: Iterable[DockerSecurityFinding]) -> list[DockerSecurityFinding]:
    seen: set[tuple[str, str, str]] = set()
    result: list[DockerSecurityFinding] = []
    for finding in findings:
        key = (finding.code, finding.severity, finding.detail)
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result
