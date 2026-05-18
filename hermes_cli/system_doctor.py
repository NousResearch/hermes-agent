"""Concise, non-secret system health dashboard for Hermes runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping

import yaml

from hermes_constants import get_hermes_home

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Hermes currently targets 3.11+
    tomllib = None  # type: ignore[assignment]


STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUSES = (STATUS_OK, STATUS_WARN, STATUS_FAIL)

_STATUS_RANK = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}
_SECRET_NAME_RE = re.compile(
    r"\b[A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)[A-Z0-9_]*\b"
    r"\s*[:=]\s*([^,\s|]+)",
    re.IGNORECASE,
)
_SECRET_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|"
    r"Bearer\s+[A-Za-z0-9._~+/=-]{8,})\b",
    re.IGNORECASE,
)
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


@dataclass(frozen=True)
class SystemDoctorEntry:
    """One dashboard check result."""

    name: str
    status: str
    detail: str = ""
    remediation: str = ""
    category: str = "General"

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unknown system doctor status: {self.status}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SystemDoctorReport:
    """Structured system doctor report."""

    generated_at: str
    hermes_home: str
    entries: tuple[SystemDoctorEntry, ...]

    @property
    def ok_count(self) -> int:
        return self.count_by_status(STATUS_OK)

    @property
    def warn_count(self) -> int:
        return self.count_by_status(STATUS_WARN)

    @property
    def fail_count(self) -> int:
        return self.count_by_status(STATUS_FAIL)

    @property
    def status(self) -> str:
        if self.fail_count:
            return STATUS_FAIL
        if self.warn_count:
            return STATUS_WARN
        return STATUS_OK

    @property
    def has_failures(self) -> bool:
        return self.fail_count > 0

    def count_by_status(self, status: str) -> int:
        return sum(1 for entry in self.entries if entry.status == status)

    def exit_code(self, *, fail_on_fail: bool = False) -> int:
        return 1 if fail_on_fail and self.has_failures else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "hermes_home": self.hermes_home,
            "status": self.status,
            "summary": {
                STATUS_OK: self.ok_count,
                STATUS_WARN: self.warn_count,
                STATUS_FAIL: self.fail_count,
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }


def build_system_doctor_report(
    *,
    hermes_home: Path | None = None,
    codex_config_path: Path | None = None,
    obsidian_rules_path: Path | None = None,
    check_honcho_reachability: bool = False,
    now: datetime | None = None,
) -> SystemDoctorReport:
    """Build a concise system health report without network checks by default."""

    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    current_time = now or datetime.now(timezone.utc)
    entries: list[SystemDoctorEntry] = []

    config_path = home / "config.yaml"
    config_data: Mapping[str, Any] | None = None

    entries.extend(_check_hermes_home(home))
    config_entries, config_data = _check_config(config_path)
    entries.extend(config_entries)
    entries.extend(_check_obsidian_rules(config_data, obsidian_rules_path))
    entries.extend(_check_pending_interactions(current_time))
    entries.extend(_check_memory_governance_queue())
    entries.extend(_check_honcho(check_honcho_reachability))
    entries.extend(_check_codex_readiness(codex_config_path))
    entries.extend(_check_cron_storage(home))

    return SystemDoctorReport(
        generated_at=_format_time(current_time),
        hermes_home=str(home),
        entries=tuple(entries),
    )


def render_system_doctor_dashboard(report: SystemDoctorReport) -> str:
    """Render a Discord-pasteable Markdown dashboard."""

    lines = [
        "# Hermes System Doctor Dashboard",
        "",
        f"- Generated: {_sanitize(report.generated_at)}",
        f"- Overall: {report.status}",
        f"- Summary: OK {report.ok_count} | WARN {report.warn_count} | FAIL {report.fail_count}",
        f"- Hermes home: {_sanitize(report.hermes_home)}",
        "",
    ]

    for category in _category_order(report.entries):
        lines.append(f"## {category}")
        lines.append("| Status | Check | Detail | Remediation |")
        lines.append("| --- | --- | --- | --- |")
        for entry in report.entries:
            if entry.category != category:
                continue
            lines.append(
                "| "
                + " | ".join(
                    _table_cell(value)
                    for value in (
                        entry.status,
                        entry.name,
                        entry.detail or "-",
                        entry.remediation or "-",
                    )
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def print_system_doctor_dashboard(
    *,
    fail_on_fail: bool = False,
    check_honcho_reachability: bool = False,
) -> int:
    """Print the dashboard and return the requested process-style exit code."""

    report = build_system_doctor_report(
        check_honcho_reachability=check_honcho_reachability,
    )
    print(render_system_doctor_dashboard(report), end="")
    return report.exit_code(fail_on_fail=fail_on_fail)


def _check_hermes_home(home: Path) -> list[SystemDoctorEntry]:
    entries: list[SystemDoctorEntry] = []
    if home.exists():
        if home.is_dir():
            entries.append(_entry("Hermes home path", STATUS_OK, f"{home} exists", category="Runtime Home"))
        else:
            entries.append(
                _entry(
                    "Hermes home path",
                    STATUS_FAIL,
                    f"{home} exists but is not a directory",
                    "Move the file or set HERMES_HOME to a directory.",
                    "Runtime Home",
                )
            )
    else:
        entries.append(
            _entry(
                "Hermes home path",
                STATUS_WARN,
                f"{home} does not exist yet",
                "Run hermes setup or start Hermes once to initialize it.",
                "Runtime Home",
            )
        )

    if home == Path.home() or home == Path("/"):
        entries.append(
            _entry(
                "Profile isolation",
                STATUS_FAIL,
                "Hermes home resolves to an unsafe broad directory",
                "Set HERMES_HOME to ~/.hermes or ~/.hermes/profiles/<name>.",
                "Runtime Home",
            )
        )
    elif home.parent.name == "profiles":
        entries.append(
            _entry(
                "Profile isolation",
                STATUS_OK,
                f"named profile '{home.name}' under profiles/",
                category="Runtime Home",
            )
        )
    elif home.name == ".hermes":
        entries.append(_entry("Profile isolation", STATUS_OK, "default profile layout", category="Runtime Home"))
    else:
        entries.append(
            _entry(
                "Profile isolation",
                STATUS_WARN,
                "custom HERMES_HOME layout",
                "Verify this path is not shared by another live Hermes profile.",
                "Runtime Home",
            )
        )
    return entries


def _check_config(config_path: Path) -> tuple[list[SystemDoctorEntry], Mapping[str, Any] | None]:
    if not config_path.exists():
        return (
            [
                _entry(
                    "config.yaml",
                    STATUS_WARN,
                    "not found; Hermes will rely on defaults",
                    "Run hermes setup to create a profile-local config.yaml.",
                    "Configuration",
                )
            ],
            None,
        )

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return (
            [
                _entry(
                    "config.yaml",
                    STATUS_FAIL,
                    f"unreadable or invalid YAML: {type(exc).__name__}",
                    "Fix config.yaml syntax or rerun hermes setup.",
                    "Configuration",
                )
            ],
            None,
        )

    if not isinstance(data, Mapping):
        return (
            [
                _entry(
                    "config.yaml",
                    STATUS_FAIL,
                    f"expected a mapping, found {type(data).__name__}",
                    "Replace config.yaml with a YAML mapping.",
                    "Configuration",
                )
            ],
            None,
        )
    return ([_entry("config.yaml", STATUS_OK, "exists and parses", category="Configuration")], data)


def _check_obsidian_rules(
    config_data: Mapping[str, Any] | None,
    explicit_path: Path | None,
) -> list[SystemDoctorEntry]:
    path = explicit_path or _configured_obsidian_rules_path(config_data)
    if path is None:
        return [
            _entry(
                "Obsidian vault rules",
                STATUS_WARN,
                "not configured or known",
                "Set obsidian.vault_rules_path or pass an explicit path when auditing a known vault.",
                "Knowledge Stores",
            )
        ]
    if path.exists() and path.is_file():
        return [_entry("Obsidian vault rules", STATUS_OK, f"{path.name} is available", category="Knowledge Stores")]
    return [
        _entry(
            "Obsidian vault rules",
            STATUS_WARN,
            f"{path.name} not found at configured location",
            "Create the rules file or update obsidian.vault_rules_path.",
            "Knowledge Stores",
        )
    ]


def _configured_obsidian_rules_path(config_data: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(config_data, Mapping):
        return None
    candidates: list[Any] = []
    obsidian_cfg = config_data.get("obsidian")
    if isinstance(obsidian_cfg, Mapping):
        candidates.extend(
            [
                obsidian_cfg.get("vault_rules_path"),
                obsidian_cfg.get("rules_path"),
            ]
        )
        vault_path = obsidian_cfg.get("vault_path")
        if vault_path:
            candidates.append(Path(str(vault_path)) / "90. setting" / "Vault 운영 원칙.md")
    memory_cfg = config_data.get("memory")
    if isinstance(memory_cfg, Mapping):
        candidates.append(memory_cfg.get("obsidian_rules_path"))
    for candidate in candidates:
        if candidate:
            return Path(str(candidate)).expanduser()
    return None


def _check_pending_interactions(now: datetime) -> list[SystemDoctorEntry]:
    try:
        from gateway import pending_interactions as pending
    except Exception as exc:
        return [
            _entry(
                "Pending interaction store",
                STATUS_WARN,
                f"module unavailable: {type(exc).__name__}",
                category="Pending Interactions",
            )
        ]

    path = pending.pending_store_path()
    if not path.exists():
        return [
            _entry(
                "Pending interaction store",
                STATUS_WARN,
                "records.json not initialized",
                "This is normal until a gateway or cron response records a pending handoff.",
                "Pending Interactions",
            )
        ]
    records, error = _read_json_records(path, "records")
    if error:
        return [
            _entry(
                "Pending interaction store",
                STATUS_FAIL,
                error,
                "Repair or remove the profile-local pending_interactions/records.json file.",
                "Pending Interactions",
            )
        ]
    active, expired = _count_pending_records(records, now)
    return [
        _entry(
            "Pending interaction store",
            STATUS_OK,
            f"total={len(records)} active={active} expired={expired}",
            category="Pending Interactions",
        )
    ]


def _check_memory_governance_queue() -> list[SystemDoctorEntry]:
    try:
        from agent import memory_governance as governance
    except Exception as exc:
        return [
            _entry(
                "Memory governance queue",
                STATUS_WARN,
                f"module unavailable: {type(exc).__name__}",
                category="Memory Governance",
            )
        ]

    path = governance.memory_governance_queue_path()
    if not path.exists():
        return [
            _entry(
                "Memory governance queue",
                STATUS_WARN,
                "review_queue.json not initialized",
                "This is normal until the memory governance gate enqueues a review.",
                "Memory Governance",
            )
        ]
    items, error = _read_json_records(path, "items")
    if error:
        return [
            _entry(
                "Memory governance queue",
                STATUS_FAIL,
                error,
                "Repair or remove the profile-local memory_governance/review_queue.json file.",
                "Memory Governance",
            )
        ]
    pending_count = sum(1 for item in items if str(item.get("status", "")) == "pending_review")
    return [
        _entry(
            "Memory governance queue",
            STATUS_OK,
            f"total={len(items)} pending_review={pending_count}",
            category="Memory Governance",
        )
    ]


def _ensure_honcho_parent_attribute() -> None:
    """Repair parent package linkage after plugin loaders import Honcho via specs."""

    try:
        import sys
        import plugins.memory as memory_pkg
    except Exception:
        return
    honcho_mod = sys.modules.get("plugins.memory.honcho")
    client_mod = sys.modules.get("plugins.memory.honcho.client")
    if honcho_mod is not None and not hasattr(memory_pkg, "honcho"):
        setattr(memory_pkg, "honcho", honcho_mod)
    if honcho_mod is not None and client_mod is not None and not hasattr(honcho_mod, "client"):
        setattr(honcho_mod, "client", client_mod)


def _check_honcho(check_reachability: bool) -> list[SystemDoctorEntry]:
    category = "Honcho"
    try:
        from plugins.memory.honcho.client import (
            HonchoClientConfig,
            get_honcho_client,
            reset_honcho_client,
            resolve_config_path,
        )
        _ensure_honcho_parent_attribute()
    except Exception as exc:
        return [
            _entry("Honcho configured", STATUS_WARN, f"plugin unavailable: {type(exc).__name__}", category=category),
            _entry("Honcho reachable", STATUS_WARN, "not checked because plugin is unavailable", category=category),
        ]

    try:
        cfg = HonchoClientConfig.from_global_config()
        cfg_path = resolve_config_path()
    except Exception as exc:
        return [
            _entry(
                "Honcho configured",
                STATUS_WARN,
                f"configuration could not be loaded: {type(exc).__name__}",
                "Run hermes memory setup or inspect honcho.json.",
                category,
            ),
            _entry("Honcho reachable", STATUS_WARN, "not checked because config could not be loaded", category=category),
        ]

    configured = bool(cfg.enabled and (cfg.api_key or cfg.base_url))
    config_detail = f"workspace={cfg.workspace_id} mode={cfg.recall_mode} freq={cfg.write_frequency}"
    entries = [
        _entry(
            "Honcho configured",
            STATUS_OK if configured else STATUS_WARN,
            config_detail if configured else f"not configured at {cfg_path.name}",
            "" if configured else "Run hermes memory setup if Honcho should be active.",
            category,
        )
    ]

    if not configured:
        entries.append(_entry("Honcho reachable", STATUS_WARN, "skipped because Honcho is not configured", category=category))
        return entries
    if not check_reachability:
        entries.append(
            _entry(
                "Honcho reachable",
                STATUS_WARN,
                "not checked; reachability checks are opt-in",
                "Run hermes doctor --dashboard-only --check-dashboard-reachability.",
                category,
            )
        )
        return entries

    try:
        reset_honcho_client()
        get_honcho_client(cfg)
        entries.append(_entry("Honcho reachable", STATUS_OK, "client initialized", category=category))
    except Exception as exc:
        entries.append(
            _entry(
                "Honcho reachable",
                STATUS_FAIL,
                f"client initialization failed: {type(exc).__name__}",
                "Check Honcho service availability and memory setup.",
                category,
            )
        )
    return entries


def _check_codex_readiness(codex_config_path: Path | None) -> list[SystemDoctorEntry]:
    category = "Codex Readiness"
    entries = []
    codex_path = shutil.which("codex")
    if codex_path:
        entries.append(_entry("codex command", STATUS_OK, "available on PATH", category=category))
    else:
        entries.append(
            _entry(
                "codex command",
                STATUS_WARN,
                "not found on PATH",
                "Install Codex CLI or ensure its bin directory is on PATH if this runtime should launch Codex.",
                category,
            )
        )

    path = codex_config_path or (Path.home() / ".codex" / "config.toml")
    if not path.exists():
        entries.append(
            _entry(
                "Codex goals feature flag",
                STATUS_WARN,
                "Codex config not found; goals flag not inspected",
                "Create or expose .codex/config.toml when Codex feature flags need auditing.",
                category,
            )
        )
        return entries
    if tomllib is None:
        entries.append(
            _entry("Codex goals feature flag", STATUS_WARN, "TOML parser unavailable", category=category)
        )
        return entries
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        entries.append(
            _entry(
                "Codex goals feature flag",
                STATUS_FAIL,
                f"Codex config could not be parsed: {type(exc).__name__}",
                "Fix .codex/config.toml syntax before relying on feature flag inspection.",
                category,
            )
        )
        return entries
    goals_flag_present = _contains_key_named(data, "goals")
    detail = "goals flag present" if goals_flag_present else "goals flag absent; defaults apply"
    entries.append(_entry("Codex goals feature flag", STATUS_OK, detail, category=category))
    return entries


def _check_cron_storage(home: Path) -> list[SystemDoctorEntry]:
    try:
        import cron.jobs  # noqa: F401
        import cron.scheduler  # noqa: F401
    except Exception as exc:
        return [
            _entry(
                "Cron storage",
                STATUS_WARN,
                f"cron modules unavailable: {type(exc).__name__}",
                category="Cron",
            )
        ]

    cron_dir = home / "cron"
    jobs_path = cron_dir / "jobs.json"
    if not cron_dir.exists():
        return [
            _entry(
                "Cron storage",
                STATUS_WARN,
                "cron directory not initialized",
                "This is normal until a cron job is created.",
                "Cron",
            )
        ]
    if not jobs_path.exists():
        return [_entry("Cron storage", STATUS_OK, "cron directory exists; no jobs.json yet", category="Cron")]
    jobs, error = _read_json_records(jobs_path, "jobs")
    if error:
        return [
            _entry(
                "Cron storage",
                STATUS_FAIL,
                error,
                "Repair or remove the profile-local cron/jobs.json file.",
                "Cron",
            )
        ]
    enabled = sum(1 for job in jobs if job.get("enabled", True))
    disabled = len(jobs) - enabled
    return [_entry("Cron storage", STATUS_OK, f"total={len(jobs)} enabled={enabled} disabled={disabled}", category="Cron")]


def _read_json_records(path: Path, key: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], f"{path.name} is invalid JSON: {exc.msg}"
    except OSError as exc:
        return [], f"{path.name} is unreadable: {type(exc).__name__}"

    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, Mapping):
        raw_items = data.get(key, [])
    else:
        return [], f"{path.name} must contain a JSON object or list"
    if not isinstance(raw_items, list):
        return [], f"{path.name}.{key} must be a list"
    return [dict(item) for item in raw_items if isinstance(item, Mapping)], None


def _count_pending_records(records: Iterable[Mapping[str, Any]], now: datetime) -> tuple[int, int]:
    active = 0
    expired = 0
    for record in records:
        status = str(record.get("status", ""))
        expires_at = _parse_time(record.get("expires_at"))
        is_expired = status == "expired" or (status == "open" and expires_at is not None and expires_at <= now)
        if status == "open" and not is_expired:
            active += 1
        elif is_expired:
            expired += 1
    return active, expired


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _contains_key_named(value: Any, key_name: str) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() == key_name.lower():
                return True
            if _contains_key_named(child, key_name):
                return True
    elif isinstance(value, list):
        return any(_contains_key_named(child, key_name) for child in value)
    return False


def _category_order(entries: Iterable[SystemDoctorEntry]) -> list[str]:
    seen: list[str] = []
    for entry in entries:
        if entry.category not in seen:
            seen.append(entry.category)
    return seen


def _entry(
    name: str,
    status: str,
    detail: str = "",
    remediation: str = "",
    category: str = "General",
) -> SystemDoctorEntry:
    return SystemDoctorEntry(
        name=name,
        status=status,
        detail=_sanitize(detail),
        remediation=_sanitize(remediation),
        category=category,
    )


def _table_cell(value: str) -> str:
    clean = _sanitize(str(value or "-")).replace("\n", " ").replace("\r", " ")
    return clean.replace("|", "\\|")


def _sanitize(value: str) -> str:
    text = str(value)
    text = _SECRET_NAME_RE.sub(lambda match: match.group(0).replace(match.group(1), "[REDACTED]"), text)
    text = _SECRET_TOKEN_RE.sub("[REDACTED]", text)
    text = _PRIVATE_KEY_RE.sub("-----BEGIN [REDACTED]-----", text)
    return text


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "STATUS_FAIL",
    "STATUS_OK",
    "STATUS_WARN",
    "SystemDoctorEntry",
    "SystemDoctorReport",
    "build_system_doctor_report",
    "print_system_doctor_dashboard",
    "render_system_doctor_dashboard",
]
