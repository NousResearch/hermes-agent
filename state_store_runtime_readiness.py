"""Fail-closed PostgreSQL runtime-activation readiness boundary.

The PostgreSQL StateStore slice is intentionally narrower than SessionDB.  This
module makes that gap measurable and prevents legacy ``state.db`` openers from
silently becoming a SQLite fallback when a profile explicitly selects PostgreSQL.
It does not open a database or create a Hermes home.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from state_store import StateStoreConfigurationError, resolve_state_store_config


_RUNTIME_SOURCE_PATHS = (
    "cli.py",
    "run_agent.py",
    "hermes_state.py",
    "hermes_state_registry.py",
    "gateway/session_persistence.py",
    "gateway/mirror.py",
    "gateway/shutdown_flush.py",
    "gateway/channel_directory.py",
    "gateway/run.py",
    "gateway/delivery_ledger.py",
    "tools/async_delegation.py",
    "tools/bot_live_delivery.py",
    "cron/scheduler.py",
    "tui_gateway/server.py",
    "tui_gateway/methods_session.py",
    "tui_gateway/methods_profiles.py",
    "tui_gateway/compute_host.py",
    "tui_gateway/session_workdir.py",
    "gateway/platforms/api_server.py",
    "tools/react_to_message_tool.py",
    "acp_adapter/session.py",
    "hermes_cli/approvals_suggest.py",
    "hermes_cli/doctor_state.py",
    "state_store.py",
)
_RAW_OPENER_CALLS = frozenset({"SessionDB", "acquire", "connect", "open_db"})
_BACKEND_NEUTRAL_CALLS = frozenset({"open_state_store"})


class PostgreSQLRuntimeActivationError(RuntimeError):
    """Selected PostgreSQL cannot safely enter a legacy SessionDB runtime."""

    def __init__(self, report: "RuntimeActivationReport") -> None:
        self.report = report
        missing = ", ".join(report.missing_capabilities)
        super().__init__(
            "PostgreSQL state_store.backend is selected for "
            f"{report.profile_home}, but full runtime activation is blocked before state.db access: "
            f"missing capabilities: {missing}. See {report.evidence_path}."
        )


class StateDbOpenAttempt(RuntimeError):
    """A trapped root/profile ``state.db`` opener attempted legacy SQLite access."""

    def __init__(self, event: "StateDbOpenEvent") -> None:
        self.event = event
        super().__init__(
            f"state.db open trapped at {event.path} via {event.caller_path}:{event.caller_symbol} "
            f"({event.operation})"
        )


@dataclass(frozen=True)
class RawStateDbOpener:
    path: str
    symbol: str
    operation: str
    classification: str


@dataclass(frozen=True)
class StateDbOpenEvent:
    path: str
    operation: str
    caller_path: str
    caller_symbol: str


@dataclass(frozen=True)
class RuntimeActivationReport:
    selected_backend: str
    profile_home: str
    profile_name: str | None
    tenant_schema: str | None
    supported_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    raw_state_db_openers: tuple[RawStateDbOpener, ...]
    evidence_path: str = "website/docs/developer-guide/state-store-postgresql-phase13-runtime-readiness.md"

    @property
    def ready(self) -> bool:
        return not self.missing_capabilities

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "raw_state_db_openers": [asdict(opener) for opener in self.raw_state_db_openers],
            "ready": self.ready,
        }


class _InventoryVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.openers: list[tuple[str, str]] = []

    def _push_scope(self, node: ast.AST) -> None:
        self.scope.append(getattr(node, "name", "<anonymous>"))
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _push_scope
    visit_AsyncFunctionDef = _push_scope
    visit_ClassDef = _push_scope

    def visit_Call(self, node: ast.Call) -> None:
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in _RAW_OPENER_CALLS | _BACKEND_NEUTRAL_CALLS:
            self.openers.append((".".join(self.scope) or "<module>", name))
        self.generic_visit(node)


def _default_source_root() -> Path:
    return Path(__file__).resolve().parent


def static_raw_state_db_inventory(source_root: Path | None = None) -> tuple[RawStateDbOpener, ...]:
    """AST inventory of approved runtime modules' raw SQLite acquisition calls.

    The inventory is intentionally module-scoped: it distinguishes runtime
    consumers from migration/test tooling and exposes every direct SessionDB,
    registry acquire, sqlite connect, or shared ``open_db`` boundary that still needs a neutral
    contract before PostgreSQL can be activated broadly.
    """
    root = (source_root or _default_source_root()).resolve()
    inventory: list[RawStateDbOpener] = []
    for relative_path in _RUNTIME_SOURCE_PATHS:
        path = root / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _InventoryVisitor()
        visitor.visit(tree)
        for symbol, operation in visitor.openers:
            classification = "backend-neutral-contextual-contract" if operation == "open_state_store" else "unported-legacy-runtime"
            inventory.append(RawStateDbOpener(relative_path, symbol, operation, classification))
    return tuple(sorted(inventory, key=lambda item: (item.path, item.symbol, item.operation)))


def write_runtime_callsite_report(path: Path, *, source_root: Path | None = None) -> RuntimeActivationReport:
    """Publish a deterministic machine-readable report without opening state storage."""
    report = inspect_runtime_activation({}, home=Path("/unselected"), source_root=source_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _read_profile_config(home: Path) -> Mapping[str, Any]:
    config_path = home / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StateStoreConfigurationError(f"cannot read state-store config for {home}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise StateStoreConfigurationError("profile config.yaml must be a mapping")
    return raw


def inspect_runtime_activation(
    config: Mapping[str, Any] | None = None,
    *,
    home: Path,
    source_root: Path | None = None,
    secret_lookup=None,
    include_inventory: bool = True,
) -> RuntimeActivationReport:
    """Build the selected-backend capability report without opening a database.

    PostgreSQL credentials are resolved only to validate that a selected backend
    has its required secret; neither the DSN nor any connection is retained.
    """
    from hermes_constants import profile_name_for_home, reset_hermes_home_override, set_hermes_home_override
    from state_store import postgresql_tenant_schema

    canonical_home = home.expanduser().resolve()
    raw_config = config if config is not None else _read_profile_config(canonical_home)
    resolved = resolve_state_store_config(raw_config, secret_lookup=secret_lookup)
    profile_name = profile_name_for_home(canonical_home)
    inventory = static_raw_state_db_inventory(source_root) if include_inventory else ()
    if resolved.backend == "sqlite":
        return RuntimeActivationReport(
            selected_backend="sqlite", profile_home=str(canonical_home), profile_name=profile_name,
            tenant_schema=None, supported_capabilities=("legacy-sessiondb-runtime",),
            missing_capabilities=(), raw_state_db_openers=inventory,
        )
    token = set_hermes_home_override(str(canonical_home))
    try:
        tenant_schema = postgresql_tenant_schema()
    finally:
        reset_hermes_home_override(token)
    return RuntimeActivationReport(
        selected_backend="postgresql", profile_home=str(canonical_home), profile_name=profile_name,
        tenant_schema=tenant_schema,
        supported_capabilities=(
            "narrow-state-store", "profile-derived-tenant-schema",
            "cli-fresh-resume-session-contract",
            "contextual-session-search-contract",
            "gateway-session-routing-transcript",
            "acp-session-transcript-lifecycle",
            "gateway-delivery-ledger-routing",
            "async-delegation-ledger-routing",
        ),
        missing_capabilities=(
            "cron-session-transcript-lifecycle",
            "tui-api-session-runtime",
        ),
        raw_state_db_openers=inventory,
    )


def require_legacy_state_db_runtime(config: Mapping[str, Any] | None = None, *, home: Path | None = None) -> RuntimeActivationReport:
    """Refuse a selected PG profile before any legacy SessionDB side effect.

    This is the safe bounded routing boundary: legacy callers retain SQLite
    behavior, while a selected PostgreSQL profile cannot open or create a root
    or profile ``state.db`` until its complete runtime contracts are ported.
    """
    from hermes_constants import get_hermes_home

    target_home = (home or get_hermes_home()).expanduser()
    report = inspect_runtime_activation(config, home=target_home, include_inventory=False)
    if report.selected_backend == "postgresql":
        report = inspect_runtime_activation(config, home=target_home, include_inventory=True)
        raise PostgreSQLRuntimeActivationError(report)
    return report


def _target_state_db(value: object, roots: tuple[Path, ...]) -> Path | None:
    try:
        path = Path(value)  # type: ignore[arg-type]
        path = path.expanduser().resolve()
    except (TypeError, ValueError, OSError):
        return None
    if path.name != "state.db":
        return None
    for root in roots:
        try:
            if path == root.resolve() / "state.db":
                return path
        except OSError:
            continue
    return None


def _caller() -> tuple[str, str]:
    for frame in inspect.stack()[2:]:
        filename = Path(frame.filename)
        if filename.name != Path(__file__).name:
            return str(filename), frame.function
    return "<unknown>", "<unknown>"


@contextmanager
def trap_state_db_opens(*roots: Path, fail_fast: bool = True) -> Iterator[list[StateDbOpenEvent]]:
    """Record root/profile ``state.db`` file-open attempts with exact caller data.

    ``sqlite3.connect`` covers the native SQLite opening path; ``open`` covers
    direct Python file operations.  The trap is test-only instrumentation and
    never changes production file-open behavior outside its context.
    """
    events: list[StateDbOpenEvent] = []
    normalized_roots = tuple(root.expanduser().resolve() for root in roots)
    original_connect, original_open = sqlite3.connect, builtins.open

    def record(value: object, operation: str) -> None:
        target = _target_state_db(value, normalized_roots)
        if target is None:
            return
        caller_path, caller_symbol = _caller()
        event = StateDbOpenEvent(str(target), operation, caller_path, caller_symbol)
        events.append(event)
        if fail_fast:
            raise StateDbOpenAttempt(event)

    def trapped_connect(database, *args, **kwargs):
        record(database, "sqlite3.connect")
        return original_connect(database, *args, **kwargs)

    def trapped_open(file, *args, **kwargs):
        record(file, "open")
        return original_open(file, *args, **kwargs)

    sqlite3.connect = trapped_connect
    builtins.open = trapped_open
    try:
        yield events
    finally:
        sqlite3.connect = original_connect
        builtins.open = original_open
