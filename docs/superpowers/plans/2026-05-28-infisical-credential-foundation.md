# Infisical Credential Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native Infisical-backed Hermes secret foundation with local self-hosted Infisical on this device, Tailscale-secured remote access, OS credential-store bootstrap, multi-project aliases, and full `.env` migration.

**Architecture:** Add a small `agent.secrets` package with a backend-neutral resolver, explicit secret references, an Infisical backend, and a bootstrap credential-store adapter. Add `hermes secrets` CLI commands for setup, validation, and migration, while bridging resolved values into legacy `os.getenv()` paths so existing providers and tools keep working during rollout.

**Tech Stack:** Python 3.11, `httpx` for Infisical HTTP calls, `keyring==25.7.0` lazy dependency for OS credential stores, `pytest`, `pyyaml`, existing Hermes config and `.env` helpers, Tailscale CLI preflight checks.

---

## Execution Prerequisites

The current worktree has unrelated user edits in:

- `agent/codex_runtime.py`
- `plugins/platforms/simplex/adapter.py`
- `tests/gateway/test_simplex_plugin.py`
- `tests/run_agent/test_run_agent_codex_responses.py`

Before executing code changes, create a dedicated feature worktree or branch and preserve those unrelated edits. Do not revert or stage them.

Recommended command shape:

```bash
git worktree add ../Hermes-Agent-infisical -b codex/infisical-credential-foundation HEAD
cd ../Hermes-Agent-infisical
```

Use `.venv` if present:

```bash
source .venv/bin/activate
```

If this checkout does not have `.venv`, use the repo fallback documented in `AGENTS.md` and run tests through `scripts/run_tests.sh` where practical.

## File Structure

Create focused files with these responsibilities:

- `agent/secrets/__init__.py`: public exports for resolver and errors.
- `agent/secrets/errors.py`: typed exceptions with redacted messages.
- `agent/secrets/references.py`: parse `alias:KEY` and unqualified secret names.
- `agent/secrets/config.py`: convert `config.yaml` `secrets` section into typed settings.
- `agent/secrets/backends/base.py`: protocol/base class for secret backends.
- `agent/secrets/backends/env.py`: legacy environment and `.env` backend.
- `agent/secrets/backends/infisical.py`: Universal Auth, token refresh, multi-project read/write/delete.
- `agent/secrets/bootstrap/base.py`: bootstrap credential-store protocol.
- `agent/secrets/bootstrap/memory_store.py`: deterministic test store.
- `agent/secrets/bootstrap/keyring_store.py`: OS credential store backed by `keyring`.
- `agent/secrets/resolver.py`: runtime resolution order and optional env injection.
- `hermes_cli/secrets_commands.py`: `hermes secrets` argparse dispatcher.
- `hermes_cli/secrets_setup.py`: local Infisical and Tailscale setup flow.
- `hermes_cli/secrets_migration.py`: `.env` scan, mapping, write, verify, backup, cleanup.
- `website/docs/user-guide/features/infisical-secrets.md`: user-facing setup guide.
- `tests/agent/secrets/`: unit tests for parser, config, resolver, bootstrap store, backend.
- `tests/hermes_cli/test_secrets_*.py`: CLI setup and migration tests.

Modify these existing files:

- `hermes_cli/config.py`: add `secrets` defaults and resolver-aware helpers.
- `hermes_cli/env_loader.py`: initialize resolver-backed environment injection after `.env` load.
- `hermes_cli/main.py`: register `hermes secrets` command.
- `tools/lazy_deps.py`: add lazy dependency entry for `keyring==25.7.0`.
- `pyproject.toml`: add an `infisical` optional extra with `keyring==25.7.0`.
- `agent/redact.py`: add Infisical token prefixes only if tests prove existing patterns miss them.
- `tests/agent/test_redact.py`: cover Universal Auth and Infisical token redaction.

Do not add a vendor copy of Infisical, a Docker compose file, or privileged service management.

## Task 1: Secret References, Errors, And Backend Interfaces

**Files:**
- Create: `agent/secrets/__init__.py`
- Create: `agent/secrets/errors.py`
- Create: `agent/secrets/references.py`
- Create: `agent/secrets/backends/__init__.py`
- Create: `agent/secrets/backends/base.py`
- Test: `tests/agent/secrets/test_references.py`

- [ ] **Step 1: Write failing reference parser tests**

Create `tests/agent/secrets/test_references.py`:

```python
import pytest

from agent.secrets.errors import InvalidSecretReference
from agent.secrets.references import SecretReference, parse_secret_reference


def test_parse_alias_qualified_reference():
    ref = parse_secret_reference("models:OPENAI_API_KEY")

    assert ref == SecretReference(alias="models", key="OPENAI_API_KEY")
    assert ref.qualified == "models:OPENAI_API_KEY"


def test_parse_unqualified_reference():
    ref = parse_secret_reference("OPENAI_API_KEY")

    assert ref == SecretReference(alias=None, key="OPENAI_API_KEY")
    assert ref.qualified == "OPENAI_API_KEY"


@pytest.mark.parametrize("value", ["", "models:", ":OPENAI_API_KEY", "mail:bad-key", "mail:bad key"])
def test_invalid_reference_rejected(value):
    with pytest.raises(InvalidSecretReference):
        parse_secret_reference(value)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest tests/agent/secrets/test_references.py -q
```

Expected: import failure because `agent.secrets.references` does not exist.

- [ ] **Step 3: Implement errors and references**

Create `agent/secrets/errors.py`:

```python
from __future__ import annotations


class SecretError(RuntimeError):
    """Base class for redacted secret-resolution errors."""


class InvalidSecretReference(SecretError, ValueError):
    """A secret reference cannot be parsed safely."""


class SecretNotFound(SecretError):
    """A named secret was not found in the selected source."""


class SecretBackendUnavailable(SecretError):
    """The configured secret backend cannot be reached."""


class SecretPermissionDenied(SecretError):
    """The configured identity cannot access the requested secret."""


class BootstrapCredentialMissing(SecretError):
    """Infisical bootstrap credentials are missing from the OS store."""
```

Create `agent/secrets/references.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass

from agent.secrets.errors import InvalidSecretReference


_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


@dataclass(frozen=True)
class SecretReference:
    alias: str | None
    key: str

    @property
    def qualified(self) -> str:
        return f"{self.alias}:{self.key}" if self.alias else self.key


def parse_secret_reference(value: str) -> SecretReference:
    raw = str(value or "").strip()
    if not raw:
        raise InvalidSecretReference("Secret reference is empty")

    if ":" in raw:
        alias, key = raw.split(":", 1)
        if not _ALIAS_RE.fullmatch(alias):
            raise InvalidSecretReference(f"Invalid secret project alias: {alias!r}")
        if not _KEY_RE.fullmatch(key):
            raise InvalidSecretReference(f"Invalid secret key name: {key!r}")
        return SecretReference(alias=alias, key=key)

    if not _KEY_RE.fullmatch(raw):
        raise InvalidSecretReference(f"Invalid secret key name: {raw!r}")
    return SecretReference(alias=None, key=raw)
```

Create `agent/secrets/backends/base.py`:

```python
from __future__ import annotations

from typing import Protocol

from agent.secrets.references import SecretReference


class SecretBackend(Protocol):
    def get_secret(self, ref: SecretReference) -> str:
        """Return a secret value or raise a SecretError subclass."""

    def set_secret(self, ref: SecretReference, value: str) -> None:
        """Create or update a secret value."""

    def delete_secret(self, ref: SecretReference) -> None:
        """Delete a secret value."""
```

Create package exports:

```python
# agent/secrets/__init__.py
from agent.secrets.errors import (
    BootstrapCredentialMissing,
    InvalidSecretReference,
    SecretBackendUnavailable,
    SecretError,
    SecretNotFound,
    SecretPermissionDenied,
)
from agent.secrets.references import SecretReference, parse_secret_reference

__all__ = [
    "BootstrapCredentialMissing",
    "InvalidSecretReference",
    "SecretBackendUnavailable",
    "SecretError",
    "SecretNotFound",
    "SecretPermissionDenied",
    "SecretReference",
    "parse_secret_reference",
]
```

```python
# agent/secrets/backends/__init__.py
from agent.secrets.backends.base import SecretBackend

__all__ = ["SecretBackend"]
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/agent/secrets/test_references.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/secrets tests/agent/secrets/test_references.py
git commit -m "feat: add secret reference primitives"
```

## Task 2: Secret Config Model And Defaults

**Files:**
- Create: `agent/secrets/config.py`
- Modify: `hermes_cli/config.py`
- Test: `tests/agent/secrets/test_config.py`
- Test: `tests/hermes_cli/test_secrets_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/agent/secrets/test_config.py`:

```python
from agent.secrets.config import InfisicalProjectConfig, SecretsConfig, load_secrets_config


def test_loads_infisical_backend_config():
    raw = {
        "secrets": {
            "active_backend": "infisical",
            "backends": {
                "infisical": {
                    "type": "infisical",
                    "host": "http://127.0.0.1:8080",
                    "environment": "prod",
                    "auth": {
                        "method": "universal_auth",
                        "bootstrap_store": "os_keyring",
                        "identity_name": "hermes-agent-default",
                    },
                    "projects": {
                        "models": {"project_id": "project-models", "path": "/hermes/models"}
                    },
                }
            },
            "default_aliases": {"OPENAI_API_KEY": "models"},
        }
    }

    cfg = load_secrets_config(raw)

    assert isinstance(cfg, SecretsConfig)
    assert cfg.active_backend == "infisical"
    assert cfg.infisical.host == "http://127.0.0.1:8080"
    assert cfg.infisical.projects["models"] == InfisicalProjectConfig(
        project_id="project-models",
        path="/hermes/models",
    )
    assert cfg.default_alias_for("OPENAI_API_KEY") == "models"


def test_empty_config_disables_backend():
    cfg = load_secrets_config({})

    assert cfg.active_backend == ""
    assert cfg.infisical is None
```

Create `tests/hermes_cli/test_secrets_config.py`:

```python
from hermes_cli.config import DEFAULT_CONFIG


def test_default_config_has_secrets_section():
    secrets = DEFAULT_CONFIG["secrets"]

    assert secrets["active_backend"] == ""
    assert secrets["backends"]["infisical"]["type"] == "infisical"
    assert secrets["backends"]["infisical"]["remote_access"]["mode"] == "tailscale"
    assert secrets["backends"]["infisical"]["remote_access"]["public_funnel_allowed"] is False
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
pytest tests/agent/secrets/test_config.py tests/hermes_cli/test_secrets_config.py -q
```

Expected: import failure for `agent.secrets.config` and missing `DEFAULT_CONFIG["secrets"]`.

- [ ] **Step 3: Implement typed config loader**

Create `agent/secrets/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InfisicalProjectConfig:
    project_id: str
    path: str = "/"


@dataclass(frozen=True)
class InfisicalAuthConfig:
    method: str = "universal_auth"
    bootstrap_store: str = "os_keyring"
    identity_name: str = "hermes-agent-default"


@dataclass(frozen=True)
class InfisicalRemoteAccessConfig:
    mode: str = "tailscale"
    tailnet_url: str = ""
    public_funnel_allowed: bool = False


@dataclass(frozen=True)
class InfisicalBackendConfig:
    host: str
    environment: str
    auth: InfisicalAuthConfig
    remote_access: InfisicalRemoteAccessConfig
    projects: dict[str, InfisicalProjectConfig]


@dataclass(frozen=True)
class SecretsConfig:
    active_backend: str = ""
    infisical: InfisicalBackendConfig | None = None
    default_aliases: dict[str, str] = field(default_factory=dict)

    def default_alias_for(self, key: str) -> str | None:
        return self.default_aliases.get(key)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_secrets_config(config: dict[str, Any]) -> SecretsConfig:
    raw = _as_dict(config.get("secrets"))
    active = str(raw.get("active_backend") or "").strip()
    default_aliases = {
        str(k): str(v)
        for k, v in _as_dict(raw.get("default_aliases")).items()
        if str(k).strip() and str(v).strip()
    }
    infisical_raw = _as_dict(_as_dict(raw.get("backends")).get("infisical"))
    infisical = None
    if infisical_raw:
        auth_raw = _as_dict(infisical_raw.get("auth"))
        remote_raw = _as_dict(infisical_raw.get("remote_access"))
        projects = {
            str(alias): InfisicalProjectConfig(
                project_id=str(project.get("project_id") or "").strip(),
                path=str(project.get("path") or "/").strip() or "/",
            )
            for alias, project in _as_dict(infisical_raw.get("projects")).items()
            if isinstance(project, dict)
        }
        infisical = InfisicalBackendConfig(
            host=str(infisical_raw.get("host") or "").strip(),
            environment=str(infisical_raw.get("environment") or "").strip(),
            auth=InfisicalAuthConfig(
                method=str(auth_raw.get("method") or "universal_auth").strip(),
                bootstrap_store=str(auth_raw.get("bootstrap_store") or "os_keyring").strip(),
                identity_name=str(auth_raw.get("identity_name") or "hermes-agent-default").strip(),
            ),
            remote_access=InfisicalRemoteAccessConfig(
                mode=str(remote_raw.get("mode") or "tailscale").strip(),
                tailnet_url=str(remote_raw.get("tailnet_url") or "").strip(),
                public_funnel_allowed=bool(remote_raw.get("public_funnel_allowed", False)),
            ),
            projects=projects,
        )
    return SecretsConfig(active_backend=active, infisical=infisical, default_aliases=default_aliases)
```

- [ ] **Step 4: Add `secrets` defaults**

Modify `hermes_cli/config.py` inside `DEFAULT_CONFIG` near other top-level config sections:

```python
    "secrets": {
        "active_backend": "",
        "default_aliases": {
            "OPENAI_API_KEY": "models",
            "OPENROUTER_API_KEY": "models",
            "ANTHROPIC_API_KEY": "models",
            "ANTHROPIC_TOKEN": "models",
            "GOOGLE_API_KEY": "models",
            "GEMINI_API_KEY": "models",
            "GMAIL_CLIENT_SECRET": "mail",
            "GOOGLE_REFRESH_TOKEN": "mail",
            "SIMPLEX_PRIVATE_KEY": "chat",
            "TELEGRAM_BOT_TOKEN": "gateway",
            "DISCORD_BOT_TOKEN": "gateway",
            "SLACK_BOT_TOKEN": "gateway",
        },
        "backends": {
            "infisical": {
                "type": "infisical",
                "host": "http://127.0.0.1:8080",
                "remote_access": {
                    "mode": "tailscale",
                    "tailnet_url": "",
                    "public_funnel_allowed": False,
                },
                "environment": "prod",
                "auth": {
                    "method": "universal_auth",
                    "bootstrap_store": "os_keyring",
                    "identity_name": "hermes-agent-default",
                },
                "projects": {},
            },
        },
    },
```

Do not bump `_config_version` for this additive key.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/agent/secrets/test_config.py tests/hermes_cli/test_secrets_config.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/secrets/config.py hermes_cli/config.py tests/agent/secrets/test_config.py tests/hermes_cli/test_secrets_config.py
git commit -m "feat: add secret backend config"
```

## Task 3: Resolver And Legacy Environment Backend

**Files:**
- Create: `agent/secrets/backends/env.py`
- Create: `agent/secrets/resolver.py`
- Test: `tests/agent/secrets/test_resolver.py`

- [ ] **Step 1: Write failing resolver tests**

Create `tests/agent/secrets/test_resolver.py`:

```python
import os

from agent.secrets.backends.env import EnvSecretBackend
from agent.secrets.config import SecretsConfig
from agent.secrets.references import SecretReference
from agent.secrets.resolver import SecretResolver


class DictBackend:
    def __init__(self, values):
        self.values = values

    def get_secret(self, ref: SecretReference) -> str:
        return self.values[ref.qualified]

    def set_secret(self, ref: SecretReference, value: str) -> None:
        self.values[ref.qualified] = value

    def delete_secret(self, ref: SecretReference) -> None:
        self.values.pop(ref.qualified, None)


def test_runtime_env_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    resolver = SecretResolver(
        config=SecretsConfig(active_backend="infisical", default_aliases={"OPENAI_API_KEY": "models"}),
        active_backend=DictBackend({"models:OPENAI_API_KEY": "from-vault"}),
        env_backend=EnvSecretBackend({"OPENAI_API_KEY": "from-dotenv"}),
    )

    assert resolver.resolve("OPENAI_API_KEY") == "from-env"


def test_unqualified_name_uses_default_alias_before_dotenv(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolver = SecretResolver(
        config=SecretsConfig(active_backend="infisical", default_aliases={"OPENAI_API_KEY": "models"}),
        active_backend=DictBackend({"models:OPENAI_API_KEY": "from-vault"}),
        env_backend=EnvSecretBackend({"OPENAI_API_KEY": "from-dotenv"}),
    )

    assert resolver.resolve("OPENAI_API_KEY") == "from-vault"


def test_explicit_alias_does_not_search_dotenv(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolver = SecretResolver(
        config=SecretsConfig(active_backend="infisical"),
        active_backend=DictBackend({"models:OPENAI_API_KEY": "from-vault"}),
        env_backend=EnvSecretBackend({"OPENAI_API_KEY": "from-dotenv"}),
    )

    assert resolver.resolve("models:OPENAI_API_KEY") == "from-vault"


def test_inject_known_values_sets_process_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolver = SecretResolver(
        config=SecretsConfig(active_backend="infisical", default_aliases={"OPENAI_API_KEY": "models"}),
        active_backend=DictBackend({"models:OPENAI_API_KEY": "from-vault"}),
        env_backend=EnvSecretBackend({}),
    )

    injected = resolver.inject(["OPENAI_API_KEY"])

    assert injected == {"OPENAI_API_KEY": "from-vault"}
    assert os.environ["OPENAI_API_KEY"] == "from-vault"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/agent/secrets/test_resolver.py -q
```

Expected: import failure for `agent.secrets.resolver`.

- [ ] **Step 3: Implement env backend**

Create `agent/secrets/backends/env.py`:

```python
from __future__ import annotations

from agent.secrets.errors import SecretNotFound
from agent.secrets.references import SecretReference


class EnvSecretBackend:
    def __init__(self, values: dict[str, str] | None = None):
        self._values = dict(values or {})

    def get_secret(self, ref: SecretReference) -> str:
        if ref.alias:
            raise SecretNotFound(f"Secret {ref.qualified} not found in environment backend")
        value = self._values.get(ref.key)
        if value:
            return value
        raise SecretNotFound(f"Secret {ref.key} not found in environment backend")

    def set_secret(self, ref: SecretReference, value: str) -> None:
        if ref.alias:
            raise SecretNotFound(f"Cannot write qualified secret {ref.qualified} to environment backend")
        self._values[ref.key] = value

    def delete_secret(self, ref: SecretReference) -> None:
        if not ref.alias:
            self._values.pop(ref.key, None)
```

- [ ] **Step 4: Implement resolver**

Create `agent/secrets/resolver.py`:

```python
from __future__ import annotations

import os
from collections.abc import Iterable

from agent.secrets.backends.base import SecretBackend
from agent.secrets.config import SecretsConfig
from agent.secrets.errors import SecretError, SecretNotFound
from agent.secrets.references import SecretReference, parse_secret_reference


class SecretResolver:
    def __init__(
        self,
        *,
        config: SecretsConfig,
        active_backend: SecretBackend | None,
        env_backend: SecretBackend,
    ):
        self.config = config
        self.active_backend = active_backend
        self.env_backend = env_backend

    def resolve(self, name: str) -> str:
        ref = parse_secret_reference(name)

        if not ref.alias and os.environ.get(ref.key):
            return os.environ[ref.key]

        if ref.alias:
            if not self.active_backend:
                raise SecretNotFound(f"Secret backend is not configured for {ref.qualified}")
            return self.active_backend.get_secret(ref)

        default_alias = self.config.default_alias_for(ref.key)
        if default_alias and self.active_backend:
            try:
                return self.active_backend.get_secret(SecretReference(alias=default_alias, key=ref.key))
            except SecretNotFound:
                pass

        return self.env_backend.get_secret(ref)

    def inject(self, names: Iterable[str]) -> dict[str, str]:
        injected: dict[str, str] = {}
        for name in names:
            ref = parse_secret_reference(name)
            if ref.alias:
                continue
            try:
                value = self.resolve(ref.key)
            except SecretError:
                continue
            if value:
                os.environ[ref.key] = value
                injected[ref.key] = value
        return injected
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/agent/secrets/test_resolver.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/secrets/backends/env.py agent/secrets/resolver.py tests/agent/secrets/test_resolver.py
git commit -m "feat: add secret resolver"
```

## Task 4: Bootstrap Credential Store

**Files:**
- Create: `agent/secrets/bootstrap/__init__.py`
- Create: `agent/secrets/bootstrap/base.py`
- Create: `agent/secrets/bootstrap/memory_store.py`
- Create: `agent/secrets/bootstrap/keyring_store.py`
- Modify: `tools/lazy_deps.py`
- Modify: `pyproject.toml`
- Test: `tests/agent/secrets/test_bootstrap_store.py`

- [ ] **Step 1: Write failing bootstrap tests**

Create `tests/agent/secrets/test_bootstrap_store.py`:

```python
import sys
import types

import pytest

from agent.secrets.bootstrap.base import BootstrapCredentials
from agent.secrets.bootstrap.keyring_store import KeyringBootstrapStore
from agent.secrets.bootstrap.memory_store import MemoryBootstrapStore


def test_memory_store_round_trip():
    store = MemoryBootstrapStore()
    creds = BootstrapCredentials(client_id="client-id", client_secret="client-secret")

    store.set("hermes-agent-default", creds)

    assert store.get("hermes-agent-default") == creds


def test_memory_store_missing_identity_raises():
    store = MemoryBootstrapStore()

    with pytest.raises(KeyError):
        store.get("missing")


def test_keyring_store_round_trip_with_fake_module(monkeypatch):
    values = {}

    fake_keyring = types.SimpleNamespace(
        get_password=lambda service, user: values.get((service, user)),
        set_password=lambda service, user, password: values.__setitem__((service, user), password),
        delete_password=lambda service, user: values.pop((service, user), None),
    )
    monkeypatch.setitem(sys.modules, "keyring", fake_keyring)

    store = KeyringBootstrapStore(service_name="hermes-test")
    creds = BootstrapCredentials(client_id="client-id", client_secret="client-secret")

    store.set("hermes-agent-default", creds)

    assert store.get("hermes-agent-default") == creds
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/agent/secrets/test_bootstrap_store.py -q
```

Expected: import failure for `agent.secrets.bootstrap`.

- [ ] **Step 3: Implement bootstrap base and memory store**

Create `agent/secrets/bootstrap/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BootstrapCredentials:
    client_id: str
    client_secret: str


class BootstrapStore(Protocol):
    def get(self, identity_name: str) -> BootstrapCredentials:
        """Load bootstrap credentials for an Infisical identity."""

    def set(self, identity_name: str, credentials: BootstrapCredentials) -> None:
        """Store bootstrap credentials for an Infisical identity."""

    def delete(self, identity_name: str) -> None:
        """Delete bootstrap credentials for an Infisical identity."""
```

Create `agent/secrets/bootstrap/memory_store.py`:

```python
from __future__ import annotations

from agent.secrets.bootstrap.base import BootstrapCredentials


class MemoryBootstrapStore:
    def __init__(self):
        self._values: dict[str, BootstrapCredentials] = {}

    def get(self, identity_name: str) -> BootstrapCredentials:
        return self._values[identity_name]

    def set(self, identity_name: str, credentials: BootstrapCredentials) -> None:
        self._values[identity_name] = credentials

    def delete(self, identity_name: str) -> None:
        self._values.pop(identity_name, None)
```

Create `agent/secrets/bootstrap/__init__.py`:

```python
from agent.secrets.bootstrap.base import BootstrapCredentials, BootstrapStore
from agent.secrets.bootstrap.memory_store import MemoryBootstrapStore

__all__ = ["BootstrapCredentials", "BootstrapStore", "MemoryBootstrapStore"]
```

- [ ] **Step 4: Add lazy dependency metadata for keyring**

Modify `tools/lazy_deps.py` inside `LAZY_DEPS`:

```python
    # Secret backends
    "secrets.keyring": ("keyring==25.7.0",),
```

Modify `pyproject.toml` in `[project.optional-dependencies]`:

```toml
infisical = ["keyring==25.7.0"]
```

- [ ] **Step 5: Implement keyring store**

Create `agent/secrets/bootstrap/keyring_store.py`:

```python
from __future__ import annotations

import json

from agent.secrets.bootstrap.base import BootstrapCredentials
from agent.secrets.errors import BootstrapCredentialMissing, SecretBackendUnavailable


_SERVICE_NAME = "hermes-agent.infisical"


class KeyringBootstrapStore:
    def __init__(self, service_name: str = _SERVICE_NAME):
        try:
            import keyring
        except ImportError:
            from tools.lazy_deps import FeatureUnavailable, ensure

            try:
                ensure("secrets.keyring")
                import keyring
            except FeatureUnavailable as exc:
                raise SecretBackendUnavailable(str(exc)) from exc
        self._keyring = keyring
        self._service_name = service_name

    def get(self, identity_name: str) -> BootstrapCredentials:
        raw = self._keyring.get_password(self._service_name, identity_name)
        if not raw:
            raise BootstrapCredentialMissing(
                f"Missing Infisical bootstrap credentials for identity {identity_name!r}"
            )
        data = json.loads(raw)
        return BootstrapCredentials(
            client_id=str(data["client_id"]),
            client_secret=str(data["client_secret"]),
        )

    def set(self, identity_name: str, credentials: BootstrapCredentials) -> None:
        payload = json.dumps(
            {"client_id": credentials.client_id, "client_secret": credentials.client_secret},
            separators=(",", ":"),
        )
        self._keyring.set_password(self._service_name, identity_name, payload)

    def delete(self, identity_name: str) -> None:
        try:
            self._keyring.delete_password(self._service_name, identity_name)
        except Exception:
            return
```

- [ ] **Step 6: Run bootstrap tests**

Run:

```bash
pytest tests/agent/secrets/test_bootstrap_store.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Run dependency metadata check**

Run:

```bash
python - <<'PY'
from tools.lazy_deps import LAZY_DEPS
assert LAZY_DEPS["secrets.keyring"] == ("keyring==25.7.0",)
PY
```

Expected: command exits with status 0.

- [ ] **Step 8: Commit**

```bash
git add agent/secrets/bootstrap tools/lazy_deps.py pyproject.toml tests/agent/secrets/test_bootstrap_store.py
git commit -m "feat: add bootstrap credential store"
```

## Task 5: Infisical HTTP Backend

**Files:**
- Create: `agent/secrets/backends/infisical.py`
- Test: `tests/agent/secrets/test_infisical_backend.py`

- [ ] **Step 1: Write failing Infisical backend tests**

Create `tests/agent/secrets/test_infisical_backend.py`:

```python
import httpx
import pytest

from agent.secrets.backends.infisical import InfisicalBackend
from agent.secrets.bootstrap.base import BootstrapCredentials
from agent.secrets.bootstrap.memory_store import MemoryBootstrapStore
from agent.secrets.config import (
    InfisicalAuthConfig,
    InfisicalBackendConfig,
    InfisicalProjectConfig,
    InfisicalRemoteAccessConfig,
)
from agent.secrets.errors import SecretNotFound
from agent.secrets.references import SecretReference


def make_backend(handler):
    store = MemoryBootstrapStore()
    store.set("hermes-agent-default", BootstrapCredentials("client-id", "client-secret"))
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://infisical.local")
    cfg = InfisicalBackendConfig(
        host="http://infisical.local",
        environment="prod",
        auth=InfisicalAuthConfig(identity_name="hermes-agent-default"),
        remote_access=InfisicalRemoteAccessConfig(),
        projects={"models": InfisicalProjectConfig(project_id="project-models", path="/hermes/models")},
    )
    return InfisicalBackend(config=cfg, bootstrap_store=store, client=client)


def test_get_secret_uses_universal_auth_and_project_alias():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, dict(request.headers)))
        if request.url.path == "/api/v1/auth/universal-auth/login":
            return httpx.Response(200, json={"accessToken": "access-token", "expiresIn": 3600})
        if request.url.path == "/api/v3/secrets/raw/OPENAI_API_KEY":
            assert request.headers["authorization"] == "Bearer access-token"
            assert request.url.params["workspaceId"] == "project-models"
            assert request.url.params["environment"] == "prod"
            assert request.url.params["secretPath"] == "/hermes/models"
            return httpx.Response(200, json={"secret": {"secretValue": "sk-test-value"}})
        return httpx.Response(404, json={"message": "not found"})

    backend = make_backend(handler)

    assert backend.get_secret(SecretReference(alias="models", key="OPENAI_API_KEY")) == "sk-test-value"
    assert requests[0][1] == "/api/v1/auth/universal-auth/login"


def test_missing_secret_maps_to_secret_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/universal-auth/login":
            return httpx.Response(200, json={"accessToken": "access-token", "expiresIn": 3600})
        return httpx.Response(404, json={"message": "missing"})

    backend = make_backend(handler)

    with pytest.raises(SecretNotFound):
        backend.get_secret(SecretReference(alias="models", key="MISSING_SECRET"))
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/agent/secrets/test_infisical_backend.py -q
```

Expected: import failure for `agent.secrets.backends.infisical`.

- [ ] **Step 3: Implement Infisical backend**

Create `agent/secrets/backends/infisical.py`:

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from agent.secrets.bootstrap.base import BootstrapStore
from agent.secrets.config import InfisicalBackendConfig
from agent.secrets.errors import (
    SecretBackendUnavailable,
    SecretNotFound,
    SecretPermissionDenied,
)
from agent.secrets.references import SecretReference


@dataclass
class _AccessToken:
    value: str
    expires_at: float

    def fresh(self) -> bool:
        return bool(self.value) and time.time() < self.expires_at - 30


class InfisicalBackend:
    def __init__(
        self,
        *,
        config: InfisicalBackendConfig,
        bootstrap_store: BootstrapStore,
        client: httpx.Client | None = None,
    ):
        self.config = config
        self.bootstrap_store = bootstrap_store
        self.client = client or httpx.Client(base_url=config.host, timeout=15.0)
        self._token = _AccessToken("", 0)

    def _bearer(self) -> str:
        if self._token.fresh():
            return self._token.value
        creds = self.bootstrap_store.get(self.config.auth.identity_name)
        response = self.client.post(
            "/api/v1/auth/universal-auth/login",
            json={"clientId": creds.client_id, "clientSecret": creds.client_secret},
        )
        if response.status_code >= 400:
            raise SecretPermissionDenied("Infisical Universal Auth login failed")
        data = response.json()
        token = str(data.get("accessToken") or "")
        if not token:
            raise SecretBackendUnavailable("Infisical Universal Auth response did not include an access token")
        expires_in = int(data.get("expiresIn") or 3600)
        self._token = _AccessToken(token, time.time() + expires_in)
        return token

    def _params(self, ref: SecretReference) -> dict[str, str]:
        if not ref.alias or ref.alias not in self.config.projects:
            raise SecretNotFound(f"Unknown Infisical project alias {ref.alias!r}")
        project = self.config.projects[ref.alias]
        return {
            "workspaceId": project.project_id,
            "environment": self.config.environment,
            "secretPath": project.path,
        }

    def _request(self, method: str, path: str, *, ref: SecretReference, json: dict | None = None) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._bearer()}"}
        response = self.client.request(method, path, headers=headers, params=self._params(ref), json=json)
        if response.status_code == 401:
            self._token = _AccessToken("", 0)
            headers = {"Authorization": f"Bearer {self._bearer()}"}
            response = self.client.request(method, path, headers=headers, params=self._params(ref), json=json)
        return response

    def get_secret(self, ref: SecretReference) -> str:
        response = self._request("GET", f"/api/v3/secrets/raw/{ref.key}", ref=ref)
        if response.status_code == 404:
            raise SecretNotFound(f"Secret {ref.qualified} not found")
        if response.status_code == 403:
            raise SecretPermissionDenied(f"Permission denied for Infisical alias {ref.alias!r}")
        if response.status_code >= 400:
            raise SecretBackendUnavailable(f"Infisical returned HTTP {response.status_code} for {ref.qualified}")
        data = response.json()
        value = data.get("secret", {}).get("secretValue")
        if not value:
            raise SecretNotFound(f"Secret {ref.qualified} has no value")
        return str(value)

    def set_secret(self, ref: SecretReference, value: str) -> None:
        body = {"secretName": ref.key, "secretValue": value, "type": "shared"}
        response = self._request("POST", "/api/v3/secrets/raw", ref=ref, json=body)
        if response.status_code in {400, 409}:
            response = self._request("PATCH", f"/api/v3/secrets/raw/{ref.key}", ref=ref, json=body)
        if response.status_code in {403, 401}:
            raise SecretPermissionDenied(f"Permission denied writing Infisical alias {ref.alias!r}")
        if response.status_code >= 400:
            raise SecretBackendUnavailable(f"Infisical returned HTTP {response.status_code} while writing {ref.qualified}")

    def delete_secret(self, ref: SecretReference) -> None:
        response = self._request("DELETE", f"/api/v3/secrets/raw/{ref.key}", ref=ref)
        if response.status_code not in {200, 204, 404}:
            raise SecretBackendUnavailable(f"Infisical returned HTTP {response.status_code} while deleting {ref.qualified}")
```

- [ ] **Step 4: Run backend tests**

Run:

```bash
pytest tests/agent/secrets/test_infisical_backend.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/secrets/backends/infisical.py tests/agent/secrets/test_infisical_backend.py
git commit -m "feat: add infisical secret backend"
```

## Task 6: CLI Setup And Tailscale Preflight

**Files:**
- Create: `hermes_cli/secrets_setup.py`
- Create: `hermes_cli/secrets_commands.py`
- Modify: `hermes_cli/main.py`
- Test: `tests/hermes_cli/test_secrets_setup.py`
- Test: `tests/hermes_cli/test_secrets_commands.py`

- [ ] **Step 1: Write failing setup preflight tests**

Create `tests/hermes_cli/test_secrets_setup.py`:

```python
import subprocess

import pytest

from hermes_cli.secrets_setup import PublicExposureError, validate_tailscale_remote_access


def test_rejects_funnel_like_public_url():
    with pytest.raises(PublicExposureError):
        validate_tailscale_remote_access("https://example.com", run=lambda *a, **k: None)


def test_accepts_tailnet_url_when_tailscale_status_succeeds():
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="100.64.0.1 hermes macOS\n", stderr="")

    validate_tailscale_remote_access("https://hermes-infisical.tailnet-name.ts.net", run=fake_run)

    assert calls[0] == ["tailscale", "status"]
```

Create `tests/hermes_cli/test_secrets_commands.py`:

```python
import argparse

from hermes_cli.secrets_commands import register_secrets_subparser


def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    return parser, subparsers


def test_registers_secrets_setup_and_migrate():
    parser, subparsers = make_parser()
    register_secrets_subparser(subparsers)

    args = parser.parse_args(["secrets", "setup", "infisical", "--non-interactive"])

    assert args.command == "secrets"
    assert args.secrets_action == "setup"
    assert args.backend == "infisical"
    assert args.non_interactive is True
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/hermes_cli/test_secrets_setup.py tests/hermes_cli/test_secrets_commands.py -q
```

Expected: import failure for `hermes_cli.secrets_setup`.

- [ ] **Step 3: Implement Tailscale and local preflight helpers**

Create `hermes_cli/secrets_setup.py`:

```python
from __future__ import annotations

import subprocess
from collections.abc import Callable
from urllib.parse import urlparse


class PublicExposureError(ValueError):
    pass


def _is_tailnet_host(hostname: str) -> bool:
    return hostname.endswith(".ts.net") or ".tailscale." in hostname


def validate_tailscale_remote_access(
    url: str,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if not _is_tailnet_host(hostname):
        raise PublicExposureError("Infisical remote access must use a tailnet-only URL")
    result = run(["tailscale", "status"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("Tailscale is not available or this device is not joined to a tailnet")


def validate_local_infisical_host(host: str) -> None:
    parsed = urlparse(host)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Local self-hosted Infisical must use localhost or 127.0.0.1")
```

- [ ] **Step 4: Implement command registration**

Create `hermes_cli/secrets_commands.py`:

```python
from __future__ import annotations


def register_secrets_subparser(subparsers):
    parser = subparsers.add_parser("secrets", help="Manage external secret backends")
    nested = parser.add_subparsers(dest="secrets_action")

    setup = nested.add_parser("setup", help="Configure a secret backend")
    setup.add_argument("backend", choices=["infisical"])
    setup.add_argument("--non-interactive", action="store_true")
    setup.add_argument("--host", default="")
    setup.add_argument("--environment", default="prod")
    setup.add_argument("--identity-name", default="hermes-agent-default")
    setup.add_argument("--tailnet-url", default="")

    migrate = nested.add_parser("migrate", help="Migrate local secrets into a backend")
    migrate.add_argument("backend", choices=["infisical"])
    migrate.add_argument("--dry-run", action="store_true", default=False)
    migrate.add_argument("--yes", action="store_true", default=False)

    parser.set_defaults(func=cmd_secrets)
    setup.set_defaults(func=cmd_secrets)
    migrate.set_defaults(func=cmd_secrets)
    return parser


def cmd_secrets(args):
    action = getattr(args, "secrets_action", None)
    if action == "setup":
        from hermes_cli.secrets_setup import setup_infisical

        return setup_infisical(args)
    if action == "migrate":
        from hermes_cli.secrets_migration import migrate_infisical

        return migrate_infisical(args)
    raise SystemExit("usage: hermes secrets {setup,migrate} ...")
```

Add a minimal setup implementation in `hermes_cli/secrets_setup.py` so the command imports and validates the remote-access guard:

```python
def setup_infisical(args):
    host = args.host or "http://127.0.0.1:8080"
    validate_local_infisical_host(host)
    if args.tailnet_url:
        validate_tailscale_remote_access(args.tailnet_url)
    return 0
```

- [ ] **Step 5: Register parser in `hermes_cli/main.py`**

In `hermes_cli/main.py`, near the `auth` or `status` command registration:

```python
    from hermes_cli.secrets_commands import register_secrets_subparser
    register_secrets_subparser(subparsers)
```

- [ ] **Step 6: Run setup command tests**

Run:

```bash
pytest tests/hermes_cli/test_secrets_setup.py tests/hermes_cli/test_secrets_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add hermes_cli/secrets_setup.py hermes_cli/secrets_commands.py hermes_cli/main.py tests/hermes_cli/test_secrets_setup.py tests/hermes_cli/test_secrets_commands.py
git commit -m "feat: add infisical secrets commands"
```

## Task 7: Migration Planner

**Files:**
- Create: `hermes_cli/secrets_migration.py`
- Test: `tests/hermes_cli/test_secrets_migration.py`

- [ ] **Step 1: Write failing migration planner tests**

Create `tests/hermes_cli/test_secrets_migration.py`:

```python
from pathlib import Path

from hermes_cli.secrets_migration import MigrationPlan, plan_env_migration


def test_plan_env_migration_groups_known_keys(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-model\n"
        "GMAIL_CLIENT_SECRET=gmail-secret\n"
        "TELEGRAM_BOT_TOKEN=123456789:telegram-secret\n"
        "PLAIN_SETTING=not-a-secret\n",
        encoding="utf-8",
    )

    plan = plan_env_migration(env_path, default_aliases={
        "OPENAI_API_KEY": "models",
        "GMAIL_CLIENT_SECRET": "mail",
        "TELEGRAM_BOT_TOKEN": "gateway",
    })

    assert plan == MigrationPlan(entries={
        "OPENAI_API_KEY": "models",
        "GMAIL_CLIENT_SECRET": "mail",
        "TELEGRAM_BOT_TOKEN": "gateway",
    })


def test_plan_env_migration_ignores_blank_and_comments(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("# comment\n\nOPENROUTER_API_KEY=\n", encoding="utf-8")

    plan = plan_env_migration(env_path, default_aliases={"OPENROUTER_API_KEY": "models"})

    assert plan.entries == {}
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/hermes_cli/test_secrets_migration.py -q
```

Expected: import failure for `hermes_cli.secrets_migration`.

- [ ] **Step 3: Implement migration planner**

Create `hermes_cli/secrets_migration.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MigrationPlan:
    entries: dict[str, str]


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and value:
            values[key] = value
    return values


def plan_env_migration(path: Path, *, default_aliases: dict[str, str]) -> MigrationPlan:
    values = _parse_env_file(path)
    entries = {
        key: alias
        for key, alias in default_aliases.items()
        if values.get(key) and alias
    }
    return MigrationPlan(entries=entries)


def migrate_infisical(args):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config

    config = load_config()
    aliases = config.get("secrets", {}).get("default_aliases", {})
    plan = plan_env_migration(get_hermes_home() / ".env", default_aliases=aliases)
    if args.dry_run:
        for key, alias in sorted(plan.entries.items()):
            print(f"{key} -> {alias}:{key}")
        return 0
    raise SystemExit("Run `hermes secrets migrate infisical --dry-run` first, then rerun with --yes")
```

- [ ] **Step 4: Run planner tests**

Run:

```bash
pytest tests/hermes_cli/test_secrets_migration.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/secrets_migration.py tests/hermes_cli/test_secrets_migration.py
git commit -m "feat: plan env secret migration"
```

## Task 8: Migration Write, Verify, Backup, And Cleanup

**Files:**
- Modify: `hermes_cli/secrets_migration.py`
- Test: `tests/hermes_cli/test_secrets_migration.py`

- [ ] **Step 1: Add failing migration execution tests**

Append to `tests/hermes_cli/test_secrets_migration.py`:

```python
from agent.secrets.references import SecretReference
from hermes_cli.secrets_migration import execute_migration


class RecordingBackend:
    def __init__(self):
        self.values = {}

    def get_secret(self, ref: SecretReference) -> str:
        return self.values[ref.qualified]

    def set_secret(self, ref: SecretReference, value: str) -> None:
        self.values[ref.qualified] = value

    def delete_secret(self, ref: SecretReference) -> None:
        self.values.pop(ref.qualified, None)


def test_execute_migration_writes_verifies_backs_up_and_cleans(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# keep comment\nOPENAI_API_KEY=sk-model\nPLAIN_SETTING=not-a-secret\n",
        encoding="utf-8",
    )
    backend = RecordingBackend()
    plan = MigrationPlan(entries={"OPENAI_API_KEY": "models"})

    result = execute_migration(env_path, plan=plan, backend=backend)

    assert backend.values["models:OPENAI_API_KEY"] == "sk-model"
    assert result.backup_path.exists()
    assert "OPENAI_API_KEY=" not in env_path.read_text(encoding="utf-8")
    assert "PLAIN_SETTING=not-a-secret" in env_path.read_text(encoding="utf-8")
    assert "# keep comment" in env_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the specific test and confirm failure**

Run:

```bash
pytest tests/hermes_cli/test_secrets_migration.py::test_execute_migration_writes_verifies_backs_up_and_cleans -q
```

Expected: import failure for `execute_migration`.

- [ ] **Step 3: Implement migration execution**

Modify `hermes_cli/secrets_migration.py`:

```python
from dataclasses import dataclass
from datetime import datetime, timezone

from agent.secrets.backends.base import SecretBackend
from agent.secrets.references import SecretReference


@dataclass(frozen=True)
class MigrationResult:
    migrated: dict[str, str]
    backup_path: Path


def _backup_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(f"{path.name}.{stamp}.bak")


def _remove_keys_from_env(path: Path, keys: set[str]) -> None:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines(keepends=True)
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in keys:
                continue
        kept.append(line)
    path.write_text("".join(kept), encoding="utf-8")


def execute_migration(path: Path, *, plan: MigrationPlan, backend: SecretBackend) -> MigrationResult:
    values = _parse_env_file(path)
    migrated: dict[str, str] = {}

    for key, alias in plan.entries.items():
        value = values[key]
        ref = SecretReference(alias=alias, key=key)
        backend.set_secret(ref, value)
        verified = backend.get_secret(ref)
        if verified != value:
            raise RuntimeError(f"Verification failed for migrated secret {alias}:{key}")
        migrated[key] = alias

    backup = _backup_path(path)
    backup.write_text(path.read_text(encoding="utf-8-sig", errors="replace"), encoding="utf-8")
    _remove_keys_from_env(path, set(migrated))
    return MigrationResult(migrated=migrated, backup_path=backup)
```

Ensure imports at the top of `hermes_cli/secrets_migration.py` are consolidated and no duplicate `dataclass` import remains.

- [ ] **Step 4: Run migration tests**

Run:

```bash
pytest tests/hermes_cli/test_secrets_migration.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/secrets_migration.py tests/hermes_cli/test_secrets_migration.py
git commit -m "feat: migrate env secrets into backend"
```

## Task 9: Migration Command Wiring

**Files:**
- Modify: `hermes_cli/secrets_migration.py`
- Test: `tests/hermes_cli/test_secrets_migration.py`

- [ ] **Step 1: Add failing migration command tests**

Append to `tests/hermes_cli/test_secrets_migration.py`:

```python
from argparse import Namespace


def test_migrate_infisical_dry_run_prints_destinations(tmp_path: Path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENAI_API_KEY=sk-model\n", encoding="utf-8")

    monkeypatch.setattr("hermes_cli.secrets_migration.get_hermes_home", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.secrets_migration.load_config",
        lambda: {"secrets": {"default_aliases": {"OPENAI_API_KEY": "models"}}},
    )

    from hermes_cli.secrets_migration import migrate_infisical

    result = migrate_infisical(Namespace(dry_run=True, yes=False))

    assert result == 0
    assert "OPENAI_API_KEY -> models:OPENAI_API_KEY" in capsys.readouterr().out


def test_migrate_infisical_yes_executes_migration(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENAI_API_KEY=sk-model\n", encoding="utf-8")
    backend = RecordingBackend()

    monkeypatch.setattr("hermes_cli.secrets_migration.get_hermes_home", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.secrets_migration.load_config",
        lambda: {"secrets": {"default_aliases": {"OPENAI_API_KEY": "models"}}},
    )
    monkeypatch.setattr("hermes_cli.secrets_migration.build_infisical_backend_from_config", lambda config: backend)

    from hermes_cli.secrets_migration import migrate_infisical

    result = migrate_infisical(Namespace(dry_run=False, yes=True))

    assert result == 0
    assert backend.values["models:OPENAI_API_KEY"] == "sk-model"
```

- [ ] **Step 2: Run command tests and confirm failure**

Run:

```bash
pytest tests/hermes_cli/test_secrets_migration.py::test_migrate_infisical_dry_run_prints_destinations tests/hermes_cli/test_secrets_migration.py::test_migrate_infisical_yes_executes_migration -q
```

Expected: failure because `migrate_infisical()` does not execute the backend migration.

- [ ] **Step 3: Implement migration command wiring**

Modify `hermes_cli/secrets_migration.py`:

```python
from hermes_constants import get_hermes_home
from hermes_cli.config import load_config


def build_infisical_backend_from_config(config: dict) -> SecretBackend:
    from agent.secrets.backends.infisical import InfisicalBackend
    from agent.secrets.bootstrap.keyring_store import KeyringBootstrapStore
    from agent.secrets.config import load_secrets_config

    secrets_config = load_secrets_config(config)
    if not secrets_config.infisical:
        raise RuntimeError("Infisical secret backend is not configured")
    return InfisicalBackend(
        config=secrets_config.infisical,
        bootstrap_store=KeyringBootstrapStore(),
    )


def migrate_infisical(args):
    config = load_config()
    aliases = config.get("secrets", {}).get("default_aliases", {})
    env_path = get_hermes_home() / ".env"
    plan = plan_env_migration(env_path, default_aliases=aliases)

    if args.dry_run:
        for key, alias in sorted(plan.entries.items()):
            print(f"{key} -> {alias}:{key}")
        return 0

    if not args.yes:
        raise SystemExit("Run `hermes secrets migrate infisical --dry-run`, then rerun with --yes")

    backend = build_infisical_backend_from_config(config)
    result = execute_migration(env_path, plan=plan, backend=backend)
    for key, alias in sorted(result.migrated.items()):
        print(f"migrated {key} -> {alias}:{key}")
    print(f"backup: {result.backup_path}")
    return 0
```

- [ ] **Step 4: Run migration tests**

Run:

```bash
pytest tests/hermes_cli/test_secrets_migration.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/secrets_migration.py tests/hermes_cli/test_secrets_migration.py
git commit -m "feat: wire infisical migration command"
```

## Task 10: Setup Flow Wiring

**Files:**
- Modify: `hermes_cli/secrets_setup.py`
- Modify: `hermes_cli/secrets_migration.py`
- Test: `tests/hermes_cli/test_secrets_setup.py`

- [ ] **Step 1: Add failing non-interactive setup test**

Append to `tests/hermes_cli/test_secrets_setup.py`:

```python
from argparse import Namespace

from agent.secrets.bootstrap.base import BootstrapCredentials
from agent.secrets.bootstrap.memory_store import MemoryBootstrapStore
from hermes_cli.secrets_setup import build_infisical_config, store_bootstrap_credentials


def test_build_infisical_config_local_tailscale():
    args = Namespace(
        host="http://127.0.0.1:8080",
        environment="prod",
        identity_name="hermes-agent-default",
        tailnet_url="https://hermes-infisical.tailnet-name.ts.net",
    )

    config = build_infisical_config(args, projects={"models": {"project_id": "project-models", "path": "/hermes/models"}})

    assert config["active_backend"] == "infisical"
    assert config["backends"]["infisical"]["host"] == "http://127.0.0.1:8080"
    assert config["backends"]["infisical"]["remote_access"]["mode"] == "tailscale"
    assert config["backends"]["infisical"]["remote_access"]["public_funnel_allowed"] is False
    assert config["backends"]["infisical"]["projects"]["models"]["project_id"] == "project-models"


def test_store_bootstrap_credentials():
    store = MemoryBootstrapStore()

    store_bootstrap_credentials(
        store,
        identity_name="hermes-agent-default",
        client_id="client-id",
        client_secret="client-secret",
    )

    assert store.get("hermes-agent-default") == BootstrapCredentials("client-id", "client-secret")
```

- [ ] **Step 2: Run setup tests and confirm failure**

Run:

```bash
pytest tests/hermes_cli/test_secrets_setup.py -q
```

Expected: import failure for `build_infisical_config` or `store_bootstrap_credentials`.

- [ ] **Step 3: Implement setup helpers**

Modify `hermes_cli/secrets_setup.py`:

```python
from agent.secrets.bootstrap.base import BootstrapCredentials, BootstrapStore


def store_bootstrap_credentials(
    store: BootstrapStore,
    *,
    identity_name: str,
    client_id: str,
    client_secret: str,
) -> None:
    store.set(identity_name, BootstrapCredentials(client_id=client_id, client_secret=client_secret))


def build_infisical_config(args, *, projects: dict[str, dict[str, str]]) -> dict:
    return {
        "active_backend": "infisical",
        "backends": {
            "infisical": {
                "type": "infisical",
                "host": args.host or "http://127.0.0.1:8080",
                "remote_access": {
                    "mode": "tailscale",
                    "tailnet_url": args.tailnet_url or "",
                    "public_funnel_allowed": False,
                },
                "environment": args.environment or "prod",
                "auth": {
                    "method": "universal_auth",
                    "bootstrap_store": "os_keyring",
                    "identity_name": args.identity_name or "hermes-agent-default",
                },
                "projects": projects,
            }
        },
    }
```

- [ ] **Step 4: Implement interactive setup shell**

Modify `setup_infisical(args)` in `hermes_cli/secrets_setup.py`:

```python
def setup_infisical(args):
    from hermes_cli.cli_output import print_success
    from hermes_cli.config import load_config, save_config

    host = args.host or "http://127.0.0.1:8080"
    validate_local_infisical_host(host)
    if args.tailnet_url:
        validate_tailscale_remote_access(args.tailnet_url)

    config = load_config()
    config["secrets"] = build_infisical_config(
        args,
        projects=config.get("secrets", {}).get("backends", {}).get("infisical", {}).get("projects", {}),
    )
    save_config(config)
    print_success("Infisical secret backend configured")
    return 0
```

This shell stores config and validates local/Tailscale posture. The next task expands it with credential prompts and Infisical project verification.

- [ ] **Step 5: Run setup tests**

Run:

```bash
pytest tests/hermes_cli/test_secrets_setup.py tests/hermes_cli/test_secrets_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add hermes_cli/secrets_setup.py tests/hermes_cli/test_secrets_setup.py
git commit -m "feat: wire infisical setup config"
```

## Task 11: Setup Wizard Prompts And Project Verification

**Files:**
- Modify: `hermes_cli/secrets_commands.py`
- Modify: `hermes_cli/secrets_setup.py`
- Test: `tests/hermes_cli/test_secrets_setup.py`
- Test: `tests/hermes_cli/test_secrets_commands.py`

- [ ] **Step 1: Add failing project parsing and verification tests**

Append to `tests/hermes_cli/test_secrets_setup.py`:

```python
from agent.secrets.references import SecretReference
from hermes_cli.secrets_setup import parse_project_specs, verify_infisical_projects


class RecordingBackend:
    def __init__(self):
        self.values = {}

    def get_secret(self, ref: SecretReference) -> str:
        return self.values[ref.qualified]

    def set_secret(self, ref: SecretReference, value: str) -> None:
        self.values[ref.qualified] = value

    def delete_secret(self, ref: SecretReference) -> None:
        self.values.pop(ref.qualified, None)


def test_parse_project_specs():
    projects = parse_project_specs(["models=project-models:/hermes/models", "mail=project-mail"])

    assert projects == {
        "models": {"project_id": "project-models", "path": "/hermes/models"},
        "mail": {"project_id": "project-mail", "path": "/"},
    }


def test_verify_infisical_projects_writes_reads_and_deletes():
    backend = RecordingBackend()

    verify_infisical_projects(backend, ["models"])

    assert "models:HERMES_VERIFY" not in backend.values
```

Append to `tests/hermes_cli/test_secrets_commands.py`:

```python
def test_setup_parser_accepts_bootstrap_and_projects():
    parser, subparsers = make_parser()
    register_secrets_subparser(subparsers)

    args = parser.parse_args([
        "secrets",
        "setup",
        "infisical",
        "--non-interactive",
        "--client-id",
        "client-id",
        "--client-secret",
        "client-secret",
        "--project",
        "models=project-models:/hermes/models",
    ])

    assert args.client_id == "client-id"
    assert args.client_secret == "client-secret"
    assert args.project == ["models=project-models:/hermes/models"]
```

- [ ] **Step 2: Run setup tests and confirm failure**

Run:

```bash
pytest tests/hermes_cli/test_secrets_setup.py::test_parse_project_specs tests/hermes_cli/test_secrets_setup.py::test_verify_infisical_projects_writes_reads_and_deletes tests/hermes_cli/test_secrets_commands.py::test_setup_parser_accepts_bootstrap_and_projects -q
```

Expected: import or attribute failures for project parsing and parser args.

- [ ] **Step 3: Add setup CLI arguments**

Modify the setup parser in `hermes_cli/secrets_commands.py`:

```python
    setup.add_argument("--client-id", default="")
    setup.add_argument("--client-secret", default="")
    setup.add_argument("--project", action="append", default=[], help="Project mapping alias=project_id or alias=project_id:/path")
    setup.add_argument("--skip-migration", action="store_true", default=False)
```

- [ ] **Step 4: Implement project parsing and verification**

Modify `hermes_cli/secrets_setup.py`:

```python
from agent.secrets.backends.base import SecretBackend
from agent.secrets.references import SecretReference


def parse_project_specs(specs: list[str]) -> dict[str, dict[str, str]]:
    projects: dict[str, dict[str, str]] = {}
    for spec in specs:
        alias, sep, rest = spec.partition("=")
        if not sep or not alias or not rest:
            raise ValueError(f"Invalid project mapping {spec!r}; expected alias=project_id or alias=project_id:/path")
        project_id, colon, path = rest.partition(":")
        projects[alias] = {"project_id": project_id, "path": path if colon else "/"}
    return projects


def verify_infisical_projects(backend: SecretBackend, aliases: list[str]) -> None:
    for alias in aliases:
        ref = SecretReference(alias=alias, key="HERMES_VERIFY")
        backend.set_secret(ref, "ok")
        if backend.get_secret(ref) != "ok":
            raise RuntimeError(f"Infisical verification failed for alias {alias!r}")
        backend.delete_secret(ref)
```

- [ ] **Step 5: Expand `setup_infisical()` to store bootstrap credentials and verify aliases**

Modify `setup_infisical(args)` in `hermes_cli/secrets_setup.py`:

```python
def setup_infisical(args):
    from hermes_cli.cli_output import print_success, prompt
    from hermes_cli.config import load_config, save_config
    from agent.secrets.backends.infisical import InfisicalBackend
    from agent.secrets.bootstrap.keyring_store import KeyringBootstrapStore
    from agent.secrets.config import load_secrets_config

    host = args.host or "http://127.0.0.1:8080"
    validate_local_infisical_host(host)
    if args.tailnet_url:
        validate_tailscale_remote_access(args.tailnet_url)

    client_id = args.client_id or prompt("Infisical Universal Auth client ID")
    client_secret = args.client_secret or prompt("Infisical Universal Auth client secret", password=True)
    if not client_id or not client_secret:
        raise SystemExit("Infisical Universal Auth client ID and client secret are required")

    projects = parse_project_specs(args.project)
    if not projects:
        raise SystemExit("At least one --project alias=project_id mapping is required")

    store = KeyringBootstrapStore()
    store_bootstrap_credentials(
        store,
        identity_name=args.identity_name,
        client_id=client_id,
        client_secret=client_secret,
    )

    config = load_config()
    config["secrets"] = build_infisical_config(args, projects=projects)
    typed = load_secrets_config(config)
    if not typed.infisical:
        raise SystemExit("Infisical config could not be loaded")
    verify_infisical_projects(InfisicalBackend(config=typed.infisical, bootstrap_store=store), list(projects))
    save_config(config)
    print_success("Infisical secret backend configured and verified")
    return 0
```

- [ ] **Step 6: Run setup command tests**

Run:

```bash
pytest tests/hermes_cli/test_secrets_setup.py tests/hermes_cli/test_secrets_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add hermes_cli/secrets_commands.py hermes_cli/secrets_setup.py tests/hermes_cli/test_secrets_setup.py tests/hermes_cli/test_secrets_commands.py
git commit -m "feat: verify infisical setup"
```

## Task 12: Runtime Resolver Initialization And Env Injection

**Files:**
- Modify: `agent/secrets/resolver.py`
- Modify: `hermes_cli/env_loader.py`
- Modify: `hermes_cli/config.py`
- Test: `tests/hermes_cli/test_env_loader.py`
- Test: `tests/agent/secrets/test_resolver.py`

- [ ] **Step 1: Add failing env-loader injection test**

Append to `tests/hermes_cli/test_env_loader.py`:

```python
def test_load_hermes_dotenv_injects_resolved_secrets(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")

    injected = {}

    def fake_inject_known_env_secrets():
        injected["called"] = True
        return {"OPENAI_API_KEY": "from-vault"}

    monkeypatch.setattr("agent.secrets.resolver.inject_known_env_secrets", fake_inject_known_env_secrets)

    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv(hermes_home=home)

    assert injected["called"] is True
```

- [ ] **Step 2: Run the specific test and confirm failure**

Run:

```bash
pytest tests/hermes_cli/test_env_loader.py::test_load_hermes_dotenv_injects_resolved_secrets -q
```

Expected: assertion failure because `load_hermes_dotenv()` does not call the resolver.

- [ ] **Step 3: Add resolver factory**

Modify `agent/secrets/resolver.py`:

```python
_ACTIVE_RESOLVER: SecretResolver | None = None


def set_active_resolver(resolver: SecretResolver | None) -> None:
    global _ACTIVE_RESOLVER
    _ACTIVE_RESOLVER = resolver


def get_active_resolver() -> SecretResolver | None:
    return _ACTIVE_RESOLVER


def build_resolver_from_config(config: dict, env_values: dict[str, str] | None = None) -> SecretResolver:
    from agent.secrets.backends.env import EnvSecretBackend
    from agent.secrets.config import load_secrets_config

    secrets_config = load_secrets_config(config)
    active_backend = None
    if secrets_config.active_backend == "infisical" and secrets_config.infisical:
        from agent.secrets.backends.infisical import InfisicalBackend
        from agent.secrets.bootstrap.keyring_store import KeyringBootstrapStore

        active_backend = InfisicalBackend(
            config=secrets_config.infisical,
            bootstrap_store=KeyringBootstrapStore(),
        )
    return SecretResolver(
        config=secrets_config,
        active_backend=active_backend,
        env_backend=EnvSecretBackend(env_values or {}),
    )


def inject_known_env_secrets() -> dict[str, str]:
    from hermes_cli.config import OPTIONAL_ENV_VARS, load_config, load_env

    config = load_config()
    resolver = build_resolver_from_config(config, load_env())
    set_active_resolver(resolver)
    return resolver.inject(OPTIONAL_ENV_VARS.keys())
```

- [ ] **Step 4: Call resolver from env loader**

Modify `hermes_cli/env_loader.py` at the end of `load_hermes_dotenv()` before `return loaded`:

```python
    try:
        from agent.secrets.resolver import inject_known_env_secrets

        inject_known_env_secrets()
    except Exception:
        pass
```

This is best-effort to avoid breaking early boot paths when setup is incomplete.

- [ ] **Step 5: Run env-loader and resolver tests**

Run:

```bash
pytest tests/hermes_cli/test_env_loader.py tests/agent/secrets/test_resolver.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/secrets/resolver.py hermes_cli/env_loader.py tests/hermes_cli/test_env_loader.py tests/agent/secrets/test_resolver.py
git commit -m "feat: inject resolved secrets at startup"
```

## Task 13: Resolver-Aware Secret Capture

**Files:**
- Modify: `hermes_cli/config.py`
- Modify: `hermes_cli/callbacks.py`
- Test: `tests/cli/test_cli_secret_capture.py`
- Test: `tests/hermes_cli/test_secrets_migration.py`

- [ ] **Step 1: Add failing secure save routing test**

Append to `tests/cli/test_cli_secret_capture.py`:

```python
def test_save_env_value_secure_routes_to_active_secret_backend(monkeypatch):
    written = {}

    class FakeResolver:
        def set_secret(self, name, value):
            written[name] = value

    monkeypatch.setattr("agent.secrets.resolver.get_active_resolver", lambda: FakeResolver())

    from hermes_cli.config import save_env_value_secure

    result = save_env_value_secure("OPENAI_API_KEY", "sk-secret")

    assert result["stored_as"] == "OPENAI_API_KEY"
    assert written["OPENAI_API_KEY"] == "sk-secret"
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
pytest tests/cli/test_cli_secret_capture.py::test_save_env_value_secure_routes_to_active_secret_backend -q
```

Expected: assertion failure because `save_env_value_secure()` writes only to `.env`.

- [ ] **Step 3: Add resolver write method**

Modify `agent/secrets/resolver.py`:

```python
def set_secret(self, name: str, value: str) -> None:
    ref = parse_secret_reference(name)
    if ref.alias:
        if not self.active_backend:
            raise SecretNotFound(f"Secret backend is not configured for {ref.qualified}")
        self.active_backend.set_secret(ref, value)
        return
    default_alias = self.config.default_alias_for(ref.key)
    if default_alias and self.active_backend:
        self.active_backend.set_secret(SecretReference(alias=default_alias, key=ref.key), value)
        return
    self.env_backend.set_secret(ref, value)
```

Place `set_secret()` as a method on `SecretResolver`, not at module scope.

- [ ] **Step 4: Route secure saves to active backend**

Modify `hermes_cli/config.py`:

```python
def save_env_value_secure(key: str, value: str) -> Dict[str, Any]:
    try:
        from agent.secrets.resolver import get_active_resolver

        resolver = get_active_resolver()
        if resolver is not None:
            resolver.set_secret(key, value)
            return {
                "success": True,
                "stored_as": key,
                "validated": False,
                "backend": "active_secret_backend",
            }
    except Exception:
        pass
    save_env_value(key, value)
    return {
        "success": True,
        "stored_as": key,
        "validated": False,
        "backend": "env",
    }
```

- [ ] **Step 5: Run secret capture tests**

Run:

```bash
pytest tests/cli/test_cli_secret_capture.py tests/agent/secrets/test_resolver.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/secrets/resolver.py hermes_cli/config.py tests/cli/test_cli_secret_capture.py tests/agent/secrets/test_resolver.py
git commit -m "feat: route secret capture through resolver"
```

## Task 14: Redaction Coverage For Infisical Auth Material

**Files:**
- Modify: `agent/redact.py`
- Modify: `tests/agent/test_redact.py`

- [ ] **Step 1: Add redaction tests**

Append to `tests/agent/test_redact.py`:

```python
class TestInfisicalRedaction:
    def test_infisical_access_token_json_redacted(self):
        text = '{"accessToken": "st.infisical.access.token.value.1234567890"}'
        result = redact_sensitive_text(text)
        assert "infisical.access.token.value" not in result

    def test_universal_auth_client_secret_redacted(self):
        text = "INFISICAL_CLIENT_SECRET=client-secret-value-1234567890"
        result = redact_sensitive_text(text)
        assert "client-secret-value" not in result
```

- [ ] **Step 2: Run redaction tests**

Run:

```bash
pytest tests/agent/test_redact.py::TestInfisicalRedaction -q
```

Expected: if the tests pass with existing generic JSON/env redactors, do not change `agent/redact.py`. If the first test fails because `accessToken` casing is missed, continue to Step 3.

- [ ] **Step 3: Extend JSON redaction key names only if needed**

Modify `_JSON_KEY_NAMES` in `agent/redact.py` to include camelCase token fields:

```python
_JSON_KEY_NAMES = r"(?:api_?[Kk]ey|apiKey|token|accessToken|refreshToken|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)"
```

- [ ] **Step 4: Run full redaction tests**

Run:

```bash
pytest tests/agent/test_redact.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/redact.py tests/agent/test_redact.py
git commit -m "test: cover infisical secret redaction"
```

If `agent/redact.py` did not change because the tests passed, commit only the test file with the same message.

## Task 15: Documentation

**Files:**
- Create: `website/docs/user-guide/features/infisical-secrets.md`
- Modify: `website/sidebars.ts`
- Test: `cd website && npm run build`

- [ ] **Step 1: Create docs page**

Create `website/docs/user-guide/features/infisical-secrets.md`:

````markdown
# Infisical Secrets

Hermes can resolve credentials from a self-hosted Infisical instance instead of keeping active secrets in `~/.hermes/.env`.

The recommended personal setup is Infisical self-hosted on the same device as Hermes. Remote access to the Infisical web UI should go through Tailscale Serve or another tailnet-only reverse proxy. Do not expose Infisical with Tailscale Funnel or any public internet route.

## Setup Shape

```yaml
secrets:
  active_backend: infisical
  backends:
    infisical:
      type: infisical
      host: http://127.0.0.1:8080
      remote_access:
        mode: tailscale
        tailnet_url: https://hermes-infisical.example-tailnet.ts.net
        public_funnel_allowed: false
      environment: prod
      auth:
        method: universal_auth
        bootstrap_store: os_keyring
        identity_name: hermes-agent-default
      projects:
        models:
          project_id: project-models
          path: /hermes/models
```

## Commands

Run setup:

```bash
hermes secrets setup infisical
```

Preview migration:

```bash
hermes secrets migrate infisical --dry-run
```

Migrate after review:

```bash
hermes secrets migrate infisical --yes
```

After a successful migration, Hermes writes selected secrets to Infisical, verifies reads, creates a timestamped `.env` backup, and removes active plaintext secret values from `~/.hermes/.env`.

## Security Notes

The operating system credential store holds only the Infisical Universal Auth client ID and client secret. Infisical remains the source of truth for Hermes application secrets.

Use Tailscale access controls to restrict which user devices can reach the Infisical service.
````

- [ ] **Step 2: Add docs sidebar entry**

Modify `website/sidebars.ts` inside the `Features` -> `Management` category and add:

```ts
"user-guide/features/infisical-secrets",
```

- [ ] **Step 3: Run docs validation**

Run:

```bash
cd website && npm run build
```

Expected: Docusaurus build succeeds. If dependencies are not installed, run:

```bash
cd website && npm install && npm run build
```

- [ ] **Step 4: Commit**

```bash
git add website/docs/user-guide/features/infisical-secrets.md website/sidebars.ts
git commit -m "docs: add infisical secrets guide"
```

## Task 16: End-To-End Verification

**Files:**
- Modify test files only if verification exposes missing coverage.

- [ ] **Step 1: Run targeted test suite**

Run:

```bash
pytest \
  tests/agent/secrets \
  tests/hermes_cli/test_secrets_config.py \
  tests/hermes_cli/test_secrets_setup.py \
  tests/hermes_cli/test_secrets_migration.py \
  tests/hermes_cli/test_env_loader.py \
  tests/cli/test_cli_secret_capture.py \
  tests/agent/test_redact.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run config and env regression tests**

Run:

```bash
pytest \
  tests/hermes_cli/test_config.py \
  tests/hermes_cli/test_config_env_refs.py \
  tests/hermes_cli/test_config_env_expansion.py \
  tests/hermes_cli/test_redact_config_bridge.py \
  tests/cli/test_cli_save_config_value.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run repository test wrapper for touched areas**

Run:

```bash
scripts/run_tests.sh tests/agent/secrets tests/hermes_cli/test_secrets_migration.py tests/hermes_cli/test_secrets_setup.py
```

Expected: wrapper exits with status 0.

- [ ] **Step 4: Manual local-host preflight**

Run:

```bash
hermes secrets setup infisical --non-interactive --host http://127.0.0.1:8080 --environment prod --identity-name hermes-agent-default
```

Expected: if no local Infisical service is running, command fails with a clear local reachability error and does not write plaintext bootstrap credentials to config.

- [ ] **Step 5: Manual Tailscale exposure guard**

Run:

```bash
hermes secrets setup infisical --non-interactive --host http://127.0.0.1:8080 --tailnet-url https://example.com
```

Expected: command refuses setup because the remote URL is not tailnet-only.

- [ ] **Step 6: Review diff for secret leaks**

Run:

```bash
git diff --check
rg -n "client-secret|sk-test-value|access-token|client_secret.*[A-Za-z0-9]{8,}" agent hermes_cli tests website
```

Expected: `git diff --check` exits 0. The `rg` command should only find synthetic test strings in tests or docs, not live credentials.

- [ ] **Step 7: Commit final fixes**

If verification required small fixes:

```bash
git add <changed-files>
git commit -m "fix: harden infisical credential foundation"
```

If no fixes were needed, do not create an empty commit.
