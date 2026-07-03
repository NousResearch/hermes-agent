"""Configuration loading. Secrets come from env vars, never from the YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LLMConfig:
    model: str = "deepseek-v4-flash"
    provider: str = "deepseek"
    base_url: str = "http://llm.internal:8000/v1"
    api_key_env: str = "ORCHARD_LLM_API_KEY"
    reasoning_effort: str = "none"

    @property
    def api_key(self) -> str:
        # Dummy-tolerant: open internal endpoints often need no real key.
        return os.environ.get(self.api_key_env, "") or "sk-internal-noauth"


@dataclass
class Paths:
    root: Path = Path("./data")
    runtime: Path = Path("./run")
    registry_db: Path = Path("./data/orchard.db")

    def employees_dir(self) -> Path:
        return self.root / "employees"

    def home_for(self, employee_id: str) -> Path:
        return self.employees_dir() / employee_id

    def socket_for(self, employee_id: str) -> Path:
        # Socket lives INSIDE the tenant's own home so the sandbox (which denies
        # every other home) also denies reaching a sibling worker's control
        # socket. The router runs outside the sandbox and can still connect.
        return self.home_for(employee_id) / "run" / "worker.sock"

    def links_db(self) -> Path:
        return self.runtime / "links.db"


@dataclass
class SupervisorConfig:
    max_active_workers: int = 200
    idle_ttl_seconds: int = 900
    warm_pool_size: int = 0
    wake_timeout_seconds: int = 60


@dataclass
class SecurityConfig:
    run_as_user: str | None = None
    home_mode: str = "0700"
    secret_mode: str = "0600"
    require_provisioned: bool = True
    # Auto-onboard: on the FIRST message from an unknown platform user, create an
    # isolated profile for them (keyed by their platform user_id) and route all
    # later messages there. Overrides require_provisioned for unknown senders.
    auto_provision: bool = False
    # "seatbelt" wraps local workers in a macOS sandbox that denies other
    # tenants' dirs (dev boundary). null = FS-perms only. Prod uses the
    # docker/microvm backend instead.
    sandbox: str | None = None

    @property
    def home_mode_int(self) -> int:
        return int(self.home_mode, 8)

    @property
    def secret_mode_int(self) -> int:
        return int(self.secret_mode, 8)


@dataclass
class MattermostConfig:
    url_env: str = "MATTERMOST_URL"
    token_env: str = "MATTERMOST_TOKEN"
    # Verification token Mattermost sends with each /secret slash-command call.
    slash_token_env: str = "MATTERMOST_SLASH_TOKEN"

    @property
    def url(self) -> str:
        return os.environ.get(self.url_env, "").rstrip("/")

    @property
    def token(self) -> str:
        return os.environ.get(self.token_env, "")

    @property
    def slash_token(self) -> str:
        return os.environ.get(self.slash_token_env, "")


@dataclass
class SecretsConfig:
    store: str = "local"                 # local | vault (future)
    key_env: str = "ORCHARD_SECRET_KEY"  # reserved: at-rest encryption key
    form_base_url: str = "http://127.0.0.1:8700"  # where the entry form is reachable
    link_ttl_seconds: int = 900          # one-time link lifetime


@dataclass
class SkillsConfig:
    shared_dir: str = ""                 # read-only base skills library (admin-curated)
    # Bundled Hermes skills to disable per tenant (written into their config.yaml
    # as skills.disabled) so they don't compete with our custom ones.
    disabled: list = field(default_factory=lambda: ["github-auth", "github-code-review"])


@dataclass
class IntegrationsConfig:
    catalog_file: str = ""               # path to integrations.yaml (the catalog); "" = none


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: Paths = field(default_factory=Paths)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    mattermost: MattermostConfig = field(default_factory=MattermostConfig)
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    hermes_bin: str = "hermes"
    backend: str = "local"

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Settings":
        data: dict = {}
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text()) or {}
        s = cls()
        if "llm" in data:
            s.llm = LLMConfig(**{**s.llm.__dict__, **data["llm"]})
        if "paths" in data:
            p = data["paths"]
            s.paths = Paths(
                root=Path(p.get("root", s.paths.root)),
                runtime=Path(p.get("runtime", s.paths.runtime)),
                registry_db=Path(p.get("registry_db", s.paths.registry_db)),
            )
        if "supervisor" in data:
            s.supervisor = SupervisorConfig(**{**s.supervisor.__dict__, **data["supervisor"]})
        if "security" in data:
            s.security = SecurityConfig(**{**s.security.__dict__, **data["security"]})
        if "mattermost" in data:
            s.mattermost = MattermostConfig(**{**s.mattermost.__dict__, **data["mattermost"]})
        if "secrets" in data:
            s.secrets = SecretsConfig(**{**s.secrets.__dict__, **data["secrets"]})
        if "skills" in data:
            s.skills = SkillsConfig(**{**s.skills.__dict__, **data["skills"]})
        if "integrations" in data:
            s.integrations = IntegrationsConfig(**{**s.integrations.__dict__, **data["integrations"]})
        s.hermes_bin = data.get("hermes_bin", s.hermes_bin)
        s.backend = data.get("backend", s.backend)
        # Paths must be absolute: workers run as subprocesses with their own cwd,
        # so a relative HERMES_HOME/socket would resolve against the wrong dir.
        s.paths.root = s.paths.root.resolve()
        s.paths.runtime = s.paths.runtime.resolve()
        s.paths.registry_db = s.paths.registry_db.resolve()
        return s
