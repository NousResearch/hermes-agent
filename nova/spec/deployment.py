"""Deployment settings — the part of a configuration the *operator* owns.

A tenant bundle says what the agents are. This says where they run: which model endpoint,
which credential, which region. The split matters because the two have different authors,
different review cycles and different secrecy. An agent spec is reviewed like code and lives
in version control; an endpoint changes when infrastructure changes, and the credential
behind it must never be in a file anyone reviews at all.

Three rules hold here, and the first is the one everything else follows from.

**NOVA never writes a secret.** Not to a profile, not to a log, not to an audit record.
A credential is named, never carried: ``api_key_env: ACME_LLM_KEY`` says *which variable*
holds the key, and the value lives in ``<profile>/.env``, which NOVA is forbidden from
writing (``materialize.NEVER_WRITE``) and the runtime already reads. A value that looks like
a credential is refused at parse time rather than trusted to an author who may have pasted it
by accident — by then it is in version control and the refusal is worthless.

**The vocabulary is NOVA's.** ``custom_providers``, ``key_env`` and ``base_url`` are Hermes
words and do not appear here. This module says endpoint, credential, context window; the
adapter translates. That is the same rule that keeps ``profile`` and ``kanban`` out of the
runtime contract, applied to configuration.

**Tenant defaults, per-agent overrides.** An endpoint is usually deployment-wide and a model
choice is often per-agent, so both are expressible and the narrower one wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from nova._fields import Doc
from nova.deploy.aws import (
    InfrastructureSpec,
    IntegrationSpec,
    parse_infrastructure,
    parse_integrations,
)
from nova.errors import SpecError

#: The file a tenant bundle declares its deployment settings in.
DEPLOYMENT_FILE = "deployment.yaml"

#: A POSIX environment variable name. Deliberately strict: an author who writes a *value*
#: here almost never produces something that matches, so the refusal below catches the
#: mistake that matters rather than only the ones that are obvious.
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Substrings that mark a config key as carrying a credential rather than naming one.
#: Checked against keys in the passthrough block, where NOVA cannot otherwise tell.
SECRET_KEY_MARKERS = ("key", "secret", "token", "password", "credential", "passwd")

#: Keys NOVA compiles itself. A passthrough that set these could silently undo a governance
#: control — an ``approvals`` block that re-permits a denied tool, or a ``plugins`` block that
#: disables the enforcement plugin — while every NOVA-side check still reported it enforced.
#: Refused rather than merged, because the failure would be invisible in exactly the review
#: that was supposed to catch it.
RESERVED_CONFIG_KEYS = frozenset(
    {"plugins", "approvals", "agent", "delegation", "kanban", "nova", "model", "custom_providers"}
)


#: Prefixes that are credentials by construction. An AWS access key id is the case that
#: makes the shape check alone insufficient: ``AKIAIOSFODNN7EXAMPLE`` is a perfectly valid
#: environment variable name and is unmistakably a secret.
CREDENTIAL_PREFIXES = (
    "AKIA", "ASIA", "ABIA", "ACCA",           # AWS access key ids
    "sk-", "sk_", "pk_", "rk_",               # OpenAI / Stripe style
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_",   # GitHub
    "xox", "AIza", "ya29.", "glpat-", "dop_", "shpat_",
)

#: An env var name this long with no underscore is vanishingly rare; a credential of that
#: shape is common. Chosen to sit above real names like ``ANTHROPICAPIKEY`` (16) while
#: catching the 20+ character opaque tokens that matter.
OPAQUE_NAME_LENGTH = 20


def looks_like_a_credential(value: str) -> str:
    """Why this value reads as a secret rather than a variable name, or ``""``.

    No such check can be complete, and this one does not try. It catches the two shapes that
    actually reach a config file by accident — a recognised credential prefix, and a long
    opaque run with no underscore — because the realistic failure is someone pasting a key
    into the wrong field, not someone constructing an adversarial one.
    """
    for prefix in CREDENTIAL_PREFIXES:
        if value.startswith(prefix):
            return f"it starts with {prefix!r}, which is a credential prefix"
    if len(value) >= OPAQUE_NAME_LENGTH and "_" not in value and any(c.isdigit() for c in value):
        return (
            f"it is {len(value)} characters with no underscore, which reads as an opaque "
            "token rather than a variable name"
        )
    return ""


def check_not_a_secret(value: str, *, field: str, source: Optional[Path]) -> str:
    """Refuse a literal credential where a variable name belongs.

    The check is cheap and the failure it prevents is not: a key pasted into
    ``deployment.yaml`` is in version control, in every clone, and in the history after it is
    removed. Refusing at parse time is the only moment where saying no still helps.
    """
    if not value:
        return value
    reason = looks_like_a_credential(value)
    if reason:
        raise SpecError(
            f"looks like a credential — {reason}. This field names the variable that holds "
            "the credential; it must never contain the credential itself. Put the value in "
            "<profile>/.env, which NOVA never writes, and name it here",
            field=field,
            source=source,
        )
    if not ENV_NAME.match(value):
        raise SpecError(
            f"{value[:12]}… is not an environment variable name. This field names the "
            "variable that holds the credential; it must never contain the credential "
            "itself. Put the value in <profile>/.env, which NOVA never writes, and name it "
            "here",
            field=field,
            source=source,
        )
    return value


@dataclass(frozen=True)
class ProviderSpec:
    """How to reach a model provider. No credential, by construction."""

    #: The runtime's name for the backend — ``bedrock``, ``anthropic``, a self-hosted id.
    #: Runtime-specific by nature: NOVA carries it, the adapter interprets it.
    provider: str = ""
    #: Default model when an agent does not name one.
    model: str = ""
    #: API endpoint. May be a literal URL or a ``${VAR}`` reference resolved at runtime.
    endpoint: str = ""
    #: The NAME of the environment variable holding the credential. Never the credential.
    api_key_env: str = ""
    #: Declared context window. The runtime refuses a model below its own floor, and a
    #: server that under-reports its window makes an agent fail for a reason that reads as
    #: unrelated, so it is worth being able to state.
    context_window: Optional[int] = None
    #: Region, for providers that are regional. Carried opaquely.
    region: str = ""

    @property
    def declared(self) -> bool:
        return bool(self.provider or self.model or self.endpoint or self.api_key_env)

    @property
    def required_env(self) -> tuple[str, ...]:
        """Every environment variable this provider needs before an agent can run.

        The credential variable, plus anything the endpoint or region defers with ``${VAR}``.
        Used by the readiness report: NOVA cannot supply these and must not pretend the
        deployment is complete without them.
        """
        names: list[str] = []
        if self.api_key_env:
            names.append(self.api_key_env)
        for value in (self.endpoint, self.region):
            names.extend(env_references(value, required_only=True))
        return tuple(dict.fromkeys(names))

    def merged_with(self, override: "ProviderSpec") -> "ProviderSpec":
        """``override`` wins field by field. Absent means "inherit", not "clear".

        Field-wise rather than wholesale: an agent that names only a different model must
        keep the tenant's endpoint, or every agent would have to restate the whole block to
        change one value — and the restating is where they drift.
        """
        return ProviderSpec(
            provider=override.provider or self.provider,
            model=override.model or self.model,
            endpoint=override.endpoint or self.endpoint,
            api_key_env=override.api_key_env or self.api_key_env,
            context_window=(
                override.context_window if override.context_window is not None
                else self.context_window
            ),
            region=override.region or self.region,
        )

    @classmethod
    def parse(cls, doc: Optional[Doc], *, prefix: str = "") -> "ProviderSpec":
        if doc is None:
            return cls()
        source = doc.source
        api_key_env = doc.str_("api_key_env", expand=False)
        spec = cls(
            provider=doc.str_("provider"),
            model=doc.str_("model"),
            # Not expanded at load: a ${VAR} endpoint must reach the profile config as a
            # reference, so it resolves in the worker's environment rather than in the
            # operator's shell at apply time. Those are different machines.
            endpoint=doc.str_("endpoint", expand=False),
            api_key_env=check_not_a_secret(
                api_key_env, field=f"{prefix}api_key_env" if prefix else "api_key_env",
                source=source,
            ),
            context_window=doc.int_("context_window", minimum=1),
            region=doc.str_("region", expand=False),
        )
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (
            ("provider", self.provider), ("model", self.model), ("endpoint", self.endpoint),
            ("api_key_env", self.api_key_env), ("region", self.region),
        ):
            if value:
                out[key] = value
        if self.context_window is not None:
            out["context_window"] = self.context_window
        return out


@dataclass(frozen=True)
class DeploymentSpec:
    """One tenant's deployment settings."""

    provider: ProviderSpec = ProviderSpec()
    #: Runtime-specific settings NOVA does not model, written verbatim into the agent's
    #: runtime configuration. Guarded: see :data:`RESERVED_CONFIG_KEYS`.
    runtime_config: Mapping[str, Any] = None  # type: ignore[assignment]
    #: Where this tenant runs, when NOVA renders the infrastructure. Absent is valid and
    #: common: a runtime already installed somewhere needs none of this.
    infrastructure: InfrastructureSpec = field(default_factory=InfrastructureSpec)
    #: Customer systems the agents may reach, each rendered as a separately-scoped IAM role.
    #: Empty means the agents reach nothing outside the runtime, which is the default and
    #: the safe one.
    integrations: tuple[IntegrationSpec, ...] = ()
    source: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.runtime_config is None:
            object.__setattr__(self, "runtime_config", {})

    @property
    def required_env(self) -> tuple[str, ...]:
        names = list(self.provider.required_env)
        names.extend(env_references_in(self.runtime_config))
        return tuple(dict.fromkeys(names))

    @classmethod
    def parse(
        cls,
        data: Any,
        *,
        source: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "DeploymentSpec":
        doc = Doc(data or {}, source=source, env=env)
        provider = ProviderSpec.parse(doc.child("provider"), prefix="provider.")
        runtime_config = _parse_runtime_config(doc, source=source)
        infrastructure = parse_infrastructure(doc.child("infrastructure"))
        integrations = parse_integrations(doc._raw("integrations"), source=source, env=env)
        doc.reject_unknown()
        return cls(
            provider=provider,
            runtime_config=runtime_config,
            infrastructure=infrastructure,
            integrations=integrations,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.provider.declared:
            out["provider"] = self.provider.to_dict()
        if self.runtime_config:
            out["runtime_config"] = dict(self.runtime_config)
        infrastructure = self.infrastructure.to_tfvars()
        if infrastructure:
            out["infrastructure"] = infrastructure
        if self.integrations:
            # In the digest, so that changing what an agent may reach changes the bundle
            # identity — a grant that could be widened without the provenance moving would
            # be a grant nobody could prove the age of.
            out["integrations"] = [i.to_tfvars() for i in self.integrations]
        return out


def _parse_runtime_config(doc: Doc, *, source: Optional[Path]) -> dict[str, Any]:
    """The passthrough block, with the two things it must never carry refused.

    An escape hatch is necessary — a customer with a private CA or a provider-specific body
    parameter cannot wait for NOVA to model it — but an unguarded one would quietly undo the
    governance NOVA spends the rest of its existence enforcing. So two refusals, both at
    parse time, both naming the alternative.
    """
    raw = doc._raw("runtime_config")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SpecError("must be a mapping", field="runtime_config", source=source)

    reserved = sorted(set(raw) & RESERVED_CONFIG_KEYS)
    if reserved:
        raise SpecError(
            f"may not set {', '.join(reserved)} — NOVA compiles those from the agent spec "
            "and the tenant policy. Setting them here would override a governance control "
            "while every NOVA-side check still reported it enforced",
            field="runtime_config",
            source=source,
        )

    for key, value in _walk(raw):
        lowered = key.lower()
        if any(marker in lowered for marker in SECRET_KEY_MARKERS) and isinstance(value, str):
            if value and not env_references(value):
                raise SpecError(
                    f"{key!r} looks like a credential and has a literal value. Reference an "
                    "environment variable instead (${VAR_NAME}) and put the value in "
                    "<profile>/.env, which NOVA never writes",
                    field=f"runtime_config.{key}",
                    source=source,
                )
    return dict(raw)


def _walk(mapping: Mapping[str, Any], prefix: str = ""):
    """Every (dotted key, value) pair in a nested mapping."""
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _walk(value, f"{path}.")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    yield from _walk(item, f"{path}[{index}].")
        else:
            yield path, value


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-[^}]*)?\}")


def env_references(value: Any, *, required_only: bool = False) -> tuple[str, ...]:
    """Environment variable names a ``${VAR}`` string defers to.

    ``required_only`` drops references that supply a default (``${VAR:-eu-west-1}``). A
    defaulted reference resolves with or without the variable, so reporting it as missing
    would send an operator hunting for something that was never needed — and a readiness
    report that cries wolf is one nobody reads the third time.
    """
    if not isinstance(value, str):
        return ()
    found = [
        name for name, default in _ENV_REF.findall(value)
        if not (required_only and default)
    ]
    return tuple(dict.fromkeys(found))


def env_references_in(
    mapping: Optional[Mapping[str, Any]], *, required_only: bool = True
) -> tuple[str, ...]:
    names: list[str] = []
    for _, value in _walk(mapping or {}):
        names.extend(env_references(value, required_only=required_only))
    return tuple(dict.fromkeys(names))


def load_deployment(
    bundle_root: Path,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> DeploymentSpec:
    """Read ``deployment.yaml`` from a tenant bundle. Absent means no deployment settings.

    Absent is valid: a runtime already configured by its operator needs nothing from NOVA
    here, and that is how every deployment before this file existed worked.
    """
    import yaml

    path = bundle_root / DEPLOYMENT_FILE
    if not path.is_file():
        return DeploymentSpec()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"could not be read: {exc.strerror or exc}", source=path) from exc
    except yaml.YAMLError as exc:
        raise SpecError(f"is not valid YAML: {exc}", source=path) from exc
    return DeploymentSpec.parse(data or {}, source=path, env=env)
