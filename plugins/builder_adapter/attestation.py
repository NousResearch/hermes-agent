"""Authoritative governance artifact and effective Hermes profile attestation."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .canonical import canonical_sha256
from .errors import AdapterError
from .gitops import GitVerifier, safe_relative_path


@dataclass(frozen=True)
class ArtifactBinding:
    path: str
    commit: str
    sha256: str


@dataclass(frozen=True)
class EffectiveProfile:
    profile: str
    provider: str
    model: str
    fallback_chain: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    confinement: dict
    configuration_sha256: str

    def evidence(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "profile": self.profile,
            "profile_configuration_sha256": self.configuration_sha256,
            "fallback_chain": list(self.fallback_chain),
            "fallback_used": False,
            "attested_by": "hermes.builder_dispatch.v1",
        }


class GovernanceAttestor:
    """Loads exact Git object bytes; no caller-provided policy object is trusted."""

    def __init__(
        self,
        repository: str | Path,
        *,
        policy: ArtifactBinding,
        interface: ArtifactBinding,
    ):
        self.repository = Path(repository)
        self.policy_binding = policy
        self.interface_binding = interface
        self._git = GitVerifier({})

    def _load(self, binding: ArtifactBinding, code: str) -> tuple[bytes, dict]:
        raw = self._git.artifact_bytes(
            self.repository, binding.commit, binding.path
        )
        if hashlib.sha256(raw).hexdigest() != binding.sha256:
            raise AdapterError(code, f"authoritative hash mismatch: {binding.path}")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError(code, f"invalid authoritative JSON: {binding.path}") from exc
        if not isinstance(value, dict):
            raise AdapterError(code, f"authoritative artifact is not an object: {binding.path}")
        return raw, value

    def load(self) -> tuple[dict, dict]:
        _, policy = self._load(self.policy_binding, "PROFILE_POLICY_MISMATCH")
        _, interface = self._load(
            self.interface_binding, "PROFILE_POLICY_MISMATCH"
        )
        registered = (
            interface.get("routing_policy", {})
            .get("policy_artifact", {})
        )
        if (
            registered.get("path") != self.policy_binding.path
            or registered.get("sha256") != self.policy_binding.sha256
        ):
            raise AdapterError(
                "PROFILE_POLICY_MISMATCH",
                "interface does not bind the registered policy",
            )
        return policy, interface


class GovernanceSnapshot:
    """One runtime-approved commit and its complete transitive hash registry."""

    CONTRACT_PATH = (
        "ai-engineering-orchestrator/contracts/active/"
        "FEAT-HERMES-BUILDER-DISPATCH-001.json"
    )
    REGISTERED_CONTRACT_PATH = (
        "contracts/active/FEAT-HERMES-BUILDER-DISPATCH-001.json"
    )
    PREFIX = "ai-engineering-orchestrator/"
    REQUIRED_BINDINGS = frozenset(
        {
            "allowed_path_manifest",
            "interface_contract",
            "builder_profile_policy",
            "validation_profile",
            "allowed_path_manifest_schema",
            "builder_profile_policy_schema",
            "completion_evidence_schema",
            "dispatch_request_schema",
            "dispatch_result_schema",
            "validation_tool_request_schema",
            "validation_profile_schema",
            "capability_registry",
            "capability_registry_schema",
            "adr_schema",
            "adr_2026_006_json",
            "adr_2026_006_markdown",
            "adr_2026_007_json",
            "adr_2026_007_markdown",
            "adr_2026_008_json",
            "adr_2026_008_markdown",
            "adr_2026_009_json",
            "adr_2026_009_markdown",
            "adr_2026_010_json",
            "adr_2026_010_markdown",
            "adr_2026_011_approval_provenance",
            "adr_2026_011_json",
            "adr_2026_011_markdown",
            "governance_registry_verifier",
            "constitution",
            "audit_verifier",
            "audit_verifier_tests",
            "audit_frozen_fixture",
            "authoritative_audit_log",
        }
    )

    def __init__(
        self,
        repository: str | Path,
        commit: str,
        *,
        registered_contract_path: str | None = None,
    ):
        if not re.fullmatch(r"[0-9a-f]{40}([0-9a-f]{24})?", commit):
            raise AdapterError("CONTRACT_MISMATCH", "approved governance commit is invalid")
        self.repository = Path(repository)
        self.commit = commit
        self.registered_contract_path = (
            safe_relative_path(registered_contract_path)
            if registered_contract_path is not None
            else self.REGISTERED_CONTRACT_PATH
        )
        self.contract_path = self.PREFIX + self.registered_contract_path
        self._git = GitVerifier({})
        contract_raw = self._git.artifact_bytes(
            self.repository, self.commit, self.contract_path
        )
        self.contract_raw = contract_raw
        self.contract_sha256 = hashlib.sha256(contract_raw).hexdigest()
        try:
            self.contract = json.loads(contract_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError("CONTRACT_MISMATCH", "invalid governance root JSON") from exc
        bindings = self.contract.get("artifact_bindings")
        if not isinstance(bindings, dict) or set(bindings) != self.REQUIRED_BINDINGS:
            raise AdapterError(
                "CONTRACT_MISMATCH", "governance transitive registry is incomplete"
            )
        self.bindings = {}
        for artifact_id, binding in bindings.items():
            if not isinstance(binding, dict) or "path" not in binding:
                raise AdapterError("CONTRACT_MISMATCH", "invalid artifact registration")
            relative = safe_relative_path(str(binding["path"]))
            path = self.PREFIX + relative
            raw = self._git.artifact_bytes(self.repository, self.commit, path)
            if artifact_id == "authoritative_audit_log":
                if set(binding) != {"path", "chain_head_event_id", "chain_head_hash"}:
                    raise AdapterError("CONTRACT_MISMATCH", "invalid audit registration")
                try:
                    head = json.loads(raw.splitlines()[-1])
                except (IndexError, json.JSONDecodeError) as exc:
                    raise AdapterError("CONTRACT_MISMATCH", "invalid audit log") from exc
                if (
                    head.get("event_id") != binding["chain_head_event_id"]
                    or head.get("entry_hash") != binding["chain_head_hash"]
                ):
                    raise AdapterError("CONTRACT_MISMATCH", "audit chain head mismatch")
                value = None
            else:
                if set(binding) != {"path", "sha256"}:
                    raise AdapterError("CONTRACT_MISMATCH", "invalid artifact registration")
                if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
                    raise AdapterError(
                        "CONTRACT_MISMATCH",
                        f"registered artifact hash mismatch: {artifact_id}",
                    )
                value = (
                    json.loads(raw)
                    if relative.endswith(".json")
                    else None
                )
            self.bindings[artifact_id] = (relative, raw, value)
        self._validate_derived_bindings()

    def _schema_validate(
        self, schema_id: str, artifact_id: str, label: str
    ) -> None:
        try:
            from jsonschema import Draft202012Validator, FormatChecker
        except ImportError as exc:
            raise AdapterError(
                "CONTRACT_MISMATCH", "registered JSON Schema validator unavailable"
            ) from exc
        schema = self.value(schema_id)
        artifact = self.value(artifact_id)
        if not isinstance(schema, dict) or not isinstance(artifact, dict):
            raise AdapterError(
                "CONTRACT_MISMATCH", f"registered {label} is unavailable"
            )
        try:
            Draft202012Validator.check_schema(schema)
            errors = sorted(
                Draft202012Validator(
                    schema, format_checker=FormatChecker()
                ).iter_errors(artifact),
                key=lambda error: list(error.path),
            )
        except Exception as exc:
            raise AdapterError(
                "CONTRACT_MISMATCH", f"registered {label} schema is invalid"
            ) from exc
        if errors:
            path = ".".join(str(item) for item in errors[0].path) or "$"
            raise AdapterError(
                "CONTRACT_MISMATCH",
                f"registered {label} fails schema validation at {path}",
            )

    def _validate_derived_bindings(self) -> None:
        self._schema_validate(
            "capability_registry_schema",
            "capability_registry",
            "capability registry",
        )
        self._schema_validate(
            "builder_profile_policy_schema",
            "builder_profile_policy",
            "builder profile policy",
        )
        self._schema_validate(
            "validation_profile_schema",
            "validation_profile",
            "validation profile",
        )

        capability_id = "hermes.builder_dispatch.v1"
        registry = self.value("capability_registry")
        interface = self.value("interface_contract")
        matches = [
            item
            for item in registry.get("capabilities", [])
            if item.get("capability_id") == capability_id
        ]
        if len(matches) != 1:
            raise AdapterError(
                "CONTRACT_MISMATCH",
                "capability registry must contain exactly one builder capability",
            )
        interface_binding = self.contract["artifact_bindings"][
            "interface_contract"
        ]
        expected_interface = {
            "path": interface_binding["path"],
            "version": interface.get("interface_contract_version"),
            "sha256": interface_binding["sha256"],
        }
        if matches[0].get("interface_contract") != expected_interface:
            raise AdapterError(
                "CONTRACT_MISMATCH",
                "capability registry interface binding mismatch",
            )
        if interface.get("capability_id") != capability_id:
            raise AdapterError(
                "CONTRACT_MISMATCH", "interface capability identity mismatch"
            )

        routing = interface.get("routing_policy", {})
        for key, artifact_id in (
            ("policy_artifact", "builder_profile_policy"),
            ("validation_profile_artifact", "validation_profile"),
        ):
            reference = routing.get(key, {})
            registered = self.contract["artifact_bindings"][artifact_id]
            if reference != registered:
                raise AdapterError(
                    "CONTRACT_MISMATCH", "transitive interface binding mismatch"
                )

    def _json_path(self, path: str, expected_hash: str | None) -> dict:
        raw = self._git.artifact_bytes(self.repository, self.commit, path)
        if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
            raise AdapterError("CONTRACT_MISMATCH", "governance root hash mismatch")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError("CONTRACT_MISMATCH", "invalid governance JSON") from exc
        if not isinstance(value, dict):
            raise AdapterError("CONTRACT_MISMATCH", "governance JSON must be an object")
        return value

    def raw(self, artifact_id: str) -> bytes:
        try:
            return self.bindings[artifact_id][1]
        except KeyError as exc:
            raise AdapterError("CONTRACT_MISMATCH", "artifact ID is not registered") from exc

    def value(self, artifact_id: str) -> dict:
        try:
            return self.bindings[artifact_id][2]
        except KeyError as exc:
            raise AdapterError("CONTRACT_MISMATCH", "artifact ID is not registered") from exc

    def load(self) -> tuple[dict, dict]:
        return self.value("builder_profile_policy"), self.value("interface_contract")


@contextmanager
def _profile_scope(profile_dir: Path) -> Iterator[None]:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(profile_dir))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


class HermesProfileResolver:
    """Resolves configuration through public Hermes profile/config interfaces."""

    def resolve(self, policy: dict) -> EffectiveProfile:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.profiles import get_profile_dir, profile_exists

        name = str(policy.get("profile", ""))
        if not name or not profile_exists(name):
            raise AdapterError(
                "PROFILE_POLICY_MISMATCH", "required Hermes profile is unavailable"
            )
        profile_dir = get_profile_dir(name)
        with _profile_scope(profile_dir):
            config = load_config_readonly()
            from hermes_cli.plugins import discover_plugins

            discover_plugins(force=True)

        model_cfg = config.get("model")
        if not isinstance(model_cfg, dict):
            raise AdapterError("PROFILE_POLICY_MISMATCH", "profile model config missing")
        configured_toolsets = config.get("platform_toolsets", {}).get("cli")
        enabled_plugins = config.get("plugins", {}).get("enabled", [])
        from model_tools import _compute_tool_definitions

        from .native import BUILDER_WORKER_POLICY

        worker_env = __import__("os").environ
        marker_names = (
            "HERMES_INTERNAL_WORKER_POLICY",
            "HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST",
            "HERMES_KANBAN_TASK",
        )
        previous_markers = {name: worker_env.get(name) for name in marker_names}
        worker_env["HERMES_INTERNAL_WORKER_POLICY"] = BUILDER_WORKER_POLICY[
            "policy_id"
        ]
        worker_env["HERMES_INTERNAL_WORKER_TOOL_ALLOWLIST"] = __import__(
            "json"
        ).dumps(
            BUILDER_WORKER_POLICY["tool_allowlist"],
            separators=(",", ":"),
        )
        # Registry check functions expose lifecycle tools only inside a Kanban
        # worker context. This sentinel resolves definitions but never invokes
        # a tool or creates/claims a task.
        worker_env["HERMES_KANBAN_TASK"] = "__builder_profile_attestation__"
        try:
            with _profile_scope(profile_dir):
                definitions = _compute_tool_definitions(
                    list(configured_toolsets or []), quiet_mode=True
                )
        finally:
            for marker_name, previous in previous_markers.items():
                if previous is None:
                    worker_env.pop(marker_name, None)
                else:
                    worker_env[marker_name] = previous
        actual_tools = sorted(item["function"]["name"] for item in definitions)
        observed = {
            "profile": name,
            "provider": model_cfg.get("provider"),
            "model": model_cfg.get("default") or model_cfg.get("model"),
            "fallback_chain": config.get("fallback_providers", []),
            "allowed_tools": actual_tools,
            "cli_toolsets": configured_toolsets,
            "plugin_enabled": "builder_adapter" in enabled_plugins,
            "confinement": config.get("builder_dispatch", {}).get("confinement"),
        }
        expected_tools = sorted(policy.get("allowed_tools", []))
        if (
            observed["provider"] != policy.get("provider")
            or observed["model"] != policy.get("model")
            or observed["fallback_chain"] != []
            or sorted(observed["allowed_tools"]) != expected_tools
            or sorted(observed["cli_toolsets"] or [])
            != ["builder_adapter", "no_mcp"]
            or observed["plugin_enabled"] is not True
            or not isinstance(observed["confinement"], dict)
            or observed["confinement"].get("kind") != "application_tool_mediated"
            or observed["confinement"].get("os_sandbox") is not False
            or observed["confinement"].get("terminal_tools") is not False
            or observed["confinement"].get("process_tools") is not False
        ):
            raise AdapterError(
                "PROFILE_POLICY_MISMATCH",
                "effective Hermes profile does not satisfy governed policy",
            )
        sanitized = {
            "profile": observed["profile"],
            "provider": observed["provider"],
            "model": observed["model"],
            "fallback_chain": observed["fallback_chain"],
            "allowed_tools": sorted(observed["allowed_tools"]),
            "cli_toolsets": sorted(observed["cli_toolsets"]),
            "plugin_enabled": observed["plugin_enabled"],
            "confinement": observed["confinement"],
        }
        return EffectiveProfile(
            profile=name,
            provider=observed["provider"],
            model=observed["model"],
            fallback_chain=tuple(observed["fallback_chain"]),
            allowed_tools=tuple(sorted(observed["allowed_tools"])),
            confinement=dict(observed["confinement"]),
            configuration_sha256=canonical_sha256(sanitized),
        )
