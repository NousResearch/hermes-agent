"""AWS deployment inputs, derived from what the tenant declared.

The shape of the problem: an agent's reach into a customer's systems must be reviewable by
that customer's own security team, and must not drift from what the bundle says. Both follow
from deriving the IAM from the declaration rather than maintaining it beside the declaration.

What this module refuses is more interesting than what it renders. A grant that says
``s3:*`` on ``*`` is indistinguishable, at apply time, from a grant somebody thought about;
the moment to tell them apart is now, in a file a human is reading, not later in an incident
review. Each refusal therefore names the narrower thing to write instead — a refusal that
leaves an operator with nowhere to go gets worked around, and a worked-around control is
worse than none because it still reports success.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from nova._fields import Doc
from nova.errors import SpecError

#: The file `nova deploy render` writes. The `.auto.` segment is Terraform's own convention
#: for "loaded without being named on the command line", which matters here: an operator who
#: forgets `-var-file` should not silently apply a deployment with no integrations at all.
TFVARS_FILENAME = "nova.auto.tfvars.json"

#: Terraform root module this renders input for. Checked in beside the platform layer.
MODULE_PATH = "deploy/aws"

#: ``service:Action``. Deliberately strict — a wildcard is refused separately and with a
#: better message, so anything left over here is a typo rather than a policy decision.
ACTION = re.compile(r"^[a-z0-9-]+:[A-Za-z0-9]+$")

#: An integration id: a role-name segment, a directory name and a log field.
INTEGRATION_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,48}[a-z0-9]$")

#: Services that grant the ability to grant. An integration holding any of these is not a
#: scoped integration: it is a route to an unscoped one, and the scoping above it is theatre.
ESCALATION_SERVICES = frozenset({"iam", "sts", "organizations", "account", "kms"})

#: Actions that read or write the account's own shape rather than a customer system. An agent
#: needs none of them, and an agent that has them can see every other tenant's resources in a
#: shared account.
ACCOUNT_WIDE_ACTIONS = frozenset(
    {
        "ec2:describeinstances",
        "ec2:describesecuritygroups",
        "s3:listallmybuckets",
        "rds:describedbinstances",
        "lambda:listfunctions",
        "secretsmanager:listsecrets",
        "ssm:describeparameters",
        "cloudtrail:lookupevents",
    }
)


@dataclass(frozen=True)
class Statement:
    """One least-privilege grant: literal actions on literal resources."""

    actions: tuple[str, ...]
    resources: tuple[str, ...]
    #: IAM condition block, carried opaquely — ``{"StringEquals": {"s3:prefix": ["acme/"]}}``.
    #: NOVA does not model conditions: they are the customer's own narrowing and inventing a
    #: schema for them would only limit what they can express.
    condition: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)

    def to_tfvars(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "actions": list(self.actions),
            "resources": list(self.resources),
        }
        if self.condition:
            out["condition"] = {
                test: {key: list(values) for key, values in variables.items()}
                for test, variables in self.condition.items()
            }
        return out


@dataclass(frozen=True)
class IntegrationSpec:
    """One customer system, reachable through one separately-scoped role."""

    id: str
    statements: tuple[Statement, ...]
    description: str = ""
    #: Agents this integration exists for. Recorded rather than enforced in IAM: the runtime
    #: assumes the role as one identity, so a per-agent IAM boundary would be a claim the
    #: infrastructure cannot keep. It is here because an operator reading the bundle should
    #: be able to see why an integration exists, and because `nova deploy render` refuses an
    #: integration naming an agent the bundle does not contain — which catches the rename
    #: that would otherwise leave a grant behind with nobody to use it.
    agents: tuple[str, ...] = ()

    def to_tfvars(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "statements": [statement.to_tfvars() for statement in self.statements],
        }


@dataclass(frozen=True)
class InfrastructureSpec:
    """Where this tenant runs. Every field an input to the Terraform module."""

    region: str = ""
    vpc_id: str = ""
    subnet_id: str = ""
    instance_type: str = ""
    image_uri: str = ""
    ami_id: str = ""
    volume_gb: Optional[int] = None
    kms_key_arn: str = ""
    log_retention_days: Optional[int] = None
    bedrock_model_ids: tuple[str, ...] = ()
    bedrock_inference_profile_arns: tuple[str, ...] = ()
    secret_prefix: str = ""

    def to_tfvars(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (
            ("region", self.region),
            ("vpc_id", self.vpc_id),
            ("subnet_id", self.subnet_id),
            ("instance_type", self.instance_type),
            ("image_uri", self.image_uri),
            ("ami_id", self.ami_id),
            ("kms_key_arn", self.kms_key_arn),
            ("secret_prefix", self.secret_prefix),
        ):
            if value:
                out[key] = value
        for key, number in (
            ("volume_gb", self.volume_gb),
            ("log_retention_days", self.log_retention_days),
        ):
            if number is not None:
                out[key] = number
        for key, values in (
            ("bedrock_model_ids", self.bedrock_model_ids),
            ("bedrock_inference_profile_arns", self.bedrock_inference_profile_arns),
        ):
            if values:
                out[key] = list(values)
        return out


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def check_action(action: str, *, field_path: str, source: Optional[Path]) -> str:
    """Refuse an action that grants more than someone reviewed.

    Three shapes, three different mistakes. A wildcard is usually haste; an ``iam:`` grant is
    usually someone solving a permissions error by removing the permission system; an
    account-wide describe is usually a misunderstanding of what the agent needs. All three
    read as reasonable in a pull request, which is why they are refused mechanically.
    """
    if "*" in action:
        raise SpecError(
            f"{action!r} contains a wildcard. Name the actions this integration needs — "
            f"'s3:GetObject', not 's3:*'. The list is what the customer's security team "
            f"reads to answer 'what can the agents do?', and a wildcard makes that question "
            f"unanswerable",
            field=field_path,
            source=source,
        )
    if not ACTION.match(action):
        raise SpecError(
            f"{action!r} is not an IAM action. Write service:Action, e.g. 's3:GetObject'",
            field=field_path,
            source=source,
        )
    service = action.split(":", 1)[0].lower()
    if service in ESCALATION_SERVICES:
        raise SpecError(
            f"{action!r} is a {service} action, which grants the ability to grant. An "
            f"integration holding it is not scoped — it is a route to an unscoped role, and "
            f"the scoping around it stops meaning anything. If the customer's own IAM "
            f"pipeline genuinely needs this, attach it to the role outside NOVA so it is "
            f"reviewed as the exception it is",
            field=field_path,
            source=source,
        )
    if action.lower() in ACCOUNT_WIDE_ACTIONS:
        raise SpecError(
            f"{action!r} enumerates the whole account rather than reaching one system. An "
            f"agent that can list every bucket can see every other tenant's. Name the "
            f"resources this integration needs instead",
            field=field_path,
            source=source,
        )
    return action


def check_resource(resource: str, *, field_path: str, source: Optional[Path]) -> str:
    """Refuse ``*``, and refuse a wildcard that spans a whole service or account."""
    if resource == "*":
        raise SpecError(
            "'*' grants these actions on every resource in the account. Name the buckets, "
            "tables or queues this integration reaches — 'arn:aws:s3:::acme-crm-export/*' "
            "narrows to one bucket and still allows every object in it",
            field=field_path,
            source=source,
        )
    if not resource.startswith("arn:"):
        raise SpecError(
            f"{resource!r} is not an ARN. Resources must be named as ARNs so the grant can "
            f"be read without knowing which service it belongs to",
            field=field_path,
            source=source,
        )
    # arn:partition:service:region:account:rest — a wildcard before `rest` spans every
    # resource of that service, which is '*' with extra steps.
    parts = resource.split(":", 5)
    if len(parts) < 6:
        raise SpecError(
            f"{resource!r} is not a complete ARN (arn:partition:service:region:account:resource)",
            field=field_path,
            source=source,
        )
    if "*" in parts[2] or parts[5] in ("", "*"):
        raise SpecError(
            f"{resource!r} wildcards the service or every resource in it, which is '*' "
            f"written longer. Name the resource",
            field=field_path,
            source=source,
        )
    return resource


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_statement(doc: Doc, *, index: int, source: Optional[Path]) -> Statement:
    prefix = f"integrations[{index}]"
    actions = doc.str_list("actions", unique=True)
    if not actions:
        raise SpecError("needs at least one action", field=f"{prefix}.actions", source=source)
    resources = doc.str_list("resources", unique=True)
    if not resources:
        raise SpecError(
            "needs at least one resource. An integration with actions and no resources "
            "cannot be rendered, because the only thing it could mean is '*'",
            field=f"{prefix}.resources",
            source=source,
        )
    condition = doc._raw("condition") or {}
    if condition and not isinstance(condition, Mapping):
        raise SpecError("must be a mapping", field=f"{prefix}.condition", source=source)
    doc.reject_unknown()
    return Statement(
        actions=tuple(
            check_action(a, field_path=f"{prefix}.actions", source=source) for a in actions
        ),
        resources=tuple(
            check_resource(r, field_path=f"{prefix}.resources", source=source) for r in resources
        ),
        condition={
            test: {key: tuple(values) for key, values in variables.items()}
            for test, variables in condition.items()
        },
    )


def parse_integrations(
    data: Any, *, source: Optional[Path] = None, env: Optional[Mapping[str, str]] = None
) -> tuple[IntegrationSpec, ...]:
    if data is None:
        return ()
    if not isinstance(data, (list, tuple)):
        raise SpecError("must be a list", field="integrations", source=source)

    out: list[IntegrationSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(data):
        doc = Doc(entry or {}, source=source, prefix=f"integrations[{index}]", env=env)
        integration_id = doc.str_("id")
        if not INTEGRATION_ID.match(integration_id or ""):
            raise SpecError(
                f"{integration_id!r} is not an integration id: lowercase letters, digits, "
                f"hyphens and underscores, 3-50 characters. It becomes an IAM role name",
                field=f"integrations[{index}].id",
                source=source,
            )
        if integration_id in seen:
            raise SpecError(
                f"{integration_id!r} is declared twice. Two integrations with one name would "
                f"render one role, and the second set of grants would vanish silently",
                field=f"integrations[{index}].id",
                source=source,
            )
        seen.add(integration_id)

        raw_statements = doc._raw("allow")
        if raw_statements is None:
            raise SpecError(
                "needs an 'allow' block listing what it may do",
                field=f"integrations[{index}].allow",
                source=source,
            )
        if not isinstance(raw_statements, (list, tuple)):
            raise SpecError(
                "must be a list of {actions, resources} entries",
                field=f"integrations[{index}].allow",
                source=source,
            )
        statements = tuple(
            _parse_statement(
                Doc(item or {}, source=source, prefix=f"integrations[{index}].allow[{n}]", env=env),
                index=index,
                source=source,
            )
            for n, item in enumerate(raw_statements)
        )
        if not statements:
            raise SpecError(
                "has an empty 'allow' block. Remove the integration, or say what it grants",
                field=f"integrations[{index}].allow",
                source=source,
            )

        spec = IntegrationSpec(
            id=integration_id,
            description=doc.str_("description"),
            agents=tuple(doc.str_list("agents", unique=True)),
            statements=statements,
        )
        doc.reject_unknown()
        out.append(spec)
    return tuple(out)


def parse_infrastructure(
    doc: Optional[Doc],
) -> InfrastructureSpec:
    if doc is None:
        return InfrastructureSpec()
    spec = InfrastructureSpec(
        region=doc.str_("region"),
        vpc_id=doc.str_("vpc_id"),
        subnet_id=doc.str_("subnet_id"),
        instance_type=doc.str_("instance_type"),
        image_uri=doc.str_("image_uri"),
        ami_id=doc.str_("ami_id"),
        volume_gb=doc.int_("volume_gb", minimum=20),
        kms_key_arn=doc.str_("kms_key_arn"),
        log_retention_days=doc.int_("log_retention_days", minimum=1),
        bedrock_model_ids=tuple(doc.str_list("bedrock_model_ids", unique=True)),
        bedrock_inference_profile_arns=tuple(
            doc.str_list("bedrock_inference_profile_arns", unique=True)
        ),
        secret_prefix=doc.str_("secret_prefix"),
    )
    doc.reject_unknown()
    for value, name in (
        (spec.secret_prefix, "secret_prefix"),
        (spec.image_uri, "image_uri"),
    ):
        if "*" in value:
            raise SpecError(
                f"must be literal, not a wildcard pattern",
                field=f"infrastructure.{name}",
                source=doc.source,
            )
    for model_id in spec.bedrock_model_ids:
        if "*" in model_id:
            raise SpecError(
                f"{model_id!r} contains a wildcard, which grants every model in the region. "
                f"List the model ids this deployment uses",
                field="infrastructure.bedrock_model_ids",
                source=doc.source,
            )
    return spec


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_tfvars(
    *,
    tenant_id: str,
    infrastructure: InfrastructureSpec,
    integrations: Sequence[IntegrationSpec],
    known_agents: Sequence[str] = (),
    tags: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Terraform input for one tenant.

    ``known_agents`` is checked rather than written: an integration naming an agent the
    bundle does not have is almost always a rename that left the grant behind, and a grant
    with nobody to use it is the kind of thing that survives three audits and then gets used.
    """
    if known_agents:
        known = set(known_agents)
        for integration in integrations:
            unknown = sorted(set(integration.agents) - known)
            if unknown:
                raise SpecError(
                    f"names agent(s) this bundle does not contain: {', '.join(unknown)}. "
                    f"Either the agent was renamed and this grant was left behind, or the "
                    f"integration is for an agent that does not exist yet — both are worth "
                    f"fixing before they reach IAM",
                    field=f"integrations[{integration.id}].agents",
                )

    out: dict[str, Any] = {"tenant_id": tenant_id}
    out.update(infrastructure.to_tfvars())
    out["integrations"] = [integration.to_tfvars() for integration in integrations]
    if tags:
        out["tags"] = dict(tags)
    return out


def write_tfvars(destination: Path, payload: Mapping[str, Any]) -> Path:
    """Write the tfvars file. Returns the path, so a caller can print it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
