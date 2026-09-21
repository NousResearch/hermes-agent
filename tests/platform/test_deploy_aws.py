"""The deployment seam: what a tenant declares becomes IAM, and what it must never become.

The refusals carry the weight here. ``ARCHITECTURE_BOUNDARIES.md`` §6 promises that agents
never receive broad access to a customer's AWS account; these tests are what makes that a
control rather than a sentence. Each one names the shape of grant that would break the
promise, because the realistic failure is not malice — it is somebody resolving an
AccessDenied at five o'clock by widening the thing that denied them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from nova.deploy.aws import (
    IntegrationSpec,
    InfrastructureSpec,
    Statement,
    check_action,
    check_resource,
    parse_infrastructure,
    parse_integrations,
    render_tfvars,
    write_tfvars,
)
from nova.errors import SpecError
from nova.spec.deployment import DeploymentSpec

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "deploy" / "aws"


def integration(**overrides):
    data = {
        "id": "crm-export",
        "description": "Read-only CRM export bucket",
        "allow": [{"actions": ["s3:GetObject"], "resources": ["arn:aws:s3:::acme-crm/*"]}],
    }
    data.update(overrides)
    return [data]


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, expected",
    [
        ("s3:*", "wildcard"),
        ("*", "wildcard"),
        ("iam:PutRolePolicy", "ability to grant"),
        ("sts:AssumeRole", "ability to grant"),
        ("kms:CreateGrant", "ability to grant"),
        ("organizations:ListAccounts", "ability to grant"),
        ("s3:ListAllMyBuckets", "enumerates the whole account"),
        ("secretsmanager:ListSecrets", "enumerates the whole account"),
        ("notanaction", "not an IAM action"),
    ],
)
def test_dangerous_actions_are_refused(action, expected):
    with pytest.raises(SpecError) as exc:
        check_action(action, field_path="integrations[0].actions", source=None)
    assert expected in str(exc.value)


def test_a_refusal_names_the_narrower_thing_to_write():
    """A refusal with nowhere to go gets worked around, and a worked-around control is
    worse than none: it still reports success."""
    with pytest.raises(SpecError) as exc:
        check_action("s3:*", field_path="f", source=None)
    assert "s3:GetObject" in str(exc.value)


@pytest.mark.parametrize(
    "resource, expected",
    [
        ("*", "every resource in the account"),
        ("arn:aws:s3", "not a complete ARN"),
        ("arn:aws:s3:::", "wildcards the service or every resource"),
        ("arn:aws:*:eu-west-2:111122223333:thing/x", "wildcards the service"),
        ("arn:aws:s3:::*", "wildcards the service or every resource"),
        ("acme-bucket", "not an ARN"),
    ],
)
def test_over_broad_resources_are_refused(resource, expected):
    with pytest.raises(SpecError) as exc:
        check_resource(resource, field_path="integrations[0].resources", source=None)
    assert expected in str(exc.value)


def test_a_narrow_grant_is_accepted():
    specs = parse_integrations(integration())
    assert len(specs) == 1
    assert specs[0].statements[0].actions == ("s3:GetObject",)


def test_an_integration_with_no_resources_is_refused():
    """Actions with no resources can only mean '*', so it is refused as '*' would be."""
    with pytest.raises(SpecError, match="cannot be rendered"):
        parse_integrations(integration(allow=[{"actions": ["s3:GetObject"], "resources": []}]))


def test_an_empty_allow_block_is_refused():
    with pytest.raises(SpecError, match="Remove the integration, or say what it grants"):
        parse_integrations(integration(allow=[]))


def test_duplicate_integration_ids_are_refused():
    with pytest.raises(SpecError, match="declared twice"):
        parse_integrations(integration() + integration())


def test_an_integration_id_must_be_a_role_name():
    with pytest.raises(SpecError, match="not an integration id"):
        parse_integrations(integration(id="Not A Role Name"))


def test_a_wildcard_model_id_is_refused():
    """A wildcard here grants every model in the region, including ones nobody priced."""
    from nova._fields import Doc

    with pytest.raises(SpecError, match="grants every model in the region"):
        parse_infrastructure(Doc({"bedrock_model_ids": ["anthropic.*"]}))


def test_conditions_are_carried_opaquely():
    """NOVA does not model IAM conditions: a schema for them would only limit what the
    customer can express, and the narrowing is theirs to write."""
    specs = parse_integrations(
        integration(
            allow=[
                {
                    "actions": ["s3:ListBucket"],
                    "resources": ["arn:aws:s3:::acme-crm"],
                    "condition": {"StringLike": {"s3:prefix": ["exports/"]}},
                }
            ]
        )
    )
    rendered = specs[0].to_tfvars()["statements"][0]
    assert rendered["condition"] == {"StringLike": {"s3:prefix": ["exports/"]}}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_refuses_an_integration_for_an_agent_that_does_not_exist():
    """Almost always a rename that left a grant behind — and a grant with nobody to use it
    is the kind of thing that survives three audits and is then used."""
    specs = parse_integrations(integration(agents=["ghost"]))
    with pytest.raises(SpecError, match="does not contain"):
        render_tfvars(
            tenant_id="acme",
            infrastructure=InfrastructureSpec(),
            integrations=specs,
            known_agents=["operations"],
        )


def test_render_produces_terraform_input(tmp_path):
    payload = render_tfvars(
        tenant_id="acme",
        infrastructure=InfrastructureSpec(region="eu-west-2", vpc_id="vpc-1", subnet_id="sub-1"),
        integrations=parse_integrations(integration(agents=["operations"])),
        known_agents=["operations"],
    )
    assert payload["tenant_id"] == "acme"
    assert payload["region"] == "eu-west-2"
    assert payload["integrations"][0]["id"] == "crm-export"

    written = write_tfvars(tmp_path / "nova.auto.tfvars.json", payload)
    assert json.loads(written.read_text()) == payload


def test_no_integrations_means_the_agents_reach_nothing():
    payload = render_tfvars(
        tenant_id="acme", infrastructure=InfrastructureSpec(), integrations=()
    )
    assert payload["integrations"] == []


def test_grants_are_in_the_bundle_digest():
    """Widening a grant must move the provenance. A permission that could change without
    the digest changing would be a permission nobody could prove the age of."""
    narrow = DeploymentSpec.parse({"integrations": integration()})
    wider = DeploymentSpec.parse(
        {
            "integrations": integration(
                allow=[
                    {
                        "actions": ["s3:GetObject", "s3:PutObject"],
                        "resources": ["arn:aws:s3:::acme-crm/*"],
                    }
                ]
            )
        }
    )
    assert narrow.to_dict() != wider.to_dict()


def test_the_example_bundle_renders(bundle):
    """The shipped example must survive its own rules, or nobody can copy it."""
    payload = render_tfvars(
        tenant_id=bundle.tenant_id,
        infrastructure=bundle.deployment.infrastructure,
        integrations=bundle.deployment.integrations,
        known_agents=[spec.id for spec in bundle.agents],
    )
    assert payload["tenant_id"] == bundle.tenant_id
    assert payload["integrations"], "the example declares an integration worth reading"


# ---------------------------------------------------------------------------
# The Terraform module, and keeping the two copies of the rules together
# ---------------------------------------------------------------------------


def test_the_module_exists_and_ships_the_files_a_module_needs():
    for name in (
        "versions.tf", "variables.tf", "main.tf", "iam.tf", "outputs.tf",
        "user_data.sh.tftpl", "terraform.tfvars.example", "README.md", ".gitignore",
    ):
        assert (MODULE / name).is_file(), f"deploy/aws/{name} is missing"


def test_the_module_restates_every_refusal_the_generator_makes():
    """The two copies exist on purpose — the tfvars file can be hand-edited, and the person
    who does that is the one under time pressure. This test is what keeps them together."""
    variables = (MODULE / "variables.tf").read_text(encoding="utf-8")
    for rule in (
        "Integration actions must be literal",
        "Integration resources must not be",
        "iam, sts, organizations, account or kms",
    ):
        assert rule in variables, f"the module no longer refuses: {rule}"


def test_the_module_hard_codes_no_account_id_or_credential():
    """A committed account id is somebody's real account. A committed credential is worse."""
    account = re.compile(r"\b\d{12}\b")
    placeholder = "111122223333"  # AWS's own documentation placeholder
    for path in sorted(MODULE.glob("*")):
        if path.name in (".gitignore",) or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8")
        for found in account.findall(text):
            assert found == placeholder, f"{path.name} carries account id {found}"
        for marker in ("AKIA", "ASIA", "aws_secret_access_key", "-----BEGIN"):
            assert marker not in text, f"{path.name} looks like it carries a credential"


def test_the_runtime_role_grants_no_customer_data_permissions():
    """The central claim. Every grant the runtime role holds is about running itself:
    its models, its image, its logs, its secrets, its key, and the integration roles.

    Asserted against the source rather than a plan because a plan needs an AWS account.
    The rendered result was verified separately; see deploy/aws/README.md.
    """
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    body = iam[iam.index('data "aws_iam_policy_document" "runtime"') : iam.index(
        'resource "aws_iam_role_policy" "runtime"'
    )]
    allowed_services = {"bedrock", "ecr", "logs", "secretsmanager", "kms", "sts"}
    for action in re.findall(r'"([a-z0-9-]+:[A-Za-z]+)"', body):
        service = action.split(":")[0]
        assert service in allowed_services, (
            f"the runtime role now grants {action}, which is a customer-data permission. "
            f"Grants belong on an integration role, not on the runtime role"
        )


def test_the_runtime_may_assume_only_declared_integration_roles():
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    assert "AssumeDeclaredIntegrationsOnly" in iam
    assert "for role in aws_iam_role.integration : role.arn" in iam, (
        "sts:AssumeRole must be scoped to the declared integration roles, one by one"
    )


def test_the_integration_trust_policy_names_the_runtime_role_and_an_external_id():
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    trust = iam[iam.index('data "aws_iam_policy_document" "integration_trust"') :]
    trust = trust[: trust.index('resource "aws_iam_role" "integration"')]
    assert "aws_iam_role.runtime.arn" in trust
    assert "sts:ExternalId" in trust


def test_the_boundary_denies_self_escalation():
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    assert "NeverSelfEscalate" in iam
    assert '"iam:*"' in iam


def test_the_instance_takes_no_inbound_traffic():
    main = (MODULE / "main.tf").read_text(encoding="utf-8")
    assert "aws_vpc_security_group_ingress_rule" not in main, (
        "the runtime is reached through SSM Session Manager; an ingress rule means a port, "
        "a key and a bastion to maintain"
    )
    assert "http_tokens                 = \"required\"" in main or 'http_tokens' in main


def test_the_state_volume_cannot_be_destroyed_by_accident():
    main = (MODULE / "main.tf").read_text(encoding="utf-8")
    volume = main[main.index('resource "aws_ebs_volume" "state"') :]
    volume = volume[: volume.index('resource "aws_volume_attachment"')]
    assert "prevent_destroy = true" in volume, "that volume holds the audit log"


def test_no_rendered_tfvars_is_committed():
    """Generated output beside the thing that generates it becomes a second source of truth,
    and the two disagree exactly when it matters."""
    assert not (MODULE / "nova.auto.tfvars.json").exists()
    assert "nova.auto.tfvars.json" in (MODULE / ".gitignore").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Host bootstrap — two properties the first real deployment depends on
#
# Both were found by running the image against a mounted volume rather than by
# reading the template, and both fail the deployment outright rather than
# degrading, so they are pinned here.
# ---------------------------------------------------------------------------


def test_the_state_volume_is_chowned_to_the_container_user():
    """A freshly formatted volume is root:root; the image runs as uid 10001.

    Without the chown the container exits on first boot with "not writable by uid 10001"
    and systemd restarts it forever. Reproduced in a container before this test existed.
    The chown has to come AFTER `mount -a`: applied to the mountpoint beforehand it
    changes the underlying directory, which the mount then hides. It is recursive because
    a bundle copied in over SSM later arrives owned by root, and a non-recursive chown
    leaves that unreadable — a failure that surfaces at step 23 rather than at boot.
    """
    script = (MODULE / "user_data.sh.tftpl").read_text(encoding="utf-8")
    assert 'chown -R 10001:10001 "$STATE_MOUNT"' in script, (
        "user_data no longer chowns the state volume to the container user; the first "
        "boot will exit with 'not writable by uid 10001'"
    )
    assert script.index("mount -a") < script.index('chown -R 10001:10001 "$STATE_MOUNT"'), (
        "the chown must follow the mount, or it changes the directory under it"
    )


def test_the_container_runs_with_host_networking():
    """Otherwise the Control Center cannot be opened at all.

    The dashboard is a browser and cannot send a bearer token, so it relies on the
    control plane trusting loopback callers. Under Docker's default bridge network a
    published port is NAT'd and the container sees the bridge gateway as the client, not
    127.0.0.1 — so loopback trust never applies and every request, including `GET /`,
    answers 401. Verified for both the `--publish` and the TLS-certificate variants.

    Host networking makes a connection from the host's own loopback arrive as 127.0.0.1.
    Nothing is exposed by it: the process binds 127.0.0.1 by default, the security group
    has no ingress, and reaching it still means an SSM port-forward.
    """
    script = (MODULE / "user_data.sh.tftpl").read_text(encoding="utf-8")
    assert "--network host" in script, (
        "the unit no longer uses host networking; the dashboard will answer 401 to every "
        "request because loopback trust cannot apply behind Docker's bridge NAT"
    )
    # A published port alongside host networking is a contradiction Docker warns about,
    # and it would mean someone reintroduced the bridge-network assumption.
    assert "--publish" not in script, (
        "--publish is meaningless with --network host; remove one of them deliberately"
    )


# ---------------------------------------------------------------------------
# The local validator must work for a non-root operator
#
# validate-local.sh was root-only from the commit that introduced both the
# non-root image and the host-side audit read. It reported "49 passed" for a
# root operator and "46 passed, 3 failed" for everyone else, which is the
# worst possible split: the people it was written to reassure were the ones
# it lied to.
# ---------------------------------------------------------------------------


VALIDATOR = ROOT / "deploy" / "docker" / "validate-local.sh"


def test_the_validator_never_reads_container_state_from_the_host():
    """State the container writes is read back through the container, not off the host.

    The audit log is 0600 and owned by uid 10001 — deliberately, and pinned by
    ``tests/platform/test_audit.py::test_log_is_created_with_restrictive_permissions``.
    A host-side `tail` of it therefore succeeds only for root. The fix is to read it
    inside a container, which is what the image checks in the same script already do.

    Asserted as a path contract rather than a regex on one command: any new host-side
    read of `$ROOT/tenant-*/home/...` is the same defect wearing different clothes.
    """
    script = VALIDATOR.read_text(encoding="utf-8")
    offenders = []
    for lineno, line in enumerate(script.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") or "$ROOT/tenant-" not in stripped:
            continue
        # A host-side reference is fine when it is the -v argument handing the path to a
        # container, or part of the host-built scaffolding before any container runs.
        if "docker run" in stripped or stripped.startswith(("mkdir", "cp ", "sed ", "chmod", "rm ", "python3 -")):
            continue
        if "/home/" in stripped:
            offenders.append(f"  line {lineno}: {stripped[:100]}")
    assert not offenders, (
        "validate-local.sh reads container-written state directly from the host; this "
        "works only for a root operator. Read it through the container instead:\n"
        + "\n".join(offenders)
    )


def test_the_validator_hands_the_temp_tree_back_before_removing_it():
    """Otherwise `rm -rf` leaves the tree behind for every non-root operator.

    The container writes as uid 10001 into directories it creates 0755, so a non-root
    host user has no write permission inside them and cannot unlink their contents.
    Only root can chown a file to another user, so a throwaway `--user 0:0` container is
    the only way to hand the tree back.
    """
    script = VALIDATOR.read_text(encoding="utf-8")
    assert "--user 0:0" in script and "chown -R" in script, (
        "the cleanup no longer hands the temp tree back to the host user; a non-root "
        "operator will be left with an unremovable directory in /tmp after every run"
    )
    assert script.index("chown -R") < script.rindex('rm -rf "$ROOT"'), (
        "the chown must precede the rm, or the rm still fails"
    )


# ---------------------------------------------------------------------------
# The KMS key policy
#
# The first real deployment failed creating the log group:
#
#   AccessDeniedException: The specified KMS key does not exist or is not
#   allowed to be used with Arn '...:log-group:/nova/test'
#
# CloudWatch Logs calls KMS as a service principal, not as an IAM identity, so
# no IAM policy can reach it — the key policy is the only place it can be
# allowed, and the key had none, so it got the KMS default (root only).
# ---------------------------------------------------------------------------


def _main_tf() -> str:
    return (MODULE / "main.tf").read_text(encoding="utf-8")


def test_the_key_policy_allows_cloudwatch_logs():
    """Without this the log group cannot be created at all — proven in a real account."""
    main = _main_tf()
    assert 'data "aws_iam_policy_document" "state_key"' in main, (
        "the KMS key has no key policy document; CloudWatch Logs cannot use the key and "
        "aws_cloudwatch_log_group.runtime will fail with AccessDeniedException"
    )
    assert "logs.${data.aws_region.current.name}.amazonaws.com" in main, (
        "the CloudWatch Logs service principal is missing, or the region is hardcoded. "
        "The principal is regional and must follow the provider's region"
    )
    assert "policy                  = data.aws_iam_policy_document.state_key[0].json" in main, (
        "the key policy document exists but is not attached to aws_kms_key.state"
    )


def test_the_key_policy_keeps_root_administrative_control():
    """Dropping this statement bricks the key.

    KMS only honours IAM policies against a key when the key policy says so. Remove the
    root statement and no principal — including the account's administrators — can use or
    even re-open the key policy, and a key policy can only be changed by a principal the
    key policy already allows. There is no self-service recovery.
    """
    main = _main_tf()
    assert 'sid       = "EnableIAMUserPermissions"' in main
    assert 'identifiers = ["arn:${local.partition}:iam::${local.account_id}:root"]' in main, (
        "the root statement is missing or no longer names this account's root; the key "
        "would become unmanageable"
    )


def test_cloudwatch_logs_is_scoped_by_encryption_context():
    """The grant is to a service principal, so the condition is what bounds it.

    Without the encryption-context condition, CloudWatch Logs could use this tenant's key
    for any log group in the account.
    """
    main = _main_tf()
    assert 'variable = "kms:EncryptionContext:aws:logs:arn"' in main, (
        "the CloudWatch Logs grant is unconditional; any log group in the account could "
        "be encrypted with this tenant's key"
    )
    assert "log-group:${local.log_group_name}" in main, (
        "the condition no longer scopes to this deployment's own log group"
    )


def test_the_key_policy_grants_no_management_actions_to_the_service():
    """`kms:*` for the service principal would let CloudWatch Logs rewrite the policy."""
    main = _main_tf()
    start = main.index('sid    = "AllowCloudWatchLogs"')
    block = main[start:main.index("}", main.index("principals", start))]
    for forbidden in ("kms:*", "kms:PutKeyPolicy", "kms:ScheduleKeyDeletion",
                      "kms:CreateGrant", "kms:DisableKey"):
        assert forbidden not in block, (
            f"the CloudWatch Logs statement grants {forbidden}; it needs only the "
            "encrypt/decrypt/describe set"
        )


def test_an_externally_supplied_key_has_no_policy_managed_here():
    """A customer-supplied key is theirs. Rewriting its policy could lock out its owner.

    Both the document and the key are gated on the same condition, and there is no
    `aws_kms_key_policy` resource — which is the only way this module could reach out and
    modify a key it did not create.
    """
    main = _main_tf()
    document = main[main.index('data "aws_iam_policy_document" "state_key"'):]
    assert 'count = var.kms_key_arn == "" ? 1 : 0' in document[:document.index("statement")], (
        "the key policy document is not gated on this module owning the key"
    )
    # The resource declaration, not the bare name — main.tf mentions it in a comment
    # explaining why it is deliberately absent, and a substring check matched that.
    assert 'resource "aws_kms_key_policy"' not in main, (
        "an aws_kms_key_policy resource can target a key this module does not own; the "
        "policy belongs inline on the key we create"
    )


def test_the_log_group_name_is_defined_once():
    """The key policy scopes to the log group by name, so the two cannot drift apart."""
    main = _main_tf()
    assert 'log_group_name = "/nova/${var.tenant_id}"' in main
    assert "name              = local.log_group_name" in main, (
        "the log group no longer derives its name from the local the key policy scopes to"
    )


# ---------------------------------------------------------------------------
# The SSM ceiling
#
# The first real deployment came up with AmazonSSMManagedInstanceCore attached
# and the instance never appeared in Systems Manager. A permissions boundary is
# an intersection, not a grant: the managed policy allowed
# ssm:UpdateInstanceInformation and the boundary did not, so the effective
# permission was deny and the agent could not register. With no inbound rule and
# no SSH key, Session Manager is the only route in, so that is the whole
# deployment unreachable.
# ---------------------------------------------------------------------------


def _boundary_ceiling_actions() -> set[str]:
    """The actions inside the boundary's ``ServicesThisDeploymentUses`` statement.

    Parsed rather than grepped so a mention in a comment cannot pass for a grant — the
    reason this fix needed making is that the list, not the prose, is what AWS reads.
    """
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    start = iam.index('sid    = "ServicesThisDeploymentUses"')
    block = iam[start : iam.index("]", iam.index("actions = [", start))]
    body = "\n".join(line.split("#")[0] for line in block.splitlines())
    return set(re.findall(r'"([a-z0-9-]+:[A-Za-z*]+)"', body))


def _permits(ceiling: set[str], action: str) -> bool:
    return action in ceiling or f"{action.split(':')[0]}:*" in ceiling


def test_the_boundary_permits_the_ssm_agent_to_register():
    """Without this the instance is never a managed node and there is no way in.

    The SSM Agent's health module calls exactly one API — ``UpdateInstanceInformation``
    — to register and to hold the five-minute heartbeat that keeps the node Online.
    Denied, ``aws ssm start-session --target <id>`` fails against an instance whose
    security group has no ingress rule.
    """
    assert _permits(_boundary_ceiling_actions(), "ssm:UpdateInstanceInformation"), (
        "the permissions boundary does not allow ssm:UpdateInstanceInformation, so the "
        "SSM Agent cannot register. AmazonSSMManagedInstanceCore allows it, but a "
        "boundary is a ceiling: the attached policy cannot exceed it"
    )


@pytest.mark.parametrize(
    "action",
    [
        "ssmmessages:CreateControlChannel",
        "ssmmessages:CreateDataChannel",
        "ssmmessages:OpenControlChannel",
        "ssmmessages:OpenDataChannel",
    ],
)
def test_the_boundary_permits_the_session_manager_channels(action):
    """Registration gets the node listed; these four carry the session itself."""
    assert _permits(_boundary_ceiling_actions(), action), (
        f"the boundary no longer permits {action}; Session Manager cannot open a "
        "session, and it is the only route into this instance"
    )


def test_the_boundary_does_not_widen_to_all_of_ssm():
    """The shape of the five-o'clock fix this whole file exists to prevent.

    The boundary carries the Session Manager path and the parameter reads this
    deployment makes, named one at a time. ``ssm:*`` would also lift the ceiling on
    Run Command, Inventory, Patch Manager and every parameter in the account.
    """
    assert "ssm:*" not in _boundary_ceiling_actions(), (
        "the boundary now allows ssm:*. Add the specific actions the agent needs "
        "instead — a boundary that names a whole service stops being a ceiling"
    )


def test_the_session_manager_grant_still_comes_from_the_managed_policy():
    """The boundary permits; something still has to grant. Both halves are load-bearing."""
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    assert "iam::aws:policy/AmazonSSMManagedInstanceCore" in iam, (
        "the managed policy attachment is gone. The boundary only caps permissions; "
        "with nothing attached that grants them, the agent still cannot register"
    )


# ---------------------------------------------------------------------------
# ECR repository scoping
#
# The deployment runs two images from two repositories: the control plane, and
# the Hermes runtime that dispatches its work. The runtime role's pull grant was
# derived from `image_uri` alone, so a worker in its own repository could not be
# pulled — the instance would come up with a systemd unit that fails on
# `docker pull` and a dispatcher that never starts.
#
# The fix is a derivation, not a list: this module deploys into arbitrary
# customer accounts and arbitrary repositories, so a hardcoded name would be
# wrong everywhere except the account it was written in.
# ---------------------------------------------------------------------------

_ECR_PULL_ACTIONS = (
    "ecr:BatchCheckLayerAvailability",
    "ecr:BatchGetImage",
    "ecr:GetDownloadUrlForLayer",
)


def _pull_statement() -> str:
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    start = iam.index('sid    = "PullItsOwnImage"')
    return iam[start : iam.index("}", iam.index("resources", start))]


def test_the_pull_grant_covers_every_image_this_deployment_runs():
    """A worker in its own repository is unpullable if this names only one."""
    assert "local.image_repository_arns" in _pull_statement(), (
        "the ECR pull statement still scopes to a single derived repository, so a "
        "deployment with a worker_image_uri cannot pull its worker image"
    )


def test_both_repository_arns_are_derived_from_their_image_uris():
    """Never a hardcoded repository name: this module deploys into customer accounts."""
    main = _main_tf()
    assert "control_plane = var.image_uri" in main
    assert "worker        = var.worker_image_uri" in main or "worker = var.worker_image_uri" in main
    for hardcoded in ("repository/nova-runtime", "repository/nova-control-plane"):
        assert hardcoded not in main, (
            f"{hardcoded} is hardcoded; the ARN must be derived from the image URI so "
            "the module works in a customer's own account and repositories"
        )


def test_an_absent_worker_derives_exactly_one_repository():
    """The control-plane-only deployment must be unchanged, not merely still valid."""
    main = _main_tf()
    assert 'name => uri if trimspace(uri) != ""' in main, (
        "an empty worker_image_uri must drop out of the derivation; otherwise a "
        "control-plane-only deployment derives a malformed second ARN"
    )


def test_the_pull_grant_never_widens_to_a_whole_registry():
    """The refusal this file exists for. Two named repositories, never a wildcard."""
    statement = _pull_statement()
    for forbidden in ('"*"', "repository/*", ":repository/*"):
        assert forbidden not in statement, (
            f"the ECR pull statement contains {forbidden}; an instance that can pull any "
            "image in the account can pull one nobody reviewed"
        )


def test_the_pull_grant_gains_no_unrelated_ecr_permissions():
    """Layer reads only. Push, delete and describe are not part of running an image."""
    statement = _pull_statement()
    for action in re.findall(r'"(ecr:[A-Za-z]+)"', statement):
        assert action in _ECR_PULL_ACTIONS, (
            f"the pull statement now grants {action}, which is not needed to run an image"
        )


def test_the_auth_token_grant_is_unchanged_and_still_account_wide():
    """`ecr:GetAuthorizationToken` takes no resource; AWS models it as account-wide.
    Scoping it to a repository would silently break every pull."""
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    start = iam.index('sid       = "EcrAuth"')
    block = iam[start : iam.index("}", start)]
    assert '"ecr:GetAuthorizationToken"' in block
    assert 'resources = ["*"]' in block


def _boundary() -> str:
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    return iam[iam.index('data "aws_iam_policy_document" "runtime_boundary"') : iam.index(
        'resource "aws_iam_policy" "runtime_boundary"'
    )]


def test_the_boundary_caps_image_pulls_to_the_same_repositories():
    """A boundary is an intersection: whatever it omits, the runtime policy cannot grant.

    Both halves read `local.image_repository_arns`, so there is one list and the two
    cannot drift. That is what makes stating it twice safe here — two hand-written lists
    would deny with a message naming neither of them, which is how the SSM heartbeat was
    lost.
    """
    boundary = _boundary()
    assert 'sid    = "EcrLayerPullCeiling"' in boundary, (
        "the boundary no longer bounds image pulls by repository"
    )
    ceiling = boundary[boundary.index('sid    = "EcrLayerPullCeiling"'):]
    ceiling = ceiling[: ceiling.index("\n  }")]
    assert "resources = local.image_repository_arns" in ceiling, (
        "the boundary's pull ceiling does not read the same derived list as the runtime "
        "policy; two lists that can disagree is the bug this shape exists to prevent"
    )
    for action in _ECR_PULL_ACTIONS:
        assert f'"{action}"' in ceiling, (
            f"the boundary no longer permits {action}, so the runtime policy's pull grant "
            "is capped away and the instance cannot pull any image"
        )


def test_the_boundary_never_caps_the_auth_token_to_a_repository():
    """`ecr:GetAuthorizationToken` takes no resource. In the boundary's ceiling it would
    match nothing, and every pull would fail before it reached a repository."""
    boundary = _boundary()
    services = boundary[boundary.index('sid    = "ServicesThisDeploymentUses"'):]
    services = services[: services.index('resources = ["*"]')]
    assert '"ecr:GetAuthorizationToken"' in services, (
        "the auth-token action moved out of the wildcard statement; it takes no resource "
        "and a repository-scoped ceiling would deny it"
    )
    ceiling = boundary[boundary.index('sid    = "EcrLayerPullCeiling"'):]
    assert '"ecr:GetAuthorizationToken"' not in ceiling[: ceiling.index("\n  }")]


def test_the_boundary_and_the_runtime_policy_name_one_list():
    """Neither side may grow a hand-written repository ARN."""
    iam = (MODULE / "iam.tf").read_text(encoding="utf-8")
    # Counted as bindings, not mentions: the comment beside the ceiling names the local
    # too, and a test that counts prose fails the next time somebody explains it better.
    bindings = re.findall(r"resources\s*=\s*local\.image_repository_arns", iam)
    assert len(bindings) == 2, (
        "the boundary and the runtime policy must each read the derived list exactly "
        f"once; found {len(bindings)}. A second source of repository ARNs is the drift "
        "this shape exists to prevent"
    )
    assert ":repository/" not in iam, (
        "a repository ARN is written into iam.tf by hand; it must be derived in main.tf "
        "from the image URIs so the module works in a customer's own account"
    )


def test_a_non_ecr_worker_image_is_refused_with_a_sentence():
    """The derivation reads the registry host positionally, so a Docker Hub reference
    would fail mid-plan with 'Invalid index' rather than saying what is wrong."""
    variables = (MODULE / "variables.tf").read_text(encoding="utf-8")
    worker = variables[variables.index('variable "worker_image_uri"') :]
    worker = worker[: worker.index("\nvariable ")]
    assert "validation {" in worker
    assert 'trimspace(var.worker_image_uri) == ""' in worker, (
        "the validation must still admit the empty control-plane-only default"
    )
    assert "dkr" in worker and "ecr" in worker


_ARNS_LOCAL = "  image_repository_arns = distinct(values(local.ecr_repository_arns))"


def _derive_repository_arns(tmp_path, image: str, worker: str) -> list[str]:
    """What Terraform itself derives, from the module's own locals.

    Lifted verbatim into a provider-free module rather than re-implemented in Python: the
    tag-versus-digest handling is the part that is easy to get wrong, and a second
    implementation here would assert my reading of the expression instead of the
    expression. Provider-free, so it runs offline.
    """
    main = _main_tf()
    block = main[main.index("  ecr_image_uris = {") : main.index(_ARNS_LOCAL) + len(_ARNS_LOCAL)]
    (tmp_path / "main.tf").write_text(
        'variable "image_uri" { type = string }\n'
        'variable "worker_image_uri" { type = string }\n\n'
        'locals {\n  partition = "aws"\n' + block + "\n}\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["terraform", "init", "-backend=false", "-input=false"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    result = subprocess.run(
        ["terraform", "console", "-var", f"image_uri={image}",
         "-var", f"worker_image_uri={worker}"],
        cwd=tmp_path, input="local.image_repository_arns\n",
        text=True, capture_output=True, check=True,
    )
    return re.findall(r'"(arn:[^"]+)"', result.stdout)


@pytest.mark.skipif(
    shutil.which("terraform") is None, reason="terraform is not installed here"
)
@pytest.mark.parametrize(
    "image, worker, expected",
    [
        pytest.param(
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova-control-plane@sha256:" + "5" * 64,
            "",
            ["arn:aws:ecr:eu-west-2:369607682697:repository/nova-control-plane"],
            id="control-plane-only-by-digest",
        ),
        pytest.param(
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova-control-plane@sha256:" + "5" * 64,
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova-runtime@sha256:" + "2" * 64,
            [
                "arn:aws:ecr:eu-west-2:369607682697:repository/nova-control-plane",
                "arn:aws:ecr:eu-west-2:369607682697:repository/nova-runtime",
            ],
            id="both-images",
        ),
        pytest.param(
            "111122223333.dkr.ecr-fips.us-east-1.amazonaws.com/team/nova:1.4.0",
            "444455556666.dkr.ecr.eu-west-2.amazonaws.com/nova-runtime:0.21.1",
            [
                "arn:aws:ecr:us-east-1:111122223333:repository/team/nova",
                "arn:aws:ecr:eu-west-2:444455556666:repository/nova-runtime",
            ],
            id="fips-host-namespaced-repo-and-a-second-account",
        ),
    ],
)
def test_terraform_derives_the_expected_repository_arns(tmp_path, image, worker, expected):
    """Evaluated by Terraform itself, not re-implemented in Python.

    The tag-versus-digest handling is the part that is easy to get wrong, and a second
    implementation of it here would assert my reading of the expression rather than the
    expression. The locals are lifted verbatim into a provider-free module so this runs
    offline.
    """
    assert _derive_repository_arns(tmp_path, image, worker) == expected


def test_one_repository_serving_both_images_is_listed_once():
    """Both images in one repository under different tags is an ordinary layout.

    IAM ignores the duplicate, so this is not a security property — it is a readability
    and diff-churn one, and a policy nobody can read at a glance is a policy nobody
    reviews.
    """
    main = _main_tf()
    assert "distinct(values(local.ecr_repository_arns))" in main, (
        "two images in one repository would produce the same ARN twice in the policy"
    )


@pytest.mark.skipif(
    shutil.which("terraform") is None, reason="terraform is not installed here"
)
@pytest.mark.parametrize(
    "image, worker, expected",
    [
        pytest.param(
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova:control-1.0",
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova:worker-0.21.1",
            ["arn:aws:ecr:eu-west-2:369607682697:repository/nova"],
            id="one-repository-two-tags",
        ),
        pytest.param(
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova@sha256:" + "a" * 64,
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova@sha256:" + "b" * 64,
            ["arn:aws:ecr:eu-west-2:369607682697:repository/nova"],
            id="one-repository-two-digests",
        ),
        pytest.param(
            "369607682697.dkr.ecr.eu-west-2.amazonaws.com/nova-control-plane:1.0",
            "   ",
            ["arn:aws:ecr:eu-west-2:369607682697:repository/nova-control-plane"],
            id="whitespace-only-worker-is-empty",
        ),
    ],
)
def test_terraform_deduplicates_and_tolerates_a_blank_worker(
    tmp_path, image, worker, expected,
):
    assert _derive_repository_arns(tmp_path, image, worker) == expected
