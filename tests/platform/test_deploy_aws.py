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
