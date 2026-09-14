"""Deployment artifacts — turning a declaration into infrastructure NOVA can prove.

``docs/platform/ARCHITECTURE_BOUNDARIES.md`` §6 promises three things about a customer's
AWS account: the runtime role carries no customer-data permissions, every integration is a
separately-scoped role assumed by that runtime role, and **agents never receive broad access
to the customer's AWS account**. Until this package existed those were prose, and prose is
not a control — a promise whose violation is silent is a promise the first rushed deployment
will break.

So the grants are declared in the tenant bundle, checked here, and rendered into Terraform
input. The checking is the point. The Terraform module restates the same refusals in
``variable "integrations"`` validation blocks, because a check that lives only in the
generator is a check an operator skips by hand-editing the tfvars — and the one who does
that is exactly the one under time pressure.

NOVA still writes no secrets and touches no AWS API. It renders a file; the customer's own
DevOps team runs Terraform with their own credentials. We never require, hold, or ask for
root or admin access to their account.
"""

from nova.deploy.aws import (
    IntegrationSpec,
    InfrastructureSpec,
    Statement,
    render_tfvars,
    write_tfvars,
)

__all__ = [
    "IntegrationSpec",
    "InfrastructureSpec",
    "Statement",
    "render_tfvars",
    "write_tfvars",
]
