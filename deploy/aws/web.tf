# ---------------------------------------------------------------------------
# The Control Centre on the web: a load balancer that signs people in with Cognito
# ---------------------------------------------------------------------------
#
# Off by default (web_console_enabled). The default way in is an SSM port forward
# (access.tf), which suits the people who operate a deployment. This is for the customer's
# own staff: a normal HTTPS address, a sign-in page with MFA, and a role per person.
#
#   browser --HTTPS--> load balancer --authenticate-cognito--> Cognito sign-in (password + MFA)
#                           | then, with the user pool's access token in x-amzn-oidc-accesstoken
#                           v
#                      instance:8787 (security group admits the load balancer only)
#                           |
#                      NOVA verifies the token's signature against the pool's own keys and
#                      maps its Cognito group to a role (nova/control/oidc.py)
#
# Roles are Cognito groups: nova-admin and nova-viewer. Signed in and in neither is refused.
#
# **In this mode the tunnel needs a sign-in too.** The control plane runs behind a TLS proxy,
# so it stops trusting loopback (a proxy's traffic would otherwise all look local). Operators
# are simply nova-admin members and sign in like everyone else.
#
# Needs the customer's own: a domain name for the console, an ACM certificate for it in this
# region, and two public subnets in different availability zones for the load balancer.

locals {
  web_enabled     = var.web_console_enabled
  cognito_domain  = "${var.web_cognito_domain_prefix}.auth.${var.region}.amazoncognito.com"
  web_logout_url  = local.web_enabled ? "https://${local.cognito_domain}/logout?client_id=${aws_cognito_user_pool_client.console[0].id}&logout_uri=https://${var.web_domain_name}/" : ""
  web_oidc_issuer = local.web_enabled ? "https://cognito-idp.${var.region}.amazonaws.com/${aws_cognito_user_pool.console[0].id}" : ""
}

# -- who can sign in ----------------------------------------------------------------------

resource "aws_cognito_user_pool" "console" {
  count = local.web_enabled ? 1 : 0
  name  = "${local.name_prefix}-console"

  # People are invited by an administrator; nobody signs themselves up to a company's
  # control plane.
  admin_create_user_config {
    allow_admin_create_user_only = true
  }

  mfa_configuration = "ON"
  software_token_mfa_configuration {
    enabled = true
  }

  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = false
    temporary_password_validity_days = 3
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }
  auto_verified_attributes = ["email"]
  username_attributes      = ["email"]

  # The pool is who may administer the tenant; deleting it by accident locks everyone out.
  deletion_protection = "ACTIVE"
}

resource "aws_cognito_user_pool_domain" "console" {
  count        = local.web_enabled ? 1 : 0
  domain       = var.web_cognito_domain_prefix
  user_pool_id = aws_cognito_user_pool.console[0].id
}

resource "aws_cognito_user_pool_client" "console" {
  count        = local.web_enabled ? 1 : 0
  name         = "${local.name_prefix}-console"
  user_pool_id = aws_cognito_user_pool.console[0].id

  # The load balancer is a confidential client: it needs the secret, and it runs the code flow.
  generate_secret                      = true
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email"]
  supported_identity_providers         = ["COGNITO"]
  callback_urls                        = ["https://${var.web_domain_name}/oauth2/idpresponse"]
  logout_urls                          = ["https://${var.web_domain_name}/"]
  prevent_user_existence_errors        = "ENABLED"

  access_token_validity  = 60
  id_token_validity      = 60
  refresh_token_validity = 12
  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "hours"
  }
}

resource "aws_cognito_user_group" "admin" {
  count        = local.web_enabled ? 1 : 0
  name         = "nova-admin"
  description  = "Control Centre administrators: every change the Control Centre can make."
  user_pool_id = aws_cognito_user_pool.console[0].id
}

resource "aws_cognito_user_group" "viewer" {
  count        = local.web_enabled ? 1 : 0
  name         = "nova-viewer"
  description  = "Control Centre viewers: read everything, change nothing."
  user_pool_id = aws_cognito_user_pool.console[0].id
}

# -- the way in ---------------------------------------------------------------------------

resource "aws_security_group" "web" {
  count       = local.web_enabled ? 1 : 0
  name        = "${local.name_prefix}-web"
  description = "NOVA Control Centre load balancer: HTTPS in, the control plane out."
  vpc_id      = var.vpc_id
  tags        = { Name = "${local.name_prefix}-web" }
}

resource "aws_vpc_security_group_ingress_rule" "web_https" {
  for_each          = local.web_enabled ? toset(var.web_allowed_cidrs) : toset([])
  security_group_id = aws_security_group.web[0].id
  description       = "Control Centre over HTTPS"
  cidr_ipv4         = each.value
  from_port         = 443
  to_port           = 443
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_ingress_rule" "web_http_redirect" {
  for_each          = local.web_enabled ? toset(var.web_allowed_cidrs) : toset([])
  security_group_id = aws_security_group.web[0].id
  description       = "Redirected to HTTPS; nothing is served on port 80"
  cidr_ipv4         = each.value
  from_port         = 80
  to_port           = 80
  ip_protocol       = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "web_to_control_plane" {
  count                        = local.web_enabled ? 1 : 0
  security_group_id            = aws_security_group.web[0].id
  description                  = "To the control plane only"
  referenced_security_group_id = aws_security_group.runtime.id
  from_port                    = local.control_port
  to_port                      = local.control_port
  ip_protocol                  = "tcp"
}

# The runtime's own group gains one rule: the control plane port, from the load balancer.
resource "aws_vpc_security_group_ingress_rule" "control_plane_from_web" {
  count                        = local.web_enabled ? 1 : 0
  security_group_id            = aws_security_group.runtime.id
  description                  = "Control Centre, from its load balancer only"
  referenced_security_group_id = aws_security_group.web[0].id
  from_port                    = local.control_port
  to_port                      = local.control_port
  ip_protocol                  = "tcp"
}

resource "aws_lb" "web" {
  count              = local.web_enabled ? 1 : 0
  name               = substr("${local.name_prefix}-web", 0, 32)
  load_balancer_type = "application"
  internal           = false
  subnets            = var.web_public_subnet_ids
  security_groups    = [aws_security_group.web[0].id]

  # Malformed header names are dropped rather than forwarded. The sign-in headers
  # themselves (x-amzn-oidc-*) are set by the load balancer's authenticate action, which
  # replaces any a client sends; NOVA verifies the token regardless (nova/control/oidc.py).
  drop_invalid_header_fields = true
  enable_deletion_protection = true

  lifecycle {
    precondition {
      condition = (
        var.web_domain_name != "" && var.web_certificate_arn != "" &&
        var.web_cognito_domain_prefix != "" && length(var.web_public_subnet_ids) >= 2 &&
        length(var.web_allowed_cidrs) >= 1
      )
      error_message = "web_console_enabled needs web_domain_name, web_certificate_arn, web_cognito_domain_prefix, two web_public_subnet_ids and at least one web_allowed_cidrs entry."
    }
  }
}

resource "aws_lb_target_group" "control_plane" {
  count       = local.web_enabled ? 1 : 0
  name        = substr("${local.name_prefix}-cp", 0, 32)
  port        = local.control_port
  protocol    = "HTTP"
  target_type = "instance"
  vpc_id      = var.vpc_id

  # The health probe carries no sign-in, so a live control plane answers it 401 — which is
  # exactly the answer that proves it is up and refusing the unauthenticated.
  health_check {
    path    = "/platform/v1/health"
    matcher = "200,401"
  }
}

resource "aws_lb_target_group_attachment" "control_plane" {
  count            = local.web_enabled ? 1 : 0
  target_group_arn = aws_lb_target_group.control_plane[0].arn
  target_id        = aws_instance.runtime.id
  port             = local.control_port
}

resource "aws_lb_listener" "https" {
  count             = local.web_enabled ? 1 : 0
  load_balancer_arn = aws_lb.web[0].arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.web_certificate_arn

  default_action {
    type  = "authenticate-cognito"
    order = 1
    authenticate_cognito {
      user_pool_arn              = aws_cognito_user_pool.console[0].arn
      user_pool_client_id        = aws_cognito_user_pool_client.console[0].id
      user_pool_domain           = aws_cognito_user_pool_domain.console[0].domain
      on_unauthenticated_request = "authenticate"
      scope                      = "openid email"
      session_timeout            = 28800
    }
  }

  default_action {
    type             = "forward"
    order            = 2
    target_group_arn = aws_lb_target_group.control_plane[0].arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  count             = local.web_enabled ? 1 : 0
  load_balancer_arn = aws_lb.web[0].arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      protocol    = "HTTPS"
      port        = "443"
      status_code = "HTTP_301"
    }
  }
}
