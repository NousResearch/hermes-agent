# Who can open the Control Centre

The Control Centre listens on the runtime instance's loopback only. Nothing on the network
reaches it. People open it through an **AWS Systems Manager port-forwarding session**, so
the Control Centre's login is AWS's: whoever AWS lets start that session gets in, and
AWS decides that with the account's own identity system — single sign-on, MFA, and the
audit trail in CloudTrail.

The module creates one IAM policy for this, `nova-<tenant>-console-access` (Terraform output
`console_access_policy_arn`). It allows exactly one thing: forwarding a local port to the
Control Centre port on this one instance.

| It allows | It refuses |
|---|---|
| A port forward to the Control Centre port (8787) on this instance | A shell on the instance: `ssm:SessionDocumentAccessCheck` makes a StartSession without a document fail instead of opening the default shell |
| Ending or resuming the caller's own sessions | Any other port on the host: the session document `nova-<tenant>-console` fixes the remote port. AWS's generic port-forward document takes any port. |
| | Any other instance, and every other AWS action |

Everyone who gets in is an admin of the Control Centre. Loopback callers are trusted, and
anyone who can reach loopback on this host could already read its files. **Give this policy
only to people who should administer the tenant.** For per-person roles (admin/viewer) and a
normal web address, see option C.

---

## Option A: IAM Identity Center (recommended)

Single sign-on with MFA, one login for every tenant account, no long-lived keys.

IAM Identity Center can grant access to AWS **accounts** only when it is enabled from
**AWS Organizations**. A standalone account instance of Identity Center cannot. So:

1. **Enable AWS Organizations.** Console → *AWS Organizations* → *Create an organization*.
   It is free. This account becomes the management account, and nothing about the running
   deployment changes.
2. **Enable IAM Identity Center** in the deployment region (eu-west-2). Console →
   *IAM Identity Center* → *Enable*. Choose the organization instance.
3. **Require MFA.** *Settings → Authentication → Multi-factor authentication*: require it
   at every sign-in, and allow authenticator apps and security keys.
4. **Create the people**, in *Users*, and a group such as `nova-operators`. Or connect
   Google Workspace or Microsoft Entra ID under *Settings → Identity source*, so people sign
   in with the accounts they already have.
5. **Create a permission set** named `NovaConsole`. Choose *Custom permission set*, then
   *Attach customer managed policies* → `nova-<tenant>-console-access`. Set the session
   duration to 8 hours.
6. **Assign it.** *AWS accounts* → this account → *Assign users or groups* →
   `nova-operators` → `NovaConsole`.
7. **Each operator, once:**
   ```bash
   aws configure sso            # start URL from Identity Center's dashboard, region eu-west-2
   ```
   Then, each time:
   ```bash
   aws sso login --profile nova-console
   AWS_PROFILE=nova-console deploy/aws/console.sh <instance-id> nova-<tenant>-console
   ```
   Then open http://localhost:8787.

For several client accounts, add each account to the organization and assign the same
permission set. One sign-in reaches every tenant the person is assigned to.

## Option B: an IAM group with MFA (for an account without Organizations)

Set `console_iam_group_enabled = true` and apply. This creates `nova-<tenant>-console`,
which has the console policy and a rule that **denies everything until the person signs
in with MFA**.

1. Create an IAM user per person, with no console password needed and an access key for
   the CLI. Add them to the group.
2. Each person registers a virtual MFA device (*IAM → Users → Security credentials*).
3. Each time:
   ```bash
   AWS_MFA_SERIAL=arn:aws:iam::<account>:mfa/<user> \
     deploy/aws/console.sh <instance-id> nova-<tenant>-console
   ```
   The script asks for the MFA code and uses 8-hour credentials for the session.

This works today, but it means long-lived access keys per person. Move to option A when you
can.

## Option C: the Control Centre as a website (for the customer's own staff)

`web_console_enabled = true` (web.tf) serves the Control Centre at the customer's own
address, behind an HTTPS load balancer that signs people in with **Amazon Cognito**:
password plus a required authenticator app, and people are invited by an administrator only.

| | |
|---|---|
| Roles | Cognito groups: **nova-admin** (everything) and **nova-viewer** (read only). Signed in but in neither group is refused with a sentence saying so. |
| How NOVA knows who it is | The load balancer forwards the user pool's access token. NOVA **verifies its signature** against the pool's published keys and checks issuer, app client, token type and expiry (`nova/control/oidc.py`). It does not trust the header just because of where it came from. |
| Network | The load balancer's security group admits `web_allowed_cidrs` on 443 (80 only redirects). The instance admits the Control Centre port from the load balancer and nothing else. |
| Sign out | The top bar shows who is signed in and a **Sign out** link. It clears the load balancer session, then Cognito's. |

**What the customer provides:** a domain name (`web_domain_name`), an ACM certificate for it
in the deployment region, two public subnets in different availability zones, the address
ranges allowed to reach the sign-in page (`web_allowed_cidrs`; state `0.0.0.0/0` explicitly
if it really is the whole internet), and a unique `web_cognito_domain_prefix`.

**After apply:**
1. Point the domain at `terraform output web_load_balancer_dns`.
2. Invite people: Cognito → the pool (`terraform output web_user_pool_id`) → *Create user* with
   their email → add them to `nova-admin` or `nova-viewer`. Cognito emails a temporary
   password, and they register an authenticator app on first sign-in.

**Two things change in this mode:**
- **The SSM tunnel needs a sign-in too.** Behind a load balancer the control plane stops
  trusting loopback, because the load balancer's traffic would otherwise all look local.
  Operators are nova-admin members and use the website like everyone else.
- **An existing instance needs its environment updated once.** The bootstrap runs only at
  first boot (see README, "Changing the bootstrap"), so a deployment that already exists gets
  the web settings either by `terraform apply -replace=aws_instance.runtime` (the state
  volume is kept), or by appending the five `NOVA_BIND_HOST` / `NOVA_BEHIND_TLS_PROXY` /
  `NOVA_OIDC_*` lines (from the rendered bootstrap) to `/etc/nova.env` and restarting
  `nova.service`.

Cost: roughly $20–25 a month for the load balancer, plus Cognito, whose free tier covers a
small team. Check the current Cognito pricing page for the tier and user count you expect.

## What not to do

- **Do not use the deployment user** (the one that runs Terraform) to open the Control
  Centre. It can do far more than open a tunnel.
- **Do not publish port 8787** to anything but the load balancer's own security group, which
  is the only rule option C adds. The control plane refuses a non-loopback bind without
  sign-in (a principals file or Cognito) and TLS in front of it, and it is right to.

`terraform output -raw console_command` prints the exact command for a deployment.
