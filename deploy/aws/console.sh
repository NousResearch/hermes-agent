#!/bin/bash
# Open a tenant's Control Centre on this machine through an SSM port forward.
#
#   deploy/aws/console.sh <instance-id> <session-document> [region] [local-port]
#   (terraform output -raw console_command prints the full line for a deployment)
#
# Who may run this is decided by AWS, not NOVA: your IAM Identity Center sign-in
# (aws sso login --profile <p>; export AWS_PROFILE=<p>) or an IAM user in the tenant's
# console group. That group requires MFA; set AWS_MFA_SERIAL to your MFA device ARN and
# this asks for a code and uses short-lived credentials for the session.
#
# Nothing is opened to the network: the port is forwarded to 127.0.0.1 on your machine
# only, and the session can reach the Control Centre port and nothing else on the host.
set -euo pipefail

USAGE="usage: console.sh <instance-id> <session-document> [region] [local-port]"
INSTANCE=${1:?$USAGE}
# nova-<tenant>-console: the tenant's own session document, which fixes the remote port.
# Named rather than looked up, because looking it up needs ssm:ListDocuments, which the
# console policy deliberately does not grant.
DOCUMENT=${2:?$USAGE}
REGION=${3:-${AWS_REGION:-eu-west-2}}
LOCAL_PORT=${4:-8787}

command -v session-manager-plugin >/dev/null || {
  echo "The AWS Session Manager plugin is not installed:" >&2
  echo "  https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html" >&2
  exit 1
}

if [ -n "${AWS_MFA_SERIAL:-}" ]; then
  read -rp "MFA code for ${AWS_MFA_SERIAL##*/}: " CODE
  read -r AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN < <(
    aws sts get-session-token --serial-number "$AWS_MFA_SERIAL" --token-code "$CODE" \
      --duration-seconds 28800 \
      --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text)
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
fi

echo "Control Centre: http://localhost:${LOCAL_PORT}   (Ctrl+C to close)"
exec aws ssm start-session --region "$REGION" --target "$INSTANCE" \
  --document-name "$DOCUMENT" --parameters "localPortNumber=${LOCAL_PORT}"
