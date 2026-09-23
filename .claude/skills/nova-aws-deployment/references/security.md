# Security review checklist (used by nova-safety-reviewer)

For any plan, policy, image, bundle or command set, check:

- **Destruction/replacement**: any delete or replace? Of a protected type? Requested explicitly?
- **Persistent state**: EBS state volume, KMS keys (and key policies/deletion windows), S3 bundle
  data, application state — untouched or covered by a tested snapshot/restore?
- **IAM**: new actions? `Resource: "*"`? wildcard actions (`bedrock:*`, `s3:*`, `iam:*`)?
  `iam:PassRole` scope? Trust policy changes? Role used by a running instance modified?
- **Networking**: new ingress? `0.0.0.0/0` or `::/0`? port 22 or control-plane ports? public IP on
  runtime? route table / NAT / endpoint changes?
- **Encryption**: EBS encrypted with the environment KMS key? S3 SSE-KMS and public access block?
- **Secrets**: none in Terraform vars/state outputs, images, bundles, Git, logs, SSM command text.
- **Images**: referenced by digest? scanned? non-root? healthcheck?
- **Auditability**: change recorded in journal with plan hash / digest / bundle sha?
- **Reversibility**: rollback path known and recorded before apply?
- **Intent match**: does the diff contain *only* what the requested change should produce?

Verdicts: `APPROVE` (no findings), `APPROVE_WITH_NOTES`, `NEEDS_HUMAN` (any destructive, IAM
widening, network exposure, or intent mismatch), `REJECT` (violates a non-negotiable).
