# Docker images & ECR

Images: `nova-control-plane`, `nova-runtime`. Every image must be reproducible, versioned,
scanned, secret-free, non-root where practical, and have a HEALTHCHECK.

## Tag vs digest
- Tag for humans: `nova-control-plane:0.1.0-g<shortsha>`
- Digest for deployment: `<registry>/nova-control-plane@sha256:<digest>`
Deployment config references digests. A mutable tag can be moved by anyone with push rights;
a digest cannot. Enable ECR tag immutability on customer repositories where practical.

## Release pipeline (owned by nova-docker-release)
1. build (pinned base image by digest, `--pull`, build args contain no secrets)
2. test (unit tests + container starts + healthcheck passes locally)
3. scan (ECR scan on push or local scanner) — HIGH/CRITICAL findings block release
4. tag `<version>-g<sha>`
5. push
6. obtain digest: `aws ecr describe-images --repository-name <repo> --image-ids imageTag=<tag> --query 'imageDetails[0].imageDigest'`
7. update deployment config with the digest (record previous digest in journal for rollback)
8. terraform plan (if digest is a Terraform input) → verify no instance replacement
9. deploy via SSM (pull by digest, restart container)
10. verify runtime (`nova verify` ≥ level 6)

Secret check before push: `docker history --no-trunc` and inspect `ENV`/labels; no AWS keys,
tokens or `.aws` directories in any layer. Credentials at runtime come from the instance role
via IMDSv2.
