# Railway container storage

- Status: active
- Scope: `Dockerfile`, `deploy/railway/README.md`
- Introduced: Railway deployment

## Downstream intent

Railway rejects Dockerfile `VOLUME` declarations. Keep `/opt/data` as the state
directory, but require an explicit platform volume or Docker mount. Existing
Compose mounts remain authoritative. Bare `docker run` without a mount has
ephemeral state and creates no anonymous volume.

## Reconciliation

Do not restore upstream's `VOLUME` declaration. Configure Railway Dockerfile
paths, start commands and restart policy through service settings; the platform
rejects the former config-file setting.

## Validation

Build on Railway, verify `/opt/data` is mounted, and confirm saved state survives
a redeploy. Docker tests needing a volume must mount one explicitly.
