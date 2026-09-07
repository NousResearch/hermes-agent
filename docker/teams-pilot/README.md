# Teams pilot image

The `v2026.8.31` image bundles a Teams SDK pin (`2.0.13.4`) whose exact
MSAL and cryptography requirements conflict. This image keeps that exact
Hermes base and changes only the Teams SDK pin to Microsoft's stable
`2.0.16`, installing its missing dependencies. Existing base packages are
not downgraded, and Teams JWT verification is not disabled.

Build from this worktree:

```sh
docker build -t hermes-agent-teams-pilot:v2026.8.31-sdk2.0.16 docker/teams-pilot
```

The final build step checks real SDK imports and the Hermes lazy-dependency
contract. Runtime verification must additionally check `/health`, an unsigned
`POST /api/messages` returning 401, and an authorized message from Teams.

The local Compose deployment supplies the credentials, persistent state and
workspaces. None is copied into this image. Its only public ingress is the
separate Teams webhook via a Microsoft Dev Tunnel; the dashboard stays private.

This is a local pilot image, not an upstream Hermes release. Remove this
overlay when a verified official image provides compatible Teams dependencies.
