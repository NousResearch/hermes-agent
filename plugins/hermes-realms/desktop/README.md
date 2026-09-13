# Desktop half

The canonical `plugin.js` is compiled directly from this unified package by Hermes Desktop's bundled discovery. It also retains the plain ESM SDK-consumer form. It is **off by default**, independently of the profile's Python activation. Enable it in Settings → Plugins → Realms only after preparing the owning backend; see [setup and dependency](../README.md).

The session status stack and list/tile badges use each contribution's own runtime/stored identity, profile and connection. There is no global focused-session routing. APIs use the generic owner-scoped REST SDK; Watch uses the transient isolated-session preview and Pop out the native viewer SDK supplied by prerequisite #103690. Remote viewing is intentionally unsupported without a viewer tunnel.

Backend routes are `GET /realms?runtime_session_id=…&stored_session_id=…` and `POST /realms/{id}/watch`, namespaced by the SDK under `/api/plugins/hermes-realms`. Registration adds no poller or network request; mounted query components poll only after UI opt-in. See the host's `apps/desktop/src/contrib/bundled-realms.test.ts` for actual loader/toggle coverage. Full native desktop validation remains a separate approved Linux DEV lane.
