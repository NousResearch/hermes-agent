# Bundled plugins

Drop a `<name>/plugin.{ts,tsx}` here that default-exports a `HermesPlugin` and
it registers automatically at boot (vite glob in `../contrib/plugins.ts`), with
the same inventory + live enable/disable contract as runtime plugins.

The bundled Kanban surface is an in-tree plugin because it is part of the
Desktop product. It uses the same `HermesPlugin`/`PluginContext` contract as
user- and agent-authored plugins; dashboard manifests and backend-served
Desktop assets are not a delivery mechanism.

User- and agent-authored plugins load at runtime from
`$HERMES_HOME/desktop-plugins/<name>/plugin.js` (the disk door) — see the
`hermes-desktop-plugins` skill.
