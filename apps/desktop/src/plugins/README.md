# Bundled plugins

Drop a `<name>/plugin.{ts,tsx}` here that default-exports a `RabbitPlugin` and
it registers automatically at boot (vite glob in `../contrib/plugins.ts`), with
the same inventory + live enable/disable contract as runtime plugins.

Keep this tree for real shipped plugins (and the small authoring fixtures that
dogfood the SDK). One-off demos that rebuild a core chrome piece 1:1 do not
belong here — they double the UI and confuse Capabilities ▸ Plugins. Publish those
in the companion
the example-plugins repo instead.

User- and agent-authored plugins load at runtime from
`$RABBIT_HOME/desktop-plugins/<name>/plugin.js` (the disk door) — see the
`rabbit-desktop-plugins` skill.
