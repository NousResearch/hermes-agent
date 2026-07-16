# Durable Canvases

Hermes Desktop discovers durable Canvases as `*.canvas.json` files in the active profile directory. The Desktop client renders those manifests locally, so the report can be opened from a remote gateway without serving its HTML over HTTP.

## Try it

1. Build or run the Desktop branch containing the Canvases route.
2. In a chat, ask: `Create a Canvas with a searchable table of 10 green fruits.`
3. Open **Canvases** in the sidebar.

For every request containing `canvas` or `canvases`, Desktop injects the Canvas contract before the message is sent. This is enough to create, update, repair, and delete Canvas manifests; no gateway modification is required.

## Optional gateway skill

The client policy makes the feature work from Desktop. Install the included skill only when the same contract should also be available to other Hermes clients, CLI sessions, or messaging platforms:

```sh
mkdir -p ~/.hermes/skills
cp -R apps/desktop/skills/canvas ~/.hermes/skills/canvas
```

Restart the gateway (or start a new agent session) after copying it. The skill is profile-agnostic: it writes a Canvas under the active profile as `<id>.canvas.json`.

## Manifest contract

Every Canvas needs a valid JSON manifest with `schema`, `id`, `title`, `profile`, `intent`, `source`, `updatedAt`, `data`, and a free-form `document`. The document uses generic elements such as text, KPIs, tables, lists, charts, images, dividers, and callouts. Its content and layout are chosen from the user's request; there is no fixed report template.

Keep remote manifests below 460,000 UTF-8 bytes. Use summaries and bounded tables/lists for large sources, preserving refresh instructions in `source.instructions`.
