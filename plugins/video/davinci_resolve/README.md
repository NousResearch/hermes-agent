# Hermes DaVinci Resolve Plugin

This is a native Hermes Agent plugin adapter for DaVinci Resolve.

It supports two modes:

- **Studio live control**: DaVinci Resolve Studio exposes the local Python scripting API, so Hermes can inspect and modify a running project.
- **Free Resolve interchange**: free DaVinci Resolve does not expose reliable external live control, so Hermes generates importable files such as FCPXML and marker CSV manifests.

## Install Locally

Copy this directory to:

```bash
~/.hermes/plugins/davinci-resolve
```

Then restart Hermes. Hermes should discover the plugin and expose the `davinciresolve` toolset.

## Tools

- `resolve_capabilities`: explain how an LLM agent should use the Resolve tools.
- `resolve_launch`: open DaVinci Resolve locally and check scripting reachability.
- `resolve_probe`: read-only scripting diagnostic.
- `resolve_project_summary`: read-only project/timeline summary.
- `resolve_import_media`: import files into the media pool.
- `resolve_create_timeline`: create an empty or media-seeded timeline.
- `resolve_append_to_current_timeline`: import and append media to the active timeline.
- `resolve_add_timeline_marker`: add a marker to the active timeline.
- `resolve_scan_media_folder`: scan a footage/music/stills folder for media Hermes can plan with.
- `resolve_create_scripted_timeline`: build a Studio timeline from a structured edit plan with source ranges, target tracks, music, and markers.
- `resolve_render_timeline`: create and optionally start a render job for the current Studio timeline.
- `resolve_render_status`: check render queue and progress.
- `resolve_generate_fcpxml_timeline`: generate an FCPXML timeline file for free Resolve.
- `resolve_generate_marker_csv`: generate a marker CSV/manifest for free Resolve workflows.

Mutating tools default to `dry_run=true`. To actually modify a Resolve project, call the tool with `dry_run=false` and `confirm=true`.

## Requirements

- DaVinci Resolve installed locally.
- Resolve Studio open when using live-control tools.
- In Studio, **DaVinci Resolve > Preferences > System > General > External scripting using: Local** must be enabled.
- Resolve scripting module available at a standard path or configured through `RESOLVE_SCRIPT_API` for live-control tools.
- For free Resolve, use interchange tools and import generated files manually with Resolve's import menus.

## Studio Scripted Edit Workflow

For a prompt like "take this folder of clips, follow this script, add music like X, and give me a final QuickTime", Hermes should:

1. `resolve_launch` with `variant="studio"`.
2. `resolve_probe` and continue only if `recommended_mode="studio_live_control"`.
3. `resolve_scan_media_folder` on footage and music folders.
4. Read the user's script and choose clips/ranges/music as a structured edit plan.
5. Call `resolve_create_scripted_timeline` with `dry_run=true`.
6. After approval, call the same tool with `dry_run=false` and `confirm=true`.
7. Call `resolve_render_timeline` with `dry_run=true`, then with `dry_run=false` and `confirm=true`.
8. Poll `resolve_render_status` until rendering finishes.

The plugin does not make creative choices by itself. Hermes makes those choices from the user's script and available media, then passes a structured plan to Resolve.

## Free Resolve Workflow

If `resolve_probe` reports `recommended_mode="free_interchange"`, use:

1. `resolve_generate_fcpxml_timeline` to create a `.fcpxml` file.
2. In Resolve, choose **File > Import > Timeline > Import AAF, EDL, XML...**.
3. Select the generated file from `~/Documents/Hermes Resolve Exports` or the requested `output_path`.

Free Resolve workflows cannot read the current project state through Hermes. They are file-based handoffs into Resolve.

## Resolve 20 And Resolve 21 Beta

`resolve_launch` supports `variant` values for `resolve20`, `studio20`, `resolve21`, `studio21`, and `beta`. If multiple Resolve versions are installed with custom app names, pass `app_path` to launch the exact `.app` bundle.
