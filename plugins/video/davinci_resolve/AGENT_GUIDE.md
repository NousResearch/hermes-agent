# Agent Guide

You are controlling DaVinci Resolve through either Studio live control or free Resolve interchange files.

Start every unfamiliar session with:

1. `resolve_capabilities`
2. `resolve_launch`
3. `resolve_probe`
4. If `resolve_probe.recommended_mode` is `studio_live_control`, call `resolve_project_summary`.
5. If `resolve_probe.recommended_mode` is `free_interchange`, do not call live-control tools. Generate import files instead.

For changes, use dry-run first. Only call a mutating tool with `dry_run=false` when the user has approved the plan and you also set `confirm=true`.

Good first real tests:

- Add a marker to the active timeline.
- Import one known media file into a test bin.
- Append one known media file to the active timeline.
- Scan a small media folder, dry-run a scripted timeline, then render only after the user approves.

Do not delete anything unless the user explicitly asks for deletion.

## Studio Scripted Edit Mode

When the user asks for a finished edit from a script and folder of clips:

1. Scan source folders with `resolve_scan_media_folder`.
2. Interpret the script yourself: identify beats, required visuals, clip order, approximate clip ranges, music mood, and markers.
3. Build a structured `clips` array for `resolve_create_scripted_timeline`.
4. Dry-run `resolve_create_scripted_timeline` and explain the edit plan.
5. Only after approval, call `resolve_create_scripted_timeline` with `dry_run=false` and `confirm=true`.
6. Dry-run `resolve_render_timeline`, then render with `dry_run=false` and `confirm=true`.
7. Poll `resolve_render_status` and report the final output location.

Use `start_frame` and `end_frame` for source ranges when the user or your analysis gives ranges. Use `record_frame`, `track_index`, and `media_type` when placing clips on specific tracks. Add markers for script beats, missing-shot notes, or review points.

Do not claim final color, sound mix, effects, or advanced editorial polish unless the plan explicitly encodes those operations or the user asks for a rough cut.

## Free Resolve Mode

Free DaVinci Resolve can import timelines, but it does not reliably expose the external `scriptapp("Resolve")` bridge. When `resolve_probe` reports `free_interchange`, use:

- `resolve_generate_fcpxml_timeline` for timeline assembly from media paths.
- `resolve_generate_marker_csv` for marker manifests.

Tell the user to import generated FCPXML with **File > Import > Timeline > Import AAF, EDL, XML...**. Do not claim that Hermes can inspect the current free Resolve project live.

## Resolve 20 And Resolve 21 Beta

If the user has both Resolve 20 and Resolve 21 beta installed, ask which version should open before using `resolve_launch`.

Use:

- `variant="resolve20"` or `variant="studio20"` for Resolve 20 installs.
- `variant="resolve21"`, `variant="studio21"`, or `variant="beta"` for Resolve 21 beta installs.
- `app_path="/Applications/.../DaVinci Resolve.app"` when the exact side-by-side app bundle path is known.

Do not open production projects in Resolve 21 beta unless the user confirms they have backups. Projects opened in a beta may not be usable again in earlier Resolve versions.
