## Tasker Overlay Scenes

This release adds a bounded Tasker scenes parity slice for Hermes Android.

- Adds `create_overlay_scene_task`, `show_overlay_scene`,
  `hide_overlay_scene`, and `overlay_scene_status` automation actions.
- Lets saved automations show a title/text/button overlay panel after the user
  grants Android draw-over-other-apps permission.
- Supports top, center, and bottom scene positions with capped width and
  auto-dismiss duration.
- Keeps the feature permission-gated and data-only; Hermes does not import or
  execute arbitrary Tasker scene code.
