# Running updater compatibility

An updater can retain old Python modules after replacing its checkout. Its next
lazy import reads the new files. Current-tree tests alone cannot protect that
boundary.

`old_updater_surface.json` records historical imports union current imports.
Regenerate it from a full clone:

```sh
python3 scripts/audit-old-updater-imports.py --freeze tests/compat/old_updater_surface.json
```

The walker pins `origin/main` once. The JSON records that exact commit, the
history roots, discovered entrypoints, selected paths, and parse recoveries.
It inventories all reachable commits, including merge parents. It follows
historical filenames, deleted helpers, extraction imports, and rename edges.
A shallow clone cannot regenerate the file. Shallow CI checks the frozen names
and that the current imports remain a subset of the freeze.

The surface deliberately over-approximates reachability. Matching update
entrypoints and same-named functions can include unrelated callers. Do not trim
those entries by hand. A historical name is cheaper than a missed live import.
The resolver requires module-scope bindings, not names inside functions or
classes. It does not execute module bodies or prove every conditional export.

## Runtime behavior

Retired dependency entrypoints stop with a relaunch message. Returning `None`
can activate old pip fallback code. Other shims retain conservative return
shapes without installs, downloads, marker writes, or PM delegation. Each shim
explains why it exists. Current application code must use the live implementation,
not these historical exports.

A fresh source launch checks PM's existing successful dependency facts. It
synchronizes stale state and switches to the managed interpreter before
activating application dependencies. No new incomplete-marker protocol is used.

`old_updater_dependencies.py` retains historical caller functions with their
lazy imports intact. The shim tests execute those callers against the new tree.
`tests/pm/test_source_update_launch.py` exercises real worker publication,
failed-build retention, and fresh-process bootstrap.

## Manual review of dynamic edges

Complete history enumeration is not a complete Python call-graph proof. The
JSON retains `unresolved_dynamic` rather than silently dropping those edges.
The reviewed categories are:

- `hermes_constants` reloads: fixed first-party module, present and audited.
  Reload execution still depends on the running interpreter and process state.
- `managed_scope`, `main_dashboard`, and browser module objects: fixed modules
  passed between helpers. The modules exist. Arbitrary attribute dispatch is
  not a statically proven contract.
- `managed_uv` and `_subprocess_compat`: the historical Windows installer calls
  `_subprocess_compat.run`. `REVIEWED_DYNAMIC_LOADS` retains that named call with
  its witness commit. Its shim stops before invoking an installer.
- PM operations and build operations: fixed dispatch modules, seeded explicitly
  with the other post-swap helpers. PM registry import names can also come from
  package definitions outside the checkout.
- Plugin configuration hooks and deferred tool registration: import targets
  depend on enabled plugins and runtime manifests. A core-tree freeze cannot
  enumerate names supplied by independently installed plugins.

These limits remain visible in the generated file. The static gate proves that
its recorded imports still resolve structurally. Real updater and launch tests
prove the exercised runtime paths, not every historical platform combination.
