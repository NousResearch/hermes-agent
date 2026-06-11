<!--
    Release note draft for Rust-backed install and packaging changes.
-->

# Rust Install Manager Release Note

Hermes desktop release packages now include a small Rust install manager binary.

User-visible behavior:

- Lite uninstall can remove the managed Hermes runtime checkout even when the Python environment is broken or missing.
- Fresh desktop/bootstrap-installer installs record manager metadata under `HERMES_HOME/manager/installed-files.json`.
- The Tauri bootstrap installer handles install state probing, downloaded artifact cache writes, bootstrap marker creation,
  and initial config template setup in Rust before falling back to script-backed language dependency stages.
- Release staging writes a checksummed bundled manifest beside the Rust manager binary.

Compatibility and fallback:

- Existing Python, PowerShell, shell, and Electron fallback paths remain available.
- User data under `HERMES_HOME` is preserved for lite uninstall.
- Python and Node dependency installation still use the existing scripts until a parity-tested Rust replacement exists.
- PATH/profile mutation and platform shortcut management remain script-backed in this slice.

Operational note:

- Release builds require Rust/Cargo on the build machine so the manager binary can be compiled into the packaged app.
- End users do not need Rust installed; published packages include the manager binary.
