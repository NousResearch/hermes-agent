# OpenManus plugin boundary

This folder is the Hermes adapter for the pinned `vendor/openmanus` submodule.
Keep OpenManus upstream code unchanged here; adapter behaviour belongs in the
plugin files and Hermes configuration.

The plugin is opt-in. Hermes and MoA agents may use the `openmanus` toolset,
but live execution must remain behind the dry-run default, configured
workspace confinement, explicit side-effect acknowledgement, and the network
permission gate. Do not pass the full Hermes environment or local browser
sessions to the child process.

The child runner copies the submodule into a per-run directory and writes
redacted receipts under the active Hermes home. Keep generated checkouts and
receipts outside publication commits. Changes to OpenManus itself must be
made upstream or through a separately reviewed submodule update.
