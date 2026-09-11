# Vendored `omarchy vm`

`omarchy-vm` in this directory is [omacom/omarchy PR #10977](https://github.com/omacom/omarchy/pull/10977)'s
`bin/omarchy-vm`, which adds a disposable Omarchy guest in QEMU/KVM to Omarchy
itself. It is vendored rather than depended on so Realms works before that PR
merges and on machines whose Omarchy predates it; a system `omarchy vm` can be
preferred instead via `plugins.realms.vm.omarchy_vm_path`.

The guest it builds is installed by Omarchy's own GPG-verified release ISO. We
do not build, host or modify an image.

| | |
|---|---|
| Upstream source | `omacom/omarchy` PR #10977, `bin/omarchy-vm` |
| Upstream SHA-256 | `19a51517c2713e033b2f703bfa387a79a9880c1c7a918e327e1327cfde47c3b6` |
| Vendored SHA-256 | `realms.vm_manager.VENDORED_SHA256` |
| Licence | MIT, as the rest of Omarchy |

`vm_manager` re-checks the vendored digest before every invocation, so a
tampered or silently upgraded copy is never executed. The pin lives in Python,
not in a checksum file beside the script: a checksum an attacker can rewrite
pins nothing. After editing the script, update `VENDORED_SHA256`.

## Local patches

Four, each because upstream's single-user assumptions do not hold when one
machine runs a guest per conversation. Nothing else is changed.

**1. Per-session unit, sockets and ISO directory.** Upstream hardcodes the unit
name `omarchy-vm`, one QMP socket path and an ISO directory inside the VM's own
home. Realms runs one guest per conversation, so all three become environment
overrides (`OMARCHY_VM_UNIT`, `OMARCHY_VM_QMP_SOCKET`, `OMARCHY_VM_VNC_SOCKET`,
`OMARCHY_VM_ISO_DIR`) that default to upstream's values. The ISO directory moves
out of the per-session home so one 5 GB download is shared by every clone rather
than re-fetched per chat.

**2. Headless.** `graphics_args()` upstream opens an SDL window, which would
appear on the user's own desktop — precisely what a realm exists to avoid. The
vendored copy always renders to a software framebuffer with `-display none` and
serves RFB on a mode-0600 unix socket, which is both what the existing noVNC
viewer already speaks and what lets QMP `screendump` work at all.

**3. No package installation.** `cmd_install` upstream calls `omarchy-pkg-add`
to install `qemu-full edk2-ovmf mtools`. The plugin never installs system
packages; the vendored copy only *checks* for them with `pacman -Q`, and
`hermes realms vm doctor` reports what is missing for the user to install.

**4. An optional netdev suffix.** `OMARCHY_VM_NETDEV_EXTRA` is appended to
QEMU's `-netdev` so `plugins.realms.vm.network: false` can pass `restrict=on`,
which refuses the guest's outbound routes while keeping the SSH forward that
reaches it.

## Updating

1. Fetch the new upstream `bin/omarchy-vm` and record its SHA-256 here.
2. Re-apply the four patches above; each is marked `# hermes-realms:` in place.
3. Set `VENDORED_SHA256` in `realms/vm_manager.py` to the new digest.
4. Run `scripts/run_tests.sh tests/plugins/test_bundled_realms_vm.py` — the
   headless and parameterisation patches are covered by behaviour tests that
   execute the script's own dispatcher, so a dropped patch fails there.
5. Rebuild a base image and start one realm: the install path is not covered by
   the unit tests.
