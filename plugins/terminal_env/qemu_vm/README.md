# qemu-vm — QEMU/KVM terminal backend

A persistent, per-profile Debian virtual machine used as the agent's terminal backend.
Commands, file tools and `execute_code` run inside the guest; the host filesystem is
unreachable. Compared with the Docker backend this gives **kernel-level** isolation, and
unlike Modal/Daytona/Vercel it needs no cloud account.

```
hermes vm create                          # download pinned image, provision, boot (~2-5 min)
hermes config set terminal.backend qemu_vm
hermes                                    # the agent now works inside the VM
```

`hermes setup terminal` → *QEMU VM* does the same interactively.

## How it works

| Piece | Choice |
|---|---|
| Hypervisor | `qemu-system-x86_64` with KVM, headless (`-display none`), no libvirt |
| Image | Debian 13 `genericcloud` qcow2, **URL + SHA512 pinned in code** (`manager.IMAGES`) |
| Disk | Full copy per profile (`qemu-img convert` + `resize`); no backing-file overlays |
| First boot | cloud-init NoCloud seed ISO: user, ed25519 key, `AcceptEnv *`, git/curl/python3 |
| Access | SSH forwarded on `127.0.0.1:<random port>` only; per-VM `known_hosts`, never `~/.ssh` |
| Environment | Built-in `SSHEnvironment` subclass → file sync, persistent shell, env passthrough |
| State | `<HERMES_HOME>/sandboxes/qemu_vm/` (profile-aware); image cache shared under the Hermes root |

## Settings

`plugins.entries.terminal_env/qemu_vm.settings.<key>` in `config.yaml`:

| Key | Default | When applied |
|---|---|---|
| `cpus` | `2` | every start |
| `memory_mb` | `2048` | every start |
| `network` | `egress` | every start — `egress` = NAT out, no inbound; `none` = no outbound at all |
| `disk_gb` | `20` | `create` |
| `user` | `hermes` | `create` |
| `passwordless_sudo` | `true` | `create` |
| `autostart` | `true` | a stopped VM boots on the first terminal command |
| `boot_timeout` | `180` | seconds to wait for SSH |

`hermes vm create --disk 40 --cpus 4 --memory 4096 --network none` overrides for one VM.

## CLI

```
hermes vm create [--disk GB] [--cpus N] [--memory MB] [--network egress|none] [--force]
hermes vm start | stop | status [--json] | destroy [--yes] | ssh [--print]
```

The agent has no tool for these verbs; only the user manages the VM.

## Requirements

Linux x86_64 with `/dev/kvm` writable by your user, `qemu-system-x86`, `qemu-utils`,
`genisoimage` (or `xorriso` / `cloud-image-utils`) and `openssh-client`. TCG (no KVM) is
refused on purpose: it is too slow to be a usable backend.

## Security notes

- Network `egress` uses QEMU user-mode networking: the guest can reach whatever the host can
  (including the LAN and `10.0.2.2` = host). Use `network: none` for an air-gapped sandbox.
- `~/.hermes` sync into the guest follows the SSH backend's rules (skills, credential files
  declared by skills, media cache); `.env` and `auth.json` are never synced.
- Dangerous-command approval prompts are skipped inside the VM (same policy as Docker): the
  host cannot be damaged, only the VM disk, which `hermes vm create --force` rebuilds.
