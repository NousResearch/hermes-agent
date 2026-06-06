# Example: Linux Multiboot / GRUB A2O Fragment

```md
# All To One: openSUSE Tumbleweed Boot Recovery

## 1. One-Sentence Result
openSUSE Tumbleweed is bootable through Ubuntu GRUB by chainloading the openSUSE EFI loader at `/EFI/systemd/shim.efi`. [screenshot]

## 2. Background and Goal
- Original goal: install and boot openSUSE alongside Ubuntu and Windows.
- Constraint: no USB repair route preferred; user was operating from installed Linux systems.
- Final route: use Ubuntu GRUB as the top-level boot menu and add a custom openSUSE chainloader entry.

## 3. Final System State
- EFI partition: FAT32 UUID `B5BB-052E`, mounted at `/boot/efi`. [screenshot]
- Ubuntu root: ext4 UUID `a9d28905-777f-4f9b-a110-6b6fa4e72215`. [screenshot]
- openSUSE EFI files exist under `/boot/efi/EFI/systemd/`: `shim.efi`, `grub.efi`, `BOOT.CSV`, `installed_by_sdbootutil`. [screenshot]
- Boot route: firmware → Ubuntu GRUB → custom menuentry → `/EFI/systemd/shim.efi`. [observed]

## 4. Real Timeline
| Phase | Action / Symptom | Judgment at the Time | Later Confirmed | Result |
|---|---|---|---|---|
| Install | openSUSE installed but did not appear as boot option | Bootloader might be missing | EFI files existed, NVRAM entry was missing | Chainload from Ubuntu GRUB |
| GRUB edit | Edited `/etc/grub.d/40_custom` | Needed custom menuentry | ISO installer entry was obsolete | Added installed system entry |
| Cleanup | User asked whether to delete ISO entry | ISO entry could cause confusion | Installed system should use EFI loader, not installer ISO | Remove ISO entry |

## 6. Bug, Pitfall, Root Cause
| Problem | Symptom | Root Cause | Fix | How to Recognize Next Time | Evidence |
|---|---|---|---|---|---|
| openSUSE not in boot menu | Cannot boot directly | EFI loader exists but firmware/NVRAM entry missing | Ubuntu GRUB chainloader | `/EFI/systemd/shim.efi` exists but no firmware entry | [screenshot] |
| Installer entry confusion | GRUB entry boots installer ISO | Old DVD ISO menuentry remained | Delete ISO entry | menuentry references `/home/...openSUSE...iso` | [screenshot] |

## 10. Resume in 5-10 Minutes
1. Check `lsblk -f` for EFI UUID.
2. Check `/boot/efi/EFI/systemd/shim.efi` exists.
3. Check `/etc/grub.d/40_custom` has an openSUSE chainloader entry.
4. Run `sudo update-grub`.
5. If `shim.efi` fails, try `/EFI/systemd/grub.efi`.
6. Do not format EFI or touch BitLocker partitions.
```
