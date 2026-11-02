"""Opt-in Linux s6 launch enrollment. No drain, replacement or runtime acceptance.

The native service manager enrolls only a new, persistently down slot. The launcher
validates it after UID drop, then execs the ordinary gateway without --replace.
Same-UID filesystem cooperation is required; this is not an OS isolation boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ENROLLMENT = "launch-admission.json"


def _uid() -> int:
    if sys.platform != "linux":
        raise ValueError("Linux s6 admission required")
    return os.getuid()  # windows-footgun: ok -- guarded Linux-only admission


def _identity(path: Path) -> list[int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or path != path.resolve():
        raise ValueError("noncanonical directory")
    return [info.st_dev, info.st_ino]


def _launcher(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 65536:
            raise ValueError("invalid launcher")
        return hashlib.sha256(stream.read(65537)).hexdigest()


def _home(profile: str) -> Path:
    from hermes_cli.profiles import get_profile_dir

    return get_profile_dir(profile).resolve(strict=True)


def enroll(temporary_slot: Path, final_slot: Path, profile: str, *, launch_guard=None) -> None:
    """Called before native registration publishes the stopped service directory."""
    from hermes_cli.service_manager import _HERMES_UID, _chown_hermes, validate_profile_name

    validate_profile_name(profile)
    if not (temporary_slot / "down").is_file():
        raise ValueError("persistent down enrollment required")
    home = _home(profile)
    if home.stat().st_uid not in {_uid(), _HERMES_UID}:
        raise ValueError("profile owner mismatch")
    value = {"version": 1, "slot": str(final_slot), "slot_identity": _identity(temporary_slot),
             "profile": profile, "home": str(home), "home_identity": _identity(home),
             "uid": _HERMES_UID if _uid() == 0 else _uid(),
             "launcher_sha256": _launcher(temporary_slot / "run")}
    if launch_guard is not None:
        from hermes_cli.service_manager_launch_guard import guard_lock, read_journal
        with guard_lock(launch_guard, home):
            read_journal(launch_guard, home)
        value["launch_guard"] = launch_guard
    path = temporary_slot / ENROLLMENT
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _chown_hermes(path)


def _process(pid: int) -> dict[str, int | str]:
    # Linux start ticks + boot ID disambiguate PID reuse; no argv matching.
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8-sig")
    fields = raw[raw.rindex(")") + 2:].split()
    if fields[0] == "Z":
        raise ValueError("dead owner")
    return {"pid": pid, "parent": int(fields[1]), "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8-sig").strip()}


def admission(slot: Path, profile: str) -> dict:
    from hermes_cli.service_manager import _s6_run

    if sys.platform != "linux" or _uid() == 0:
        raise ValueError("after UID drop required")
    fd = os.open(slot / ENROLLMENT, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8-sig") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != _uid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 8192):
            raise ValueError("unsafe enrollment")
        value = json.loads(stream.read(8193))
    home = _home(profile)
    expected = {"version": 1, "slot": str(slot), "slot_identity": _identity(slot),
                "profile": profile, "home": str(home), "home_identity": _identity(home),
                "uid": _uid(), "launcher_sha256": _launcher(slot / "run")}
    if "launch_guard" in value:
        from hermes_cli.service_manager_launch_guard import reference
        reference(value["launch_guard"], home)
        expected["launch_guard"] = value["launch_guard"]
    if value != expected:
        raise ValueError("enrollment mismatch")
    current = _process(os.getpid())
    parent = _process(os.getppid())
    if (current["parent"] != parent["pid"]
            or not Path(f'/proc/{parent["pid"]}/exe').samefile('/command/s6-supervise')
            or Path(f'/proc/{parent["pid"]}/cwd').resolve() != slot):
        raise ValueError("supervisor mismatch")
    # s6 publishes status just after fork. Bound waiting only observes this same child.
    for _ in range(20):
        result = _s6_run("s6-svstat", "-o", "pid", str(slot))
        if result.returncode == 0 and result.stdout.strip() == str(os.getpid()):
            break
        time.sleep(0.05)
    else:
        raise ValueError("current run mismatch")
    if _process(os.getppid()) != parent or _process(os.getpid()) != current:
        raise ValueError("owner changed")
    result = {"launch": current, "supervisor": parent, "slot_identity": expected["slot_identity"],
            "home_identity": expected["home_identity"], "launch_profile": profile,
            "runtime_profile_binding_verified": False, "behavior_accepted": False}
    if "launch_guard" in expected:
        result["launch_guard"] = expected["launch_guard"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    try:
        command = [sys.executable, "-m", "hermes_cli.main"]
        if args.profile != "default":
            command += ["-p", args.profile]
        command += ["gateway", "run"]
        while True:
            receipt = admission(args.slot, args.profile)
            guard = receipt.get("launch_guard")
            if guard is None:
                os.execv(sys.executable, command)
            from hermes_cli.service_manager_launch_guard import guard_lock, held
            with guard_lock(guard, _home(args.profile)):
                if not held(guard, _home(args.profile), args.slot):
                    # Keep the original transaction lock through exec, never read-then-release.
                    os.execv(sys.executable, command)
            time.sleep(0.25)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print('{"error":"native_launch_admission_refused","behavior_accepted":false}', flush=True)
        return 78
    return 78


def observed_admission(home: Path) -> dict | None:
    """Return only kernel-revalidated enrollment for this answering launch process.

    Legacy/missing/unknown enrollment is absent, never positive admission evidence.
    This does not establish multiplex served-profile authority or idle drain safety.
    """
    if sys.platform != "linux":
        return None
    try:
        slot = Path(f"/proc/{os.getppid()}/cwd").resolve(strict=True)
        path = slot / ENROLLMENT
        if not path.is_file():
            return None
        # Determine the label from native enrollment, then fully revalidate it and home.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8-sig") as stream:
            value = json.loads(stream.read(8193))
        result = admission(slot, value["profile"])
        if _identity(home) != result["home_identity"]:
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
