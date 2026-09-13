"""Run the supplier's dmgbuild with failure-time, read-only handle diagnostics."""
from __future__ import annotations

import functools
import json
import subprocess
import sys


def report_detach_failure(target, attachment, output, *, run=subprocess.run, stream=None):
    stream = sys.stderr if stream is None else stream

    def log(text):
        print(f"[dmg-detach] {text}", file=stream, flush=True)

    def invoke(argv):
        log("command: " + json.dumps(argv))
        try:
            result = run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=15)
        except subprocess.TimeoutExpired as error:
            for partial in (error.stdout, error.stderr):
                if partial:
                    log(partial.decode("utf-8", errors="replace") if isinstance(partial, bytes) else partial)
            log("diagnostic timed out after 15s. Snapshot is incomplete.")
            return None
        except OSError as error:
            log(f"diagnostic unavailable: {error}")
            return None
        if result.stdout:
            log(result.stdout.rstrip())
        if result.stderr:
            log(result.stderr.rstrip())
        log(f"diagnostic exit status: {result.returncode}")
        return result

    log(f"detach failed for {target}: {output}")
    log("Open-handle snapshot before dmgbuild retries or cleanup. No processes are stopped.")
    elevated = invoke(["/usr/bin/sudo", "-n", "/usr/bin/true"])
    prefix = ["/usr/bin/sudo", "-n"] if elevated is not None and elevated.returncode == 0 else []
    if not prefix:
        log("Using an unprivileged snapshot; system-process handles may be hidden.")
    entities = attachment.get("entities", []) if attachment else []
    mounts = list(dict.fromkeys(e["mount-point"] for e in entities if e.get("mount-point")))
    devices = list(dict.fromkeys([target, *(e["dev-entry"] for e in entities if e.get("dev-entry"))]))
    raw_devices = ["/dev/r" + device[5:] for device in devices if device.startswith("/dev/disk")]
    image = attachment.get("image") if attachment else None
    log("attachment: " + json.dumps({"image": image, "entities": entities}))
    command = [*prefix, "/usr/sbin/lsof", "-nP", "+c", "0", "-R"]
    for mount in mounts:
        # Select the filesystem, not a recursive stat of every bundled file.
        invoke([*command, "+f", "--", mount])
    invoke([*command, "-f", "--", *([image] if image else []), *devices, *raw_devices])
    log("End of handle snapshot. Errors or empty output do not prove the absence of holders.")


def wrap_hdiutil(native, *, report=report_detach_failure, stream=None):
    attachments = {}

    @functools.wraps(native)
    def observed(command, *args, **kwargs):
        result = native(command, *args, **kwargs)
        code, output = result
        if command == "attach" and code == 0 and isinstance(output, dict):
            attachment = {"image": args[-1], "entities": output.get("system-entities", [])}
            for entity in attachment["entities"]:
                for key in ("dev-entry", "mount-point"):
                    if entity.get(key):
                        attachments[entity[key]] = attachment
        elif command in ("detach", "unmount") and code != 0:
            target = next((arg for arg in reversed(args) if not arg.startswith("-")), "")
            try:
                report(target, attachments.get(target), output)
            except Exception as error:
                # Diagnostics must not replace the original detach failure.
                print(f"[dmg-detach] diagnostic failed: {error}",
                      file=sys.stderr if stream is None else stream, flush=True)
        return result

    return observed


def main():
    from dmgbuild import core
    from dmgbuild.__main__ import main as build

    core.hdiutil = wrap_hdiutil(core.hdiutil)
    build()


if __name__ == "__main__":
    main()
