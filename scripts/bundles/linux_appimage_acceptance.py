"""Check one producer-digested x64 AppImage in the matching Ubuntu userland.

    python3 -I scripts/bundles/linux_appimage_acceptance.py --ubuntu 22.04 --cpu-model stories260K.gguf --sha256 DIGEST FILE.AppImage
    python3 -I scripts/bundles/linux_appimage_acceptance.py --ubuntu 24.04 --sha256 DIGEST FILE.AppImage

Requires python3 >= 3.10, bubblewrap (user, PID and network namespaces), strace,
and room in TMPDIR for the image and one full extraction. The 22.04 leg also
needs xvfb, xauth, Electron's system libraries and a working Chromium userns
sandbox. No FUSE, GPU, host Node, credentials or privileged host changes.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


def check_image(image: Path, sha256: str) -> None:
    with image.open("rb") as stream:
        header = stream.read(64)
        if (header[:6] != b"\x7fELF\x02\x01" or header[8:11] != b"AI\x02"
                or header[18:20] != b"\x3e\x00"):
            raise ValueError("expected a packaged x86-64 type 2 AppImage")
        digest = hashlib.sha256(header)
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError("AppImage SHA-256 does not match the producer")


def run_checks(ubuntu: str, image: Path, work: Path, cpu_model: Path | None) -> None:
    scripts = Path(__file__).resolve().parent
    cmd = ["/usr/bin/bwrap", "--unshare-user", "--uid", "1000", "--gid", "1000",
           "--unshare-net", "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--new-session",
           "--cap-drop", "ALL", "--proc", "/proc", "--dev", "/dev",
           "--bind", str(work / "tmp"), "/tmp"]  # no-tmp: ok — private X11 socket and AppImage extraction directory
    for system in ("/usr", "/bin", "/lib", "/lib64", "/sbin", "/etc/fonts", "/etc/ld.so.cache",
                   "/etc/alternatives", "/etc/nsswitch.conf", "/etc/os-release", "/etc/localtime"):
        if Path(system).exists():
            cmd += ["--ro-bind", system, system]
    for name, record in (("passwd", "smoke:x:1000:1000:Smoke:/work/home:/bin/bash"), ("group", "smoke:x:1000:")):
        path = work.parent / name
        path.write_text(record + "\n", encoding="ascii")
        cmd += ["--ro-bind", str(path), f"/etc/{name}"]
    cmd += ["--ro-bind", str(image), "/image.AppImage", "--bind", str(work), "/work",
            "--ro-bind", str(scripts / "appimage_checks.py"), "/checks.py",
            "--ro-bind", str(scripts.parent / "smoke-payload.sh"), "/smoke-payload.sh",
            "--chdir", "/work"]
    if ubuntu == "22.04":
        if cpu_model is None or not cpu_model.is_file():
            raise ValueError("the floor check requires the pinned CPU smoke model")
        cmd += ["--ro-bind", str(cpu_model.resolve()), "/stories260K.gguf"]
        cmd += ["/usr/bin/xvfb-run", "-a", "-s", "-screen 0 1280x800x24 -nolisten tcp"]
    cmd += ["/usr/bin/python3", "-I", "/checks.py", ubuntu]
    for directory in ("home", "runtime", "tmp"):
        (work / directory).mkdir(mode=0o700)
    env = {"PATH": "/usr/bin:/bin", "HOME": "/work/home", "XDG_RUNTIME_DIR": "/work/runtime",
           "HERMES_HOME": "/work/home/.hermes", "TMPDIR": "/work/tmp", "LANG": "C.UTF-8",
           "HERMES_DISABLE_LAZY_INSTALLS": "1", "UV_OFFLINE": "1", "npm_config_offline": "true"}
    subprocess.run(cmd, stdin=subprocess.DEVNULL, env=env, check=True, timeout=1500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ubuntu", choices=("22.04", "24.04"), required=True)
    parser.add_argument("--sha256", required=True, help="SHA-256 recorded before artifact transport")
    parser.add_argument("--cpu-model", type=Path, help="pinned stories260K.gguf for the offline 22.04 test")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    release = platform.freedesktop_os_release()
    if (release.get("ID"), release.get("VERSION_ID"), platform.machine()) != ("ubuntu", args.ubuntu, "x86_64"):
        parser.error(f"this leg requires Ubuntu {args.ubuntu} x86_64")
    if args.image.suffix != ".AppImage" or not args.image.is_file():
        parser.error("expected one regular .AppImage file, not an unpacked app directory")
    with tempfile.TemporaryDirectory(prefix="appimage-accept-") as temp:
        image = Path(temp) / "image.AppImage"
        shutil.copyfile(args.image, image)
        image.chmod(0o500)
        check_image(image, args.sha256)
        work = Path(temp) / "work"
        work.mkdir(mode=0o700)
        run_checks(args.ubuntu, image, work, args.cpu_model)
    print(f"Ubuntu {args.ubuntu} x64 AppImage accepted: sha256={args.sha256}")


if __name__ == "__main__":
    main()
