"""Create a writable overlay without copying or mutating Nix dependencies."""

from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
from importlib.metadata import distributions


def main():
    root = Path(sys.argv[1])
    target = root / ".venv"
    if target.is_symlink():
        raise SystemExit("Refusing to modify a symlinked .venv; move it aside before entering the Nix shell.")
    if target.exists() and not (target / "pyvenv.cfg").is_file():
        raise SystemExit("Existing .venv is not a virtual environment; move it aside before entering the Nix shell.")

    # Re-entry refreshes the interpreter after a Nix update but keeps local installs.
    subprocess.run(
        [sys.argv[3], "venv", "--quiet", "--offline", "--no-project", "--no-config",
         "--allow-existing", "--no-python-downloads", "--python", sys.executable, str(target)],
        check=True,
    )
    python = target / "bin/python"
    relative_site = Path(sysconfig.get_path("purelib")).relative_to(sys.prefix)
    site = target / relative_site
    # addsitedir processes the editable workspace's .pth files as well. Local
    # site-packages is already on sys.path and takes precedence over Nix's deps.
    (site / "_hermes_nix.pth").write_text(
        f"import site; site.addsitedir({sysconfig.get_path('purelib')!r})\n"
    )

    # Store entry points have store-Python shebangs. Seed local launchers so
    # pytest/hermes-acp/etc see the same overrides as python and HERMES_PYTHON.
    # Resolve the entry point at runtime so a locally upgraded package wins.
    for dist in distributions():
        for entry in dist.entry_points:
            if entry.group != "console_scripts":
                continue
            script = target / "bin" / entry.name
            if script.exists():
                continue
            script.write_text(
                f"#!{sys.argv[2]}\n"
                f"'''exec' {shlex.quote(str(python))} \"$0\" \"$@\"\n"
                "' '''\n"
                "import sys\n"
                "from importlib.metadata import distribution\n"
                f"entry = next(e for e in distribution({dist.metadata['Name']!r}).entry_points "
                f"if e.group == 'console_scripts' and e.name == {entry.name!r})\n"
                "sys.exit(entry.load()())\n"
            )
            script.chmod(0o755)


if __name__ == "__main__":
    main()
