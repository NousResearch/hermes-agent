"""Real Linux venv/XDG I/O probe; help-launch is NOT a native UI proof."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import venv


def run(argv, env, cwd):
    result = subprocess.run(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=60)
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-sha", help="Provenance for a git-archived checkout")
    args = parser.parse_args()
    repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="linux-launcher-", dir=args.output))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(root / "home"),
           "HERMES_HOME": str(root / "home/.hermes"),
           "XDG_DATA_HOME": str(root / "xdg"), "LANG": "C.UTF-8",
           "HERMES_NONINTERACTIVE": "1"}
    Path(env["HERMES_HOME"]).mkdir(parents=True)
    venv_dir = root / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(venv_dir)
    python = venv_dir / "bin/python"
    # Reuse dependencies read-only; put the tested checkout before its editable install.
    site = next(venv_dir.glob("lib/python*/site-packages"))
    dependencies = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    (site / "probe.pth").write_text(str(repo) + "\n" + str(dependencies) + "\n")
    assert python.is_symlink() and python.resolve() != python
    install = (
        "import sys,json; from pathlib import Path; "
        "from hermes_cli.linux_desktop_entry import install_desktop_entry; "
        f"sys.argv[0]={str(repo / 'hermes')!r}; "
        f"p=install_desktop_entry(Path({str(repo)!r})); "
        "print(json.dumps({'path':str(p),'text':p.read_text(),'python':sys.executable}))"
    )
    rows: dict = {"source_sha": args.source_sha or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "root": str(root), "lexical_python": str(python), "base_python": str(python.resolve())}
    rows["venv_import"] = run([str(python), "-I", "-c", "import yaml,hermes_cli.main; print(yaml.__version__)"], env, "/")
    rows["base_import_negative"] = run([str(python.resolve()), "-I", "-c", "import yaml,hermes_cli.main"], env, "/")
    rows["install"] = run([str(python), "-c", install], env, "/")
    assert rows["install"]["returncode"] == 0, rows["install"]
    installed = json.loads(rows["install"]["stdout"].splitlines()[-1])
    entry = Path(installed["path"])
    exec_line = next(x[5:] for x in installed["text"].splitlines() if x.startswith("Exec="))
    argv = shlex.split(exec_line)
    assert str(python.resolve()) not in argv
    # Exercise the generated command's real CLI imports without invoking npm/build.
    rows["generated_exec_help"] = run([*argv, "--help"], env, "/")
    original = entry.read_bytes()
    rows["second_install"] = run([str(python), "-c", install], env, "/")
    rows["stable_rewrite"] = entry.read_bytes() == original
    custom = original.replace(b"Name=Hermes\n", b"Name=Hermes custom\n").replace(b"Terminal=false", b"Terminal=true")
    entry.write_bytes(custom)
    config = Path(env["HERMES_HOME"]) / "config.yaml"
    config.write_text("desktop:\n  manage_launcher_entry: false\n")
    rows["custom_install"] = run([str(python), "-c", install], env, "/")
    rows["custom_preserved"] = entry.read_bytes() == custom
    rows["generated_exec"] = exec_line
    rows["fidelity"] = "real venv, native Linux XDG installer, real generated argv --help; NOT Electron window/menu-click proof"
    (args.output / "probe.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    assert rows["venv_import"]["returncode"] == 0
    assert rows["base_import_negative"]["returncode"] != 0
    assert rows["generated_exec_help"]["returncode"] == 0
    assert rows["stable_rewrite"]


if __name__ == "__main__":
    main()
