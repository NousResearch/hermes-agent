"""Independent CUA process fixture shared by selection and doctor callers."""
import sys
from pathlib import Path


def record_driver(version="0.20.0", *, manifest=False):
    from pm import Facts, Lockfile, current_target, get_package, paths
    from pm.store import tree_digest

    package, target, root = get_package("cua-driver"), current_target(), paths.store_root()
    entry = root / package.store_entry(version, target)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True, exist_ok=True)
    if manifest:
        capabilities = {
            "mcp": ["--socket", "--grant"],
            "serve": ["--socket", "--permission-mode", "--capability-manifest",
                      "--approve-capability-manifest", "--embedded"],
            "stop": ["--socket"],
        }
        payload = {
            "binary_version": version,
            "mcp_invocation": {"command": str(binary), "args": ["mcp"]},
            "subcommands": [{"name": name, "args": [{"name": arg} for arg in args]}
                            for name, args in capabilities.items()],
        }
        binary.write_text(f"#!{sys.executable}\nimport json, sys\nmanifest = {payload!r}\n"
                          "if sys.argv[1:] == ['manifest']:\n    print(json.dumps(manifest))\n"
                          "else:\n    assert sys.argv[1] in ('status', 'stop')\n", encoding="utf-8")
    else:
        binary.write_bytes(Path(sys.executable).read_bytes())
    binary.chmod(0o755)
    artifact = {"url": "https://example.invalid/cua-fixture", "sha256": "a" * 64}
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin(package.name, version, {target: artifact})
    lock.save()
    Facts(paths.facts_path()).record(
        package.name, version, entry.name, package.env(entry, target), root,
        target=target, artifacts=[artifact["sha256"]], digest=tree_digest(entry),
    )
    return binary
