"""Scanner-only reproducer for #105758; no JavaScript or shell fixture is executed.

The two launcher texts match TypeScript 5.9.3's published bin/tsc and lib/tsc.js.
The compiler is represented by inert text with the operator-reported byte size.
"""

from pathlib import Path

import pytest

from cron.lifecycle_guard import scan_gateway_lifecycle


@pytest.mark.platforms("posix")
@pytest.mark.xfail(
    strict=True,
    reason="Known #105758: the required compiler module exceeds the unchanged 1 MiB scan cap",
)
def test_typescript_launcher_chain_is_not_a_lifecycle_command(tmp_path):
    package = tmp_path / "node_modules" / "typescript"
    (package / "bin").mkdir(parents=True)
    (package / "lib").mkdir()
    (tmp_path / "node_modules" / ".bin").mkdir()
    (package / "bin" / "tsc").write_text(
        "#!/usr/bin/env node\nrequire('../lib/tsc.js')\n", encoding="utf-8"
    )
    (package / "lib" / "tsc.js").write_text(
        "// This file is a shim which defers loading the real module until the compile cache is enabled.\n"
        "try {\n"
        '  const { enableCompileCache } = require("node:module");\n'
        "  if (enableCompileCache) {\n"
        "    enableCompileCache();\n"
        "  }\n"
        "} catch {}\n"
        'module.exports = require("./_tsc.js");\n',
        encoding="utf-8",
    )
    (package / "lib" / "_tsc.js").write_text(" " * 6_213_092, encoding="utf-8")
    (tmp_path / "node_modules" / ".bin" / "tsc").symlink_to("../typescript/bin/tsc")
    reads = []

    def missing_remote(path):
        reads.append(path)
        return None

    result = scan_gateway_lifecycle(
        "./node_modules/.bin/tsc --noEmit -p apps/app/tsconfig.json",
        cwd=str(tmp_path),
        read_remote_script=missing_remote,
    )
    assert result == (False, None), (result, reads)


@pytest.mark.parametrize("remote_only", [False, True], ids=["local", "remote"])
@pytest.mark.parametrize(
    ("module_text", "blocked", "size_refusal"),
    [
        ("module.exports = {};\n", False, False),
        ('require("node:child_process").execSync("hermes gateway stop");\n', True, False),
        (" " * 1_048_577, True, True),
        (
            " " * 1_048_577
            + '\nrequire("node:child_process").execSync("hermes gateway stop");\n',
            True,
            True,
        ),
    ],
    ids=["benign", "lifecycle", "oversized-benign", "oversized-lifecycle"],
)
def test_required_module_is_executable_not_inert(
    tmp_path, remote_only, module_text, blocked, size_refusal
):
    """A runtime shebang must not hide an executed dependency, including remote-only code.

    Source files are only read by the scanner; neither Node nor a shell executes them.
    The over-cap twins demonstrate why treating require() as inert loses protection.
    """
    entry = tmp_path / "entry"
    shim = tmp_path / "shim.js"
    module = tmp_path / "module.js"
    sources = {
        str(entry): '#!/usr/bin/env node\nrequire("./shim.js");\n',
        str(shim): 'module.exports = require("./module.js");\n',
        str(module): module_text,
    }
    if not remote_only:
        for path, text in sources.items():
            Path(path).write_text(text, encoding="utf-8")
    reads = []

    def read_remote(path):
        reads.append(path)
        return sources.get(path)

    unsafe, refusal = scan_gateway_lifecycle(
        str(entry), cwd=str(tmp_path), read_remote_script=read_remote
    )
    assert unsafe is blocked
    if size_refusal:
        assert refusal is not None and str(module) in refusal and "scan cap" in refusal
    else:
        assert refusal is None
    assert reads == list(sources) if remote_only else reads == []


@pytest.mark.parametrize("shell", ["bash", "source", "."])
def test_explicit_shell_keeps_nested_scan_despite_runtime_shebang(tmp_path, shell):
    entry = tmp_path / "entry"
    child = tmp_path / "child"
    entry.write_text("#!/usr/bin/env node\n./child\n", encoding="utf-8")
    child.write_text("hermes gateway stop\n", encoding="utf-8")
    assert scan_gateway_lifecycle(f"{shell} {entry}", cwd=str(tmp_path)) == (True, None)
