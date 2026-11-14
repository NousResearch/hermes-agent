"""The packaged Linux app fails its build when any ELF file needs a newer glibc than the arch floor."""
from __future__ import annotations

import struct

from scripts.bundles.glibc_floor import main, scan, version_needs

_VERNEED = 0x6FFFFFFE


def write_elf(path, needs, *, wide=True, order="<"):
    """A minimal ELF whose .gnu.version_r names ``needs``: {library: [version, ...]}."""
    strings, offsets = bytearray(b"\0"), {}
    for name in [*needs, *(version for versions in needs.values() for version in versions)]:
        offsets.setdefault(name, len(strings))
        strings += name.encode() + b"\0"
    table = bytearray()
    for index, (library, versions) in enumerate(needs.items()):
        last = index == len(needs) - 1
        table += struct.pack(order + "HHIII", 1, len(versions), offsets[library], 16,
                             0 if last else 16 + 16 * len(versions))
        for slot, version in enumerate(versions):
            table += struct.pack(order + "IHHII", 0, 0, slot + 2, offsets[version],
                                 0 if slot == len(versions) - 1 else 16)
    header_size, section_layout = (64, order + "IIQQQQIIQQ") if wide else (52, order + "IIIIIIIIII")
    strings_at = header_size
    table_at = strings_at + len(strings)
    sections_at = table_at + len(table)
    sections = [
        struct.pack(section_layout, *[0] * 10),
        struct.pack(section_layout, 0, 3, 0, 0, strings_at, len(strings), 0, 0, 1, 0),
        struct.pack(section_layout, 0, _VERNEED, 0, 0, table_at, len(table), 1, len(needs), 4, 0),
    ]
    ident = b"\x7fELF" + bytes([2 if wide else 1, 1 if order == "<" else 2, 1]) + bytes(9)
    if wide:
        header = struct.pack(order + "16sHHIQQQIHHHHHH", ident, 3, 0xB7, 1, 0, 0, sections_at, 0,
                             64, 56, 0, 64, len(sections), 0)
    else:
        header = struct.pack(order + "16sHHIIIIIHHHHHH", ident, 3, 0x28, 1, 0, 0, sections_at, 0,
                             52, 32, 0, 40, len(sections), 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + strings + table + b"".join(sections))
    return path


def violations(app, arch):
    return {(row.path, row.need) for row in scan(app, arch)[1]}


def test_needs_are_read_from_every_library_in_either_elf_class_and_byte_order(tmp_path):
    needs = {"libc.so.6": ["GLIBC_2.17", "GLIBC_2.34"], "libstdc++.so.6": ["GLIBCXX_3.4.9", "CXXABI_1.3"]}
    for wide, order in ((True, "<"), (True, ">"), (False, "<"), (False, ">")):
        elf = write_elf(tmp_path / f"{wide}{order}.so", needs, wide=wide, order=order)
        assert version_needs(elf) == {"GLIBC_2.17", "GLIBC_2.34", "GLIBCXX_3.4.9", "CXXABI_1.3"}
    (tmp_path / "notes.txt").write_text("\x7fEL")
    assert version_needs(tmp_path / "notes.txt") is None


def test_each_arch_fails_on_a_need_above_its_own_floor(tmp_path):
    app = tmp_path / "linux-unpacked"
    write_elf(app / "hermes", {"libc.so.6": ["GLIBC_2.25"], "libstdc++.so.6": ["GLIBCXX_3.4.9"]})
    payload = app / "resources/agent-payload/venv/lib/python3.11/site-packages"
    write_elf(payload / "olm.abi3.so", {"libc.so.6": ["GLIBC_2.38"], "libstdc++.so.6": ["GLIBCXX_3.4.32"]})

    assert violations(app, "arm64") == set()
    assert violations(app, "x64") == {
        ("resources/agent-payload/venv/lib/python3.11/site-packages/olm.abi3.so", "GLIBC_2.38"),
        ("resources/agent-payload/venv/lib/python3.11/site-packages/olm.abi3.so", "GLIBCXX_3.4.32"),
    }
    write_elf(payload / "newer.so", {"libc.so.6": ["GLIBC_2.40"], "libstdc++.so.6": ["GLIBCXX_3.4.34"]})
    assert violations(app, "arm64") == {
        ("resources/agent-payload/venv/lib/python3.11/site-packages/newer.so", "GLIBC_2.40"),
        ("resources/agent-payload/venv/lib/python3.11/site-packages/newer.so", "GLIBCXX_3.4.34"),
    }


def test_x64_gpu_engines_have_their_own_ceiling_and_nothing_else_does(tmp_path):
    app = tmp_path / "linux-unpacked"
    tools = app / "resources/agent-payload/tools"
    upstream = {"libc.so.6": ["GLIBC_2.38"], "libstdc++.so.6": ["GLIBCXX_3.4.32"]}
    for engine in ("llamacpp-cuda-12144-linux-x64", "llamacpp-hip-12144-linux-x64"):
        write_elf(tools / engine / "libllama.so", upstream)
    assert violations(app, "x64") == set()

    write_elf(tools / "llamacpp-cpu-12144-linux-x64/libllama.so", upstream)
    write_elf(tools / "llamacpp-cuda-12144-linux-x64/llama-server", {"libc.so.6": ["GLIBC_2.39"]})
    write_elf(app / "resources/agent-payload/hermes-agent/llamacpp-cuda-1-linux-x64/x.so", upstream)
    assert violations(app, "x64") == {
        ("resources/agent-payload/tools/llamacpp-cpu-12144-linux-x64/libllama.so", "GLIBC_2.38"),
        ("resources/agent-payload/tools/llamacpp-cpu-12144-linux-x64/libllama.so", "GLIBCXX_3.4.32"),
        ("resources/agent-payload/tools/llamacpp-cuda-12144-linux-x64/llama-server", "GLIBC_2.39"),
        ("resources/agent-payload/hermes-agent/llamacpp-cuda-1-linux-x64/x.so", "GLIBC_2.38"),
        ("resources/agent-payload/hermes-agent/llamacpp-cuda-1-linux-x64/x.so", "GLIBCXX_3.4.32"),
    }


def test_links_and_private_versions_are_not_requirements(tmp_path):
    app = tmp_path / "linux-arm64-unpacked"
    target = write_elf(tmp_path / "outside.so", {"libc.so.6": ["GLIBC_2.99"]})
    (app / "lib").mkdir(parents=True)
    (app / "lib/linked.so").symlink_to(target)
    write_elf(app / "lib/private.so", {"libc.so.6": ["GLIBC_PRIVATE", "GLIBC_2.17"]})

    assert scan(app, "arm64") == (1, [])


def test_a_file_whose_needs_cannot_be_read_fails_instead_of_passing(tmp_path, capsys):
    app = tmp_path / "linux-unpacked"
    # The loader reads needs through the dynamic segment, so an executable runs without a section table.
    stripped = write_elf(app / "sstripped", {"libc.so.6": ["GLIBC_2.99"]})
    data = bytearray(stripped.read_bytes())
    struct.pack_into("<Q", data, 0x28, 0)
    stripped.write_bytes(data)
    truncated = write_elf(app / "lib/truncated.so", {"libc.so.6": ["GLIBC_2.17"]})
    truncated.write_bytes(truncated.read_bytes()[:-8])
    write_elf(app / "lib/ok.so", {"libc.so.6": ["GLIBC_2.17"]})

    assert main(["--arch", "arm64", str(app)]) == 1
    out = capsys.readouterr().out
    assert "::error::sstripped: can't read its version needs (no ELF section headers)" in out
    assert "::error::lib/truncated.so: can't read its version needs (truncated ELF file)" in out
    assert "lib/ok.so" not in out
    assert "3 ELF files" in out


def test_the_command_fails_the_build_and_names_each_requirement(tmp_path, capsys):
    app = tmp_path / "linux-unpacked"
    write_elf(app / "chrome-sandbox", {"libc.so.6": ["GLIBC_2.36"]})

    assert main(["--arch", "arm64", str(app)]) == 0
    assert main(["--arch", "x64", str(app)]) == 1
    out = capsys.readouterr().out
    assert "::error::chrome-sandbox needs GLIBC_2.36 (above GLIBC_2.35)" in out
    assert "1 ELF files" in out
