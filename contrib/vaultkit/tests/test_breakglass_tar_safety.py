"""Regression test for AIR-style High finding: breakglass tar extraction must
defeat path-traversal members (CVE-2007-4559). The bundle is stored on
untrusted media by design, so a tampered tar with a '../' member must NOT
escape the destination dir on restore.
"""
import sys, os, tarfile, tempfile, io
from pathlib import Path
from importlib.machinery import SourceFileLoader
import importlib.util

KIT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, f"{KIT}/lib")
_l = SourceFileLoader("vbg", f"{KIT}/bin/vault-breakglass")
_s = importlib.util.spec_from_loader("vbg", _l)
vbg = importlib.util.module_from_spec(_s); _l.exec_module(vbg)

fails = []

# 1. A malicious tar with a parent-traversal member must NOT write outside dest.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    dest = td / "dest"; dest.mkdir()
    outside = td / "PWNED_should_not_exist"
    tar_path = td / "evil.tar"
    with tarfile.open(tar_path, "w") as t:
        data = b"owned"
        info = tarfile.TarInfo(name="../PWNED_should_not_exist")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    blocked = False
    try:
        with tarfile.open(tar_path, "r") as t:
            vbg._safe_extractall(t, dest)
    except SystemExit:
        blocked = True   # vk.die path (manual fallback) — acceptable
    except Exception:
        blocked = True   # data filter raises FilterError — acceptable
    if outside.exists():
        fails.append("TRAVERSAL ESCAPED: ../PWNED file was written outside dest")
    else:
        print("1. parent-traversal member did NOT escape dest OK",
              "(blocked)" if blocked else "(filtered/skipped)")

# 2. An absolute-path member must also not escape.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    dest = td / "dest"; dest.mkdir()
    sentinel = td / "abs_target"
    tar_path = td / "abs.tar"
    with tarfile.open(tar_path, "w") as t:
        data = b"x"
        info = tarfile.TarInfo(name=str(sentinel))  # absolute path member
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    try:
        with tarfile.open(tar_path, "r") as t:
            vbg._safe_extractall(t, dest)
    except (SystemExit, Exception):
        pass
    if sentinel.exists():
        fails.append("ABSOLUTE-PATH member escaped dest")
    else:
        print("2. absolute-path member did NOT escape dest OK")

# 3. A LEGITIMATE bundle (normal members) still extracts correctly.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    dest = td / "dest"; dest.mkdir()
    tar_path = td / "good.tar"
    with tarfile.open(tar_path, "w") as t:
        for name, content in (("vault.kdbx", b"KDBX"), ("vault.key", b"KEY"), ("config", b"CFG")):
            data = content
            info = tarfile.TarInfo(name=name); info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    with tarfile.open(tar_path, "r") as t:
        vbg._safe_extractall(t, dest)
    ok = all((dest / n).exists() for n in ("vault.kdbx", "vault.key", "config"))
    if not ok:
        fails.append("legitimate bundle did NOT extract cleanly")
    else:
        print("3. legitimate bundle extracts normally OK")

if fails:
    print("\nFAIL:")
    for f in fails: print("  -", f)
    sys.exit(1)
print("\nALL TAR-SAFETY TESTS PASSED")
