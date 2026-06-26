import sys, os, tempfile, subprocess, tarfile, importlib
from pathlib import Path
from importlib.machinery import SourceFileLoader
import importlib.util

KIT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, f"{KIT}/lib")


def load_vbg():
    l = SourceFileLoader("vbg", f"{KIT}/bin/vault-breakglass")
    s = importlib.util.spec_from_loader("vbg", l)
    m = importlib.util.module_from_spec(s); l.exec_module(m)
    return m


class _BlockCryptography:
    """Context manager that makes `import cryptography...` raise ImportError,
    simulating a machine where the package isn't installed."""
    def __enter__(self):
        self._saved = {k: v for k, v in sys.modules.items() if k.startswith("cryptography")}
        for k in list(sys.modules):
            if k.startswith("cryptography"):
                del sys.modules[k]
        self._finder = self
        sys.meta_path.insert(0, self)
        return self
    def find_spec(self, name, path, target=None):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("cryptography blocked for fallback test")
        return None
    def __exit__(self, *a):
        sys.meta_path.remove(self)
        sys.modules.update(self._saved)


vbg = load_vbg()
words = vbg._load_wordlist()
import secrets as _s
ent = _s.token_bytes(32)

# 1. Both backends produce identical identity/recipient for the same scalar.
scalar = vbg._derive_age_identity  # via the real derive path
ident_c, recip_c = vbg._derive_age_identity(ent)
# Force pure-python by blocking cryptography and re-deriving.
with _BlockCryptography():
    pub_pp, backend = vbg._pubkey_from_scalar(
        __import__("hashlib").pbkdf2_hmac("sha256", ent, b"vaultkit-age-v1", 2048, dklen=32))
    ident_pp, recip_pp = vbg._derive_age_identity(ent)
print("backend when blocked:", backend)
assert backend == "pure-python", backend
assert ident_c == ident_pp, "identity differs between backends!"
assert recip_c == recip_pp, "recipient differs between backends!"
print("1. cryptography and pure-python backends yield IDENTICAL age keys OK")

# 2. Cross-backend interop: encrypt with cryptography present, decrypt with it
#    BLOCKED (the real fallback recovery scenario).
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    plain = td / "vault.kdbx"; plain.write_bytes(b"FAKE-KDBX-CONTENT-12345")
    bundle = td / "bg.age"
    # encrypt to recipient derived WITH cryptography
    r = subprocess.run(["age", "-r", recip_c, "-o", str(bundle), str(plain)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # restore: derive identity with cryptography BLOCKED, decrypt
    with _BlockCryptography():
        vbg2 = load_vbg()  # reload so it re-resolves backend under the block
        ent2 = vbg2._mnemonic_to_entropy(vbg2._entropy_to_mnemonic(ent, words), words)
        ident2, _ = vbg2._derive_age_identity(ent2)
    idf = td / "id"; idf.write_text(ident2 + "\n"); os.chmod(idf, 0o600)
    out = td / "out.bin"
    r = subprocess.run(["age", "-d", "-i", str(idf), "-o", str(out), str(bundle)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"decrypt failed: {r.stderr}"
    assert out.read_bytes() == b"FAKE-KDBX-CONTENT-12345", "content mismatch"
print("2. bundle encrypted WITH cryptography decrypts WITHOUT it (fallback works) OK")

# 3. Full reverse: encrypt with cryptography blocked, decrypt with it present.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    plain = td / "v.kdbx"; plain.write_bytes(b"REVERSE-TEST")
    with _BlockCryptography():
        vbg3 = load_vbg()
        _, recip_pp2 = vbg3._derive_age_identity(ent)
    bundle = td / "b.age"
    r = subprocess.run(["age", "-r", recip_pp2, "-o", str(bundle), str(plain)],
                       capture_output=True, text=True); assert r.returncode == 0, r.stderr
    ident_c2, _ = vbg._derive_age_identity(ent)  # cryptography present
    idf = td / "id"; idf.write_text(ident_c2 + "\n"); os.chmod(idf, 0o600)
    out = td / "o.bin"
    r = subprocess.run(["age", "-d", "-i", str(idf), "-o", str(out), str(bundle)],
                       capture_output=True, text=True); assert r.returncode == 0, r.stderr
    assert out.read_bytes() == b"REVERSE-TEST"
print("3. reverse interop (encrypt w/o crypto, decrypt w/ crypto) OK")

print("\nALL FALLBACK TESTS PASSED")
