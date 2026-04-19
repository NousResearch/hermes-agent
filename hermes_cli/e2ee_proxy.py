"""E2EE crypto primitives — encrypt/decrypt against an attested model's signing key.

Used by ``hermes_cli.e2ee_transport.E2EETransport`` (the in-process httpx transport
that intercepts outgoing chat completions). Supports ECDSA (SECP256K1 + ECDH +
HKDF + AES-GCM) and Ed25519 (converted to X25519 + NaCl Box).
"""
import secrets

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from nacl import bindings as _nacl
from nacl.public import Box, PrivateKey as _X25519Priv, PublicKey as _X25519Pub


def _generate_ecdsa_keypair():
    priv = ec.generate_private_key(ec.SECP256K1(), default_backend())
    pub_bytes = priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return priv, pub_bytes[1:].hex()  # strip 0x04 prefix


def _generate_ed25519_keypair():
    priv = ed25519.Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return priv, pub_bytes.hex()


def _encrypt_ecdsa(data: bytes, pub_hex: str) -> bytes:
    pub_bytes = bytes.fromhex(pub_hex)
    if len(pub_bytes) == 65 and pub_bytes[0] == 0x04:
        pub_bytes = pub_bytes[1:]
    pub_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), b"\x04" + pub_bytes)
    eph_priv = ec.generate_private_key(ec.SECP256K1(), default_backend())
    shared = eph_priv.exchange(ec.ECDH(), pub_key)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None,
        info=b"ecdsa_encryption", backend=default_backend(),
    ).derive(shared)
    nonce = secrets.token_bytes(12)
    ct = AESGCM(aes_key).encrypt(nonce, data, None)
    eph_pub = eph_priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return eph_pub + nonce + ct


def _decrypt_ecdsa(data: bytes, priv_key) -> bytes:
    eph_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), data[:65])
    shared = priv_key.exchange(ec.ECDH(), eph_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None,
        info=b"ecdsa_encryption", backend=default_backend(),
    ).derive(shared)
    return AESGCM(aes_key).decrypt(data[65:77], data[77:], None)


def _encrypt_ed25519(data: bytes, pub_hex: str) -> bytes:
    # Ed25519 pubkey → X25519 via NaCl, then NaCl Box (X25519 + ChaCha20-Poly1305)
    pub_bytes = bytes.fromhex(pub_hex)
    x25519_pub = _X25519Pub(_nacl.crypto_sign_ed25519_pk_to_curve25519(pub_bytes))
    eph_priv = _X25519Priv.generate()
    box = Box(eph_priv, x25519_pub)
    encrypted = box.encrypt(data)  # includes 24-byte nonce prepended by NaCl
    return bytes(eph_priv.public_key) + encrypted


def _decrypt_ed25519(data: bytes, priv_key) -> bytes:
    seed = priv_key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    pub = priv_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    x25519_priv = _X25519Priv(_nacl.crypto_sign_ed25519_sk_to_curve25519(seed + pub))
    eph_pub = _X25519Pub(data[:32])
    return Box(x25519_priv, eph_pub).decrypt(data[32:])


def _generate_keypair(algo: str):
    if algo == "ecdsa":
        return _generate_ecdsa_keypair()
    if algo == "ed25519":
        return _generate_ed25519_keypair()
    raise ValueError(f"Unsupported signing algo: {algo}")


def _encrypt_content(text: str, pub_hex: str, algo: str) -> str:
    if algo == "ecdsa":
        return _encrypt_ecdsa(text.encode(), pub_hex).hex()
    if algo == "ed25519":
        return _encrypt_ed25519(text.encode(), pub_hex).hex()
    raise ValueError(f"Unsupported signing algo: {algo}")


def _decrypt_content(hex_data: str, priv_key, algo: str) -> str:
    if algo == "ecdsa":
        return _decrypt_ecdsa(bytes.fromhex(hex_data), priv_key).decode()
    if algo == "ed25519":
        return _decrypt_ed25519(bytes.fromhex(hex_data), priv_key).decode()
    raise ValueError(f"Unsupported signing algo: {algo}")
