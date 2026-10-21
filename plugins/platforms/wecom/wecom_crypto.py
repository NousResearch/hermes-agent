"""WeCom callback-mode AES-CBC crypto, wire-compatible with Tencent's official ``WXBizMsgCrypt`` SDK."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import socket
import struct
from typing import Optional
from xml.etree import ElementTree as ET

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class WeComCryptoError(Exception):
    pass


class SignatureError(WeComCryptoError): pass


class DecryptError(WeComCryptoError): pass


class EncryptError(WeComCryptoError): pass


class PKCS7Encoder:
    block_size = 32

    @classmethod
    def encode(cls, text: bytes) -> bytes:
        amount_to_pad = cls.block_size - (len(text) % cls.block_size) or cls.block_size
        return text + bytes([amount_to_pad]) * amount_to_pad

    @classmethod
    def decode(cls, decrypted: bytes) -> bytes:
        if not decrypted:
            raise DecryptError("empty decrypted payload")
        pad = decrypted[-1]
        if pad < 1 or pad > cls.block_size:
            raise DecryptError("invalid PKCS7 padding")
        if decrypted[-pad:] != bytes([pad]) * pad:
            raise DecryptError("malformed PKCS7 padding")
        return decrypted[:-pad]


def _sha1_signature(token: str, timestamp: str, nonce: str, encrypt: str) -> str:
    return hashlib.sha1("".join(sorted([token, timestamp, nonce, encrypt])).encode("utf-8")).hexdigest()


class WXBizMsgCrypt:
    """Minimal WeCom callback crypto helper compatible with BizMsgCrypt semantics."""

    def __init__(self, token: str, encoding_aes_key: str, receive_id: str):
        for bad, message in (
            (not token, "token is required"), (not encoding_aes_key, "encoding_aes_key is required"),
            (len(encoding_aes_key) != 43, "encoding_aes_key must be 43 chars"), (not receive_id, "receive_id is required"),
        ):
            if bad:
                raise ValueError(message)
        self.token, self.receive_id = token, receive_id
        self.key = base64.b64decode(encoding_aes_key + "=")
        self.iv = self.key[:16]

    def _cipher(self) -> Cipher:
        return Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=default_backend())

    def verify_url(self, msg_signature: str, timestamp: str, nonce: str, echostr: str) -> str:
        return self.decrypt(msg_signature, timestamp, nonce, echostr).decode("utf-8")

    def decrypt(self, msg_signature: str, timestamp: str, nonce: str, encrypt: str) -> bytes:
        if _sha1_signature(self.token, timestamp, nonce, encrypt) != msg_signature:
            raise SignatureError("signature mismatch")
        try:
            cipher_text = base64.b64decode(encrypt)
        except Exception as exc:
            raise DecryptError(f"invalid base64 payload: {exc}") from exc
        try:
            decryptor = self._cipher().decryptor()
            content = PKCS7Encoder.decode(decryptor.update(cipher_text) + decryptor.finalize())[16:]  # skip 16-byte random prefix
            xml_length = socket.ntohl(struct.unpack("I", content[:4])[0])
            xml_content, receive_id = content[4:4 + xml_length], content[4 + xml_length:].decode("utf-8")
        except WeComCryptoError:
            raise
        except Exception as exc:
            raise DecryptError(f"decrypt failed: {exc}") from exc
        if receive_id != self.receive_id:
            raise DecryptError("receive_id mismatch")
        return xml_content

    def encrypt(self, plaintext: str, nonce: Optional[str] = None, timestamp: Optional[str] = None) -> str:
        nonce = nonce or self._random_nonce()
        timestamp = timestamp or str(int(__import__("time").time()))
        encrypt = self._encrypt_bytes(plaintext.encode("utf-8"))
        root = ET.Element("xml")
        for tag, text in (("Encrypt", encrypt), ("MsgSignature", _sha1_signature(self.token, timestamp, nonce, encrypt)), ("TimeStamp", timestamp), ("Nonce", nonce)):
            ET.SubElement(root, tag).text = text
        return ET.tostring(root, encoding="unicode")

    def _encrypt_bytes(self, raw: bytes) -> str:
        try:
            payload = os.urandom(16) + struct.pack("I", socket.htonl(len(raw))) + raw + self.receive_id.encode("utf-8")
            encryptor = self._cipher().encryptor()
            return base64.b64encode(encryptor.update(PKCS7Encoder.encode(payload)) + encryptor.finalize()).decode("utf-8")
        except Exception as exc:
            raise EncryptError(f"encrypt failed: {exc}") from exc

    @staticmethod
    def _random_nonce(length: int = 10) -> str:
        return "".join(secrets.choice("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length))
