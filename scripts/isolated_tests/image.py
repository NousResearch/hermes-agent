"""Offline image and daemon metadata checks, run before any payload.

The image is still a trusted pinned toolchain: these checks only refuse the
metadata shapes that would defeat the runner's own controls (anonymous
writable VOLUMEs, loader/interpreter ENV that `/usr/bin/env` would honour
before `env -i` clears the test shell's environment). They do not audit the
image's files.
"""

from __future__ import annotations

import json
import re

from isolated_tests.inputs import Refused
from isolated_tests.policy import IMAGE_ID_RE, sha256_json

# ENV keys a dynamic loader, libc or interpreter reads at startup.
_LOADER_RISK = re.compile(r"^(LD_|DYLD_|GCONV_PATH$|PYTHON|LOCPATH$|NLSPATH$|MALLOC_|GLIBC_TUNABLES$"
                          r"|HOSTALIASES$|RESOLV_HOST_CONF$|BASH_ENV$|ENV$|IFS$|PERL|RUBY|NODE_OPTIONS$)")
_ENV_ALLOWED = frozenset({"PATH", "LANG", "LC_ALL", "TZ", "HOME", "SSL_CERT_FILE", "SSL_CERT_DIR"})


def check_image_inspect(raw: bytes, image: str) -> dict:
    """Return the normalized metadata to bind; refuse anything unexpected."""
    try:
        doc = json.loads(raw)
    except ValueError:
        raise Refused("image inspect: output is not JSON") from None
    if isinstance(doc, list) and len(doc) == 1:
        doc = doc[0]
    if not isinstance(doc, dict) or doc.get("Id") != image or not IMAGE_ID_RE.match(image):
        raise Refused(f"image inspect: Id does not equal requested image {image!r}")
    cfg = doc.get("Config")
    if not isinstance(cfg, dict):
        raise Refused("image inspect: Config missing")
    if cfg.get("Volumes"):
        raise Refused(f"image: declares VOLUME {sorted(cfg['Volumes'])}; anonymous volumes are writable")
    env = cfg.get("Env") or []
    if not isinstance(env, list) or not all(isinstance(e, str) and "=" in e for e in env):
        raise Refused("image inspect: Config.Env malformed")
    keys = sorted(e.split("=", 1)[0] for e in env)
    risky = [k for k in keys if _LOADER_RISK.match(k) or k not in _ENV_ALLOWED]
    if risky:
        raise Refused(f"image: ENV keys outside the allowlist / loader-risk: {risky}")
    normalized = {"Id": doc["Id"], "Env": sorted(env), "Volumes": None,
                  "Os": doc.get("Os"), "Architecture": doc.get("Architecture"),
                  "RootFS": doc.get("RootFS"), "User": cfg.get("User")}
    return {"image_id": doc["Id"], "metadata_sha256": sha256_json(normalized)}


def check_server_version(raw: bytes) -> dict:
    """The daemon facts bound into the attestation; change => re-prove."""
    try:
        doc = json.loads(raw)
    except ValueError:
        raise Refused("docker version: output is not JSON") from None
    if not isinstance(doc, dict):
        raise Refused("docker version: unexpected shape")
    fields = {k: doc.get(k) for k in ("Version", "Os", "Arch", "KernelVersion")}
    if not all(isinstance(v, str) and v for v in fields.values()):
        raise Refused(f"docker version: missing server fields {sorted(k for k, v in fields.items() if not v)}")
    if fields["Os"] != "linux":
        raise Refused(f"docker server Os {fields['Os']!r}: only Linux containers are supported")
    return fields
