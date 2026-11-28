"""Replay fingerprint of a minimized webhook origin."""
import hashlib
import json


def native_digest(native):
    """Keep exact duplicate comparisons without retaining the source snapshot forever."""
    proof = {key: value for key, value in native.items() if key != 'webhook_target'}
    return hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
