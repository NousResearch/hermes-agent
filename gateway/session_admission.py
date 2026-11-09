"""Canonical payload identity after authorization and immutable media capture.

The digest is defined storage-side (``hermes_state_keys``) so the canonical store never imports
``gateway/``; this module keeps the gateway-facing name.
"""
from hermes_state_keys import admission_fingerprint

__all__ = ['admission_fingerprint']
